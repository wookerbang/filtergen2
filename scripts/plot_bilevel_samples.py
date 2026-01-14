"""Randomly sample validation jsonl and compare target vs predicted S21 for bilevel model.

Outputs:
  - Per-sample PNG plots (target vs predicted S21)
  - JSONL summary with predicted/ground-truth topologies
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import MACRO_LIBRARY, SERIES_MACROS, dsl_tokens_to_macro_sequence
from src.models import Wave2StructureModel
from src.physics.differentiable_rf import DynamicCircuitAssembler, unroll_refine_slots


def _infer_dataset_field(samples: List[dict], key: str):
    vals = {s.get(key) for s in samples if s.get(key) is not None}
    if not vals:
        return None
    if len(vals) > 1:
        print(f"[warn] dataset has multiple {key} values; using an arbitrary one.", flush=True)
    return next(iter(vals))


class BilevelPlotDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        *,
        use_wave: str = "ideal",
        mix_real_prob: float = 0.3,
        normalize_wave: bool = False,
        freq_mode: str = "log_fc",
        freq_scale: str = "none",
        include_s11: bool = False,
    ) -> None:
        self.samples: List[dict] = []
        self.macro_ir_macros: List[List[str]] = []
        with open(jsonl_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                macros = sample.get("macro_ir_macros") or []
                if not macros:
                    tokens = sample.get("dsl_tokens") or []
                    if not tokens:
                        raise ValueError(f"Missing macro_ir_macros/dsl_tokens at line {line_no} in {jsonl_path}.")
                    macros = dsl_tokens_to_macro_sequence(tokens, strict=True)
                if not macros:
                    raise ValueError(f"Empty macro sequence at line {line_no} in {jsonl_path}.")
                self.samples.append(sample)
                self.macro_ir_macros.append(macros)
        self.use_wave = use_wave
        self.mix_real_prob = mix_real_prob
        self.normalize_wave = normalize_wave
        self.freq_mode = freq_mode
        self.freq_scale = freq_scale
        self.include_s11 = include_s11
        self.q_L = _infer_dataset_field(self.samples, "q_L")
        self.q_C = _infer_dataset_field(self.samples, "q_C")
        self.q_model = _infer_dataset_field(self.samples, "q_model")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        freq = torch.tensor(s["freq_hz"], dtype=torch.float32)
        ideal_s21 = torch.tensor(s["ideal_s21_db"], dtype=torch.float32)
        ideal_s11 = torch.tensor(s["ideal_s11_db"], dtype=torch.float32)
        real_s21 = torch.tensor(s["real_s21_db"], dtype=torch.float32)
        real_s11 = torch.tensor(s["real_s11_db"], dtype=torch.float32)

        fc_hz = float(s.get("fc_hz", 0.0) or 0.0)
        if not math.isfinite(fc_hz) or fc_hz <= 0.0:
            valid = torch.isfinite(freq)
            if valid.any():
                fmin = float(freq[valid].min().item())
                fmax = float(freq[valid].max().item())
                fc_hz = math.sqrt(max(fmin * fmax, 1e-12))
            else:
                fc_hz = 1.0

        mode = self.use_wave
        if mode == "mix":
            mode = "real" if torch.rand(1).item() < self.mix_real_prob else "ideal"

        if mode == "ideal":
            wave = torch.stack([ideal_s21, ideal_s11], dim=0)
        elif mode == "real":
            wave = torch.stack([real_s21, real_s11], dim=0)
        elif mode == "ideal_s21":
            wave = ideal_s21.unsqueeze(0)
        elif mode == "real_s21":
            wave = real_s21.unsqueeze(0)
        else:
            wave = torch.stack([ideal_s21, ideal_s11, real_s21, real_s11], dim=0)

        if not self.include_s11:
            if wave.shape[0] == 4:
                wave = wave[[0, 2], :]
            elif wave.shape[0] > 1:
                wave = wave[:1]

        freq_channels = 0
        if self.freq_mode != "none" or self.freq_scale != "none":
            eps = 1e-12
            freq_clamped = freq.clamp_min(eps)
            freq_feats = []
            logf = None
            mean_logf = None
            if self.freq_mode == "log_fc":
                freq_feats.append(torch.log10(freq_clamped / fc_hz))
            elif self.freq_mode == "linear_fc":
                freq_feats.append(freq / fc_hz)
            elif self.freq_mode == "log_f":
                logf = torch.log10(freq_clamped)
                freq_feats.append(logf)
            elif self.freq_mode == "log_f_centered":
                logf = torch.log10(freq_clamped)
                mean_logf = float(logf.mean().item())
                freq_feats.append(logf - mean_logf)
            elif self.freq_mode != "none":
                raise ValueError(f"Unknown freq_mode: {self.freq_mode}")

            if self.freq_scale == "log_fc":
                freq_feats.append(torch.full_like(freq, math.log10(fc_hz)))
            elif self.freq_scale == "log_f_mean":
                if logf is None:
                    logf = torch.log10(freq_clamped)
                if mean_logf is None:
                    mean_logf = float(logf.mean().item())
                freq_feats.append(torch.full_like(freq, mean_logf))
            elif self.freq_scale != "none":
                raise ValueError(f"Unknown freq_scale: {self.freq_scale}")

            if freq_feats:
                freq_wave = torch.stack(freq_feats, dim=0)
                wave = torch.cat([freq_wave, wave], dim=0)
                freq_channels = freq_wave.shape[0]

        if self.normalize_wave:
            if freq_channels < wave.shape[0]:
                wave_sig = wave[freq_channels:]
                wave_sig = wave_sig - wave_sig.mean(dim=-1, keepdim=True)
                wave_std = wave_sig.std(dim=-1, keepdim=True).clamp_min(1e-4)
                wave[freq_channels:] = wave_sig / wave_std

        ftype = s.get("filter_type", "lowpass")
        type_map = {"lowpass": 0, "highpass": 1, "bandpass": 2, "bandstop": 3}
        type_id = type_map.get(ftype, 0)
        scalar = torch.tensor([type_id, fc_hz], dtype=torch.float32)

        return {
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            "ideal_s21_db": ideal_s21,
            "real_s21_db": real_s21,
            "macro_ir_macros": self.macro_ir_macros[idx],
            "dsl_tokens": s.get("dsl_tokens") or [],
        }


def _resolve_ckpt(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "pytorch_model.bin"
    if direct.exists():
        return direct
    candidates = []
    for sub in path.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        if name.startswith("epoch_") or name.startswith("step_"):
            try:
                num = int(name.split("_", 1)[1])
            except Exception:
                continue
            ckpt = sub / "pytorch_model.bin"
            if ckpt.exists():
                candidates.append((num, ckpt))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    raise FileNotFoundError(f"No checkpoint found under {path}")


def _find_config(start: Path) -> Path:
    cur = start.resolve()
    while True:
        cfg = cur / "input_config.json"
        if cfg.exists():
            return cfg
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("input_config.json not found near checkpoint.")


def _resolve_q(args: argparse.Namespace, dataset: BilevelPlotDataset, cfg: dict) -> tuple[float | None, float | None, str]:
    q_mode = args.q_mode or cfg.get("q_mode", "auto")
    q_model_cli = str(args.q_model or cfg.get("q_model", "freq_dependent"))
    q_l_cli = args.q if args.q_l is None else args.q_l
    if args.q is None and args.q_l is None and args.q_c is None:
        q_l_cli = cfg.get("q_L_used", cfg.get("q_l", None))
    q_c_cli = args.q if args.q_c is None else args.q_c
    if args.q is None and args.q_l is None and args.q_c is None:
        q_c_cli = cfg.get("q_C_used", cfg.get("q_c", None))

    data_q_l = getattr(dataset, "q_L", None)
    data_q_c = getattr(dataset, "q_C", None)
    data_q_model = getattr(dataset, "q_model", None)

    if q_mode == "cli":
        return q_l_cli, q_c_cli, q_model_cli
    if q_mode == "data":
        if data_q_l is None and data_q_c is None:
            print("[warn] q_mode=data but dataset has no Q; falling back to CLI values.", flush=True)
            return q_l_cli, q_c_cli, q_model_cli
        return data_q_l, data_q_c, str(data_q_model) if data_q_model is not None else q_model_cli
    if data_q_l is not None or data_q_c is not None:
        return data_q_l, data_q_c, str(data_q_model) if data_q_model is not None else q_model_cli
    return q_l_cli, q_c_cli, q_model_cli


def _ref_freq_hz(freq_hz: torch.Tensor, fc_hz: torch.Tensor | float) -> torch.Tensor:
    if isinstance(fc_hz, torch.Tensor):
        if torch.isfinite(fc_hz).item() and fc_hz.item() > 0.0:
            return fc_hz
    else:
        if math.isfinite(float(fc_hz)) and float(fc_hz) > 0.0:
            return torch.tensor(float(fc_hz), device=freq_hz.device, dtype=freq_hz.dtype)
    f_min = torch.min(freq_hz)
    f_max = torch.max(freq_hz)
    return torch.sqrt(f_min * f_max)


def _enforce_non_empty(macro_ids: torch.Tensor, g_logits: torch.Tensor, skip_id: int) -> torch.Tensor:
    if bool((macro_ids != skip_id).any()):
        return macro_ids
    logits = g_logits[:, :skip_id]
    flat_idx = int(torch.argmax(logits).item())
    cell_idx = flat_idx // skip_id
    macro_idx = flat_idx % skip_id
    macro_ids = macro_ids.clone()
    macro_ids[cell_idx] = macro_idx
    return macro_ids


def _expand_macros_with_placeholders(macro_seq: List[Tuple[int, str]], slot_count: int) -> Tuple[list, List[int]]:
    comps = []
    slot_indices: List[int] = []
    base = 1_000_000.0
    series_positions = [i for i, (_, macro) in enumerate(macro_seq) if macro in SERIES_MACROS]
    last_series_pos = series_positions[-1] if series_positions else None
    current = "in"
    node_idx = 0
    for seq_idx, (cell_pos, macro) in enumerate(macro_seq):
        if macro in SERIES_MACROS:
            if last_series_pos is not None and seq_idx == last_series_pos:
                a = current
                b = "out"
            else:
                node_idx += 1
                a = current
                b = f"n{node_idx}"
            current = b
        else:
            a = current
            b = current
        macro_def = MACRO_LIBRARY[macro]
        placeholder_vals = [base + cell_pos * slot_count + j for j in range(len(macro_def.slot_types))]
        macro_comps = macro_def.expand_fn(a, b, "gnd", placeholder_vals, cell_pos)
        for c in macro_comps:
            slot_global = int(round(float(c.value_si) - base))
            slot_indices.append(slot_global)
        comps.extend(macro_comps)
    return comps, slot_indices


def _build_circuit_and_indices(
    macro_ids: torch.Tensor,
    *,
    id_to_macro: List[str],
    skip_id: int,
    slot_count: int,
    assembler: DynamicCircuitAssembler,
    device: torch.device,
    dtype: torch.dtype,
    q_L: float | None,
    q_C: float | None,
) -> Tuple[object, torch.Tensor]:
    macro_seq = [(i, id_to_macro[int(m)]) for i, m in enumerate(macro_ids.tolist()) if int(m) != skip_id]
    comps, slot_indices = _expand_macros_with_placeholders(macro_seq, slot_count)
    circuit, _ = assembler.assemble(comps, trainable=False, device=device, dtype=dtype, q_L=q_L, q_C=q_C)
    value_comp_indices = getattr(circuit, "value_comp_indices", None)
    if value_comp_indices is None:
        slot_idx_order = slot_indices
    else:
        slot_idx_order = [slot_indices[int(i)] for i in value_comp_indices]
    return circuit, torch.tensor(slot_idx_order, device=device, dtype=torch.long)


def _macro_ids_to_list(macro_ids: torch.Tensor, id_to_macro: List[str], skip_id: int) -> List[str]:
    return [id_to_macro[int(mid)] for mid in macro_ids.tolist() if int(mid) != skip_id]


def _plot_pair(freq: np.ndarray, target: np.ndarray, pred: np.ndarray, out_path: Path, title: str, pred_label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available; skipping plot: {exc}")
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
    ax.plot(freq, target, label="Target S21", color="C0")
    ax.plot(freq, pred, label=pred_label, color="C1", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("S21 (dB)")
    ax.grid(True, ls=":")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot random bilevel samples (target vs predicted S21).")
    p.add_argument("--data", type=Path, required=True, help="Path to validation jsonl.")
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint file or directory.")
    p.add_argument("--config", type=Path, help="Optional input_config.json; auto-located if omitted.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/bilevel_samples"), help="Directory to save plots/results.")
    p.add_argument("--num", type=int, default=5, help="Number of random samples to plot.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-wave", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"], default=None)
    p.add_argument("--freq-mode", choices=["none", "log_fc", "linear_fc", "log_f", "log_f_centered"], default=None)
    p.add_argument("--freq-scale", choices=["none", "log_fc", "log_f_mean"], default=None)
    p.add_argument("--spec-mode", choices=["none", "type_fc"], default=None)
    p.add_argument("--no-s11", dest="include_s11", action="store_false")
    p.set_defaults(include_s11=None)
    p.add_argument("--wave-norm", dest="wave_norm", action="store_true")
    p.add_argument("--no-wave-norm", dest="wave_norm", action="store_false")
    p.set_defaults(wave_norm=False)
    p.add_argument(
        "--target-wave",
        choices=["auto", "ideal", "real"],
        default=None,
        help="Target S21 selection: ideal, real, or auto (match Q).",
    )
    p.add_argument(
        "--q-mode",
        choices=["auto", "data", "cli"],
        default=None,
        help="Select Q source: auto (prefer dataset), data (dataset only), cli (args only).",
    )
    p.add_argument("--q", type=float, default=None, help="Finite-Q loss model (None disables).")
    p.add_argument("--q-l", type=float, default=None, help="Override Q for inductors (None -> use --q).")
    p.add_argument("--q-c", type=float, default=None, help="Override Q for capacitors (None -> use --q).")
    p.add_argument(
        "--q-model",
        type=str,
        default=None,
        choices=["freq_dependent", "fixed_ref"],
        help="Q modeling for eval: freq_dependent or fixed_ref.",
    )
    p.add_argument("--refine-steps", type=int, default=0, help="Unroll steps for refinement (0 disables).")
    p.add_argument("--inner-lr", type=float, default=None)
    p.add_argument("--inner-max-step", type=float, default=None)
    p.add_argument("--inner-raw-min", type=float, default=None)
    p.add_argument("--inner-raw-max", type=float, default=None)
    p.add_argument("--inner-nan-backoff", type=float, default=None)
    p.add_argument("--inner-nan-tries", type=int, default=None)
    p.add_argument("--phys-init", dest="phys_init", action="store_true", help="Enable physics-aware slot init bias.")
    p.add_argument("--no-phys-init", dest="phys_init", action="store_false", help="Disable physics-aware slot init bias.")
    p.set_defaults(phys_init=None)
    p.add_argument("--phys-init-beta", type=float, default=None, help="Strength of physics-aware init bias.")
    p.add_argument(
        "--phys-init-bpbs-only",
        dest="phys_init_bpbs_only",
        action="store_true",
        help="Apply physics init bias only to bandpass/bandstop samples.",
    )
    p.add_argument(
        "--no-phys-init-bpbs-only",
        dest="phys_init_bpbs_only",
        action="store_false",
        help="Apply physics init bias to all filter types.",
    )
    p.set_defaults(phys_init_bpbs_only=None)
    p.add_argument("--phys-init-base-bias", type=float, default=None, help="Baseline raw bias offset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve_ckpt(args.ckpt)
    cfg_path = args.config or _find_config(ckpt_path.parent)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    use_wave = args.use_wave or cfg.get("use_wave", "ideal")
    freq_mode = args.freq_mode or cfg.get("freq_mode", "log_fc")
    freq_scale = args.freq_scale or cfg.get("freq_scale", "none")
    spec_mode = args.spec_mode or cfg.get("spec_mode", "type_fc")
    include_s11 = cfg.get("include_s11", True) if args.include_s11 is None else bool(args.include_s11)

    macro_vocab = list(cfg.get("macro_vocab") or [])
    if not macro_vocab:
        raise ValueError("macro_vocab missing in input_config.json.")
    k_max = int(cfg.get("k_max", 0))
    if k_max <= 0:
        raise ValueError("k_max missing in input_config.json.")
    slot_count = int(cfg.get("slot_count", 0))
    if slot_count <= 0:
        raise ValueError("slot_count missing in input_config.json.")

    dataset = BilevelPlotDataset(
        str(args.data),
        use_wave=use_wave,
        normalize_wave=bool(args.wave_norm),
        freq_mode=freq_mode,
        freq_scale=freq_scale,
        include_s11=include_s11,
    )

    q_L, q_C, q_model = _resolve_q(args, dataset, cfg)
    q_active = q_L is not None or q_C is not None
    target_wave = args.target_wave or cfg.get("target_wave", "auto")
    if target_wave == "auto":
        target_wave = "real" if q_active else "ideal"

    phys_init = bool(cfg.get("phys_init", False)) if args.phys_init is None else bool(args.phys_init)
    phys_init_beta = float(cfg.get("phys_init_beta", 1.0)) if args.phys_init_beta is None else float(args.phys_init_beta)
    phys_init_bpbs_only = (
        bool(cfg.get("phys_init_bpbs_only", True)) if args.phys_init_bpbs_only is None else bool(args.phys_init_bpbs_only)
    )
    phys_init_base_bias = float(cfg.get("phys_init_base_bias", -22.0)) if args.phys_init_base_bias is None else float(args.phys_init_base_bias)

    inner_lr = float(args.inner_lr or cfg.get("inner_lr", 5e-2))
    inner_max_step = float(args.inner_max_step or cfg.get("inner_max_step", 0.5))
    inner_raw_min = float(args.inner_raw_min or cfg.get("inner_raw_min", -32.0))
    inner_raw_max = float(args.inner_raw_max or cfg.get("inner_raw_max", -12.0))
    inner_nan_backoff = float(args.inner_nan_backoff or cfg.get("inner_nan_backoff", 0.5))
    inner_nan_tries = int(args.inner_nan_tries or cfg.get("inner_nan_tries", 3))

    device = torch.device(args.device)
    model = Wave2StructureModel(
        k_max=k_max,
        macro_vocab_size=len(macro_vocab),
        slot_count=slot_count,
        waveform_in_channels=dataset[0]["wave"].shape[0],
        d_model=int(cfg.get("d_model", 512)),
        hidden_mult=int(cfg.get("hidden_mult", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        spec_mode=spec_mode,
        use_role_queries=bool(cfg.get("use_role_queries", False)),
        role_input_frac=float(cfg.get("role_input_frac", 0.2)),
        role_output_frac=float(cfg.get("role_output_frac", 0.2)),
    ).to(device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    macro_slot_mask = torch.zeros((len(macro_vocab) + 1, slot_count), dtype=torch.float32, device=device)
    for mid, macro in enumerate(macro_vocab):
        slen = len(MACRO_LIBRARY[macro].slot_types)
        if slen > 0:
            macro_slot_mask[mid, :slen] = 1.0
    skip_id = len(macro_vocab)
    id_to_macro = list(macro_vocab)

    rng = random.Random(int(args.seed))
    n = min(int(args.num), len(dataset))
    indices = rng.sample(range(len(dataset)), n)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    assembler = DynamicCircuitAssembler(z0=50.0)

    with summary_path.open("w") as sf:
        for idx in indices:
            sample = dataset[idx]
            raw = dataset.samples[idx]
            wave = sample["wave"].unsqueeze(0).to(device)
            scalars = sample["scalar"].to(device)
            filter_type = scalars[0:1].long()
            fc_hz = scalars[1:2]
            freq = sample["freq"].to(device)
            if target_wave == "real":
                target = sample["real_s21_db"].to(device)
            else:
                target = sample["ideal_s21_db"].to(device)

            with torch.no_grad():
                g_logits, slot_raw = model(wave, filter_type=filter_type, fc_hz=fc_hz)
            g_logits = g_logits.float()
            slot_raw = slot_raw.float()
            if phys_init and phys_init_beta != 0.0:
                fc_safe = fc_hz.clamp_min(1e-6)
                raw_bias = -torch.log(fc_safe * (2.0 * math.pi))
                delta = raw_bias - float(phys_init_base_bias)
                if phys_init_bpbs_only:
                    bpbs = (filter_type == 2) | (filter_type == 3)
                    delta = delta * bpbs.to(delta.dtype)
                slot_raw = slot_raw + float(phys_init_beta) * delta.view(-1, 1, 1)

            pred_ids = torch.argmax(g_logits[0], dim=-1)
            pred_ids = _enforce_non_empty(pred_ids, g_logits[0], skip_id)
            slot_mask = macro_slot_mask[pred_ids].to(slot_raw.dtype)

            ref_freq = None
            if q_active and str(q_model) == "fixed_ref":
                ref_freq = _ref_freq_hz(freq, fc_hz[0])
            circuit, slot_idx = _build_circuit_and_indices(
                pred_ids,
                id_to_macro=id_to_macro,
                skip_id=skip_id,
                slot_count=slot_count,
                assembler=assembler,
                device=device,
                dtype=wave.dtype,
                q_L=q_L,
                q_C=q_C,
            )

            raw_pre = slot_raw[0].clamp(inner_raw_min, inner_raw_max)
            values_flat = torch.exp(raw_pre.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
            values_vec = values_flat.index_select(0, slot_idx)
            pred_pre = circuit(freq, values=values_vec, output="s21_db", q_model=str(q_model), ref_freq_hz=ref_freq)
            pred = pred_pre
            pred_label = "Pred S21"
            post_mse = None
            if int(args.refine_steps) > 0:
                raw_init = raw_pre.detach().requires_grad_(True)
                loss_post, raw_post = unroll_refine_slots(
                    raw_init,
                    slot_mask,
                    slot_idx,
                    circuit,
                    freq,
                    target,
                    steps=int(args.refine_steps),
                    lr=inner_lr,
                    max_step=inner_max_step,
                    raw_min=inner_raw_min,
                    raw_max=inner_raw_max,
                    nan_backoff=inner_nan_backoff,
                    max_backoff=inner_nan_tries,
                    create_graph=False,
                    return_raw=True,
                    q_model=str(q_model),
                    ref_freq_hz=ref_freq,
                )
                raw_post = raw_post.to(raw_pre.dtype)
                values_flat_post = torch.exp(raw_post.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                values_vec_post = values_flat_post.index_select(0, slot_idx)
                pred_post = circuit(freq, values=values_vec_post, output="s21_db", q_model=str(q_model), ref_freq_hz=ref_freq)
                pred = pred_post
                pred_label = f"Pred S21 (refine={int(args.refine_steps)})"
                post_mse = float(loss_post.detach().cpu().item())

            pred_mse = float(F.mse_loss(pred, target).detach().cpu().item())

            freq_np = freq.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            pred_macros = _macro_ids_to_list(pred_ids, id_to_macro, skip_id)
            gt_macros = sample.get("macro_ir_macros") or []
            if not gt_macros:
                gt_macros = dsl_tokens_to_macro_sequence(sample.get("dsl_tokens") or [], strict=False)
            if len(gt_macros) > k_max:
                gt_macros = gt_macros[:k_max]

            sample_id = raw.get("sample_id", f"idx_{idx}")
            ftype = raw.get("filter_type", "unknown")
            fc_val = float(raw.get("fc_hz", 0.0) or 0.0)
            title = f"{sample_id} | {ftype} | fc={fc_val:.3g}Hz"
            plot_path = out_dir / f"{sample_id}_plot.png"
            _plot_pair(freq_np, target_np, pred_np, plot_path, title=title, pred_label=pred_label)

            row = {
                "sample_id": sample_id,
                "index": int(idx),
                "filter_type": ftype,
                "fc_hz": fc_val,
                "target_wave": target_wave,
                "pred_mse": pred_mse,
                "pred_mse_post": post_mse,
                "pred_macros": pred_macros,
                "gt_macros": gt_macros,
                "plot": str(plot_path),
                "checkpoint": str(ckpt_path),
            }
            sf.write(json.dumps(row) + "\n")
            print(
                f"[sample {idx}] id={sample_id} type={ftype} fc={fc_val:.3g} "
                f"mse={pred_mse:.4f} pred_macros={pred_macros} gt_macros={gt_macros}"
            )

    print(f"[done] wrote {n} plots + summary to {out_dir}")


if __name__ == "__main__":
    main()
