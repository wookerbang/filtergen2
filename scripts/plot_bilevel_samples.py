"""Plot target vs predicted S21 for bilevel model samples and print macro sequences."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import MACRO_LIBRARY, SERIES_MACROS, dsl_tokens_to_macro_sequence
from src.models import Wave2StructureModel
from src.physics.differentiable_rf import DynamicCircuitAssembler


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


def _resolve_fmin_fmax(sample: dict, fc_hz: float) -> tuple[float, float]:
    freq_range = sample.get("freq_range")
    if freq_range and len(freq_range) >= 2:
        f_min = float(freq_range[0])
        f_max = float(freq_range[1])
    else:
        bw = sample.get("bw_frac") or sample.get("stopband_bw_frac")
        if bw is not None and fc_hz > 0.0:
            f_min = fc_hz * (1.0 - 0.5 * float(bw))
            f_max = fc_hz * (1.0 + 0.5 * float(bw))
        else:
            f_min = fc_hz
            f_max = fc_hz
    if not math.isfinite(f_min) or f_min <= 0.0:
        f_min = max(fc_hz, 1.0)
    if not math.isfinite(f_max) or f_max <= 0.0:
        f_max = max(fc_hz, 1.0)
    if f_min > f_max:
        f_min, f_max = f_max, f_min
    return f_min, f_max


def _resolve_bw_frac(sample: dict, fc_hz: float, f_min: float, f_max: float) -> float:
    bw = sample.get("bw_frac") or sample.get("stopband_bw_frac")
    if bw is None:
        if fc_hz > 0.0:
            bw = (f_max - f_min) / fc_hz
        else:
            bw = 0.0
    try:
        return float(bw)
    except Exception:
        return 0.0


class BilevelPlotDataset:
    def __init__(
        self,
        jsonl_path: str,
        *,
        use_wave: str = "ideal",
        mix_real_prob: float = 0.3,
        normalize_wave: bool = False,
        freq_mode: str = "log_fc",
        freq_scale: str = "none",
        include_s11: bool = True,
    ) -> None:
        self.samples = []
        self.macro_ir_macros = []
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
        f_min_hz, f_max_hz = _resolve_fmin_fmax(s, fc_hz)
        bw_frac = _resolve_bw_frac(s, fc_hz, f_min_hz, f_max_hz)
        ripple_db = float(s.get("ripple_db", 0.5) or 0.5)
        stopband_max_db = float(s.get("stopband_max_db", -40.0) or -40.0)
        order = float(s.get("order", 0.0) or 0.0)

        return {
            "raw": s,
            "macro_ir_macros": self.macro_ir_macros[idx],
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            "f_min_hz": float(f_min_hz),
            "f_max_hz": float(f_max_hz),
            "bw_frac": float(bw_frac),
            "ripple_db": float(ripple_db),
            "stopband_max_db": float(stopband_max_db),
            "order": float(order),
            "ideal_s21_db": ideal_s21,
            "real_s21_db": real_s21,
        }


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
) -> Tuple[object, torch.Tensor]:
    macro_seq = [(i, id_to_macro[int(m)]) for i, m in enumerate(macro_ids.tolist()) if int(m) != skip_id]
    comps, slot_indices = _expand_macros_with_placeholders(macro_seq, slot_count)
    circuit, _ = assembler.assemble(comps, trainable=False, device=device, dtype=dtype)
    value_comp_indices = getattr(circuit, "value_comp_indices", None)
    if value_comp_indices is None:
        slot_idx_order = slot_indices
    else:
        slot_idx_order = [slot_indices[int(i)] for i in value_comp_indices]
    return circuit, torch.tensor(slot_idx_order, device=device, dtype=torch.long)


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)[:80]


def _plot_sample(freq: np.ndarray, target: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available: {exc}")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(freq, target, label="Target S21", color="C0")
    plt.plot(freq, pred, label="Pred S21", color="C1", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("S21 (dB)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot bilevel predictions vs targets.")
    p.add_argument("--data", type=Path, required=True, help="Path to val jsonl.")
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint dir or pytorch_model.bin.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/bilevel_samples"))
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-wave",
        choices=["auto", "ideal", "real"],
        default="auto",
        help="Target S21 selection: auto (prefer real), ideal, or real.",
    )
    p.add_argument("--use-wave", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"], default=None)
    p.add_argument("--wave-norm", dest="wave_norm", action="store_true", help="Normalize waveforms.")
    p.add_argument("--no-wave-norm", dest="wave_norm", action="store_false", help="Disable waveform normalization.")
    p.set_defaults(wave_norm=None)
    p.add_argument("--freq-mode", choices=["none", "log_fc", "linear_fc", "log_f", "log_f_centered"], default=None)
    p.add_argument("--freq-scale", choices=["none", "log_fc", "log_f_mean"], default=None)
    p.add_argument(
        "--spec-mode",
        choices=[
            "none",
            "type_fc",
            "type_fc_bw",
            "type_fc_bw_ripple",
            "type_fc_bw_ripple_stop",
            "type_fc_bw_ripple_stop_order",
            "type_fmin_fmax",
            "type_fmin_fmax_ripple",
            "type_fmin_fmax_ripple_stop",
            "type_fmin_fmax_ripple_stop_order",
        ],
        default=None,
    )
    p.add_argument("--s11", dest="include_s11", action="store_true", help="Include S11 channels.")
    p.add_argument("--no-s11", dest="include_s11", action="store_false", help="Drop S11 channels.")
    p.set_defaults(include_s11=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = _resolve_ckpt(args.ckpt)
    cfg_path = _find_config(ckpt_path.parent)
    with cfg_path.open() as f:
        cfg = json.load(f)

    use_wave = args.use_wave or cfg.get("use_wave", "ideal")
    freq_mode = args.freq_mode or cfg.get("freq_mode", "log_fc")
    freq_scale = args.freq_scale or cfg.get("freq_scale", "none")
    spec_mode = args.spec_mode or cfg.get("spec_mode", "type_fc")
    include_s11 = bool(args.include_s11) if args.include_s11 is not None else bool(cfg.get("include_s11", True))
    wave_norm = bool(args.wave_norm) if args.wave_norm is not None else bool(cfg.get("wave_norm", False))

    dataset = BilevelPlotDataset(
        str(args.data),
        use_wave=use_wave,
        normalize_wave=wave_norm,
        freq_mode=freq_mode,
        freq_scale=freq_scale,
        include_s11=include_s11,
    )

    if len(dataset) == 0:
        raise ValueError(f"No samples found in {args.data}")

    macro_vocab = cfg["macro_vocab"]
    id_to_macro = list(macro_vocab)
    macro_vocab_size = int(cfg.get("macro_vocab_size", len(macro_vocab)))
    k_max = int(cfg["k_max"])
    slot_count = int(cfg["slot_count"])

    in_channels = dataset[0]["wave"].shape[0]
    model = Wave2StructureModel(
        k_max=k_max,
        macro_vocab_size=macro_vocab_size,
        slot_count=slot_count,
        waveform_in_channels=in_channels,
        d_model=int(cfg.get("d_model", 512)),
        hidden_mult=int(cfg.get("hidden_mult", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        spec_mode=str(spec_mode),
        gate_skip_bias=float(cfg.get("gate_skip_bias", 0.0)),
        use_role_queries=bool(cfg.get("use_role_queries", False)),
        role_input_frac=float(cfg.get("role_input_frac", 0.2)),
        role_output_frac=float(cfg.get("role_output_frac", 0.2)),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    skip_id = len(id_to_macro)
    macro_slot_mask = torch.zeros((len(id_to_macro) + 1, slot_count), dtype=torch.float32)
    for mid, macro in enumerate(id_to_macro):
        slen = len(MACRO_LIBRARY[macro].slot_types)
        if slen > 0:
            macro_slot_mask[mid, :slen] = 1.0

    assembler = DynamicCircuitAssembler(z0=50.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: max(1, int(args.num))]

    for idx in indices:
        sample = dataset[idx]
        raw_dict = sample["raw"]
        wave = sample["wave"].unsqueeze(0).to(device)
        freq = sample["freq"].to(device)
        scalar = sample["scalar"].to(device)
        f_min_hz = torch.tensor([sample["f_min_hz"]], device=device)
        f_max_hz = torch.tensor([sample["f_max_hz"]], device=device)
        bw_frac = torch.tensor([sample["bw_frac"]], device=device)
        ripple_db = torch.tensor([sample["ripple_db"]], device=device)
        stopband_max_db = torch.tensor([sample["stopband_max_db"]], device=device)
        order = torch.tensor([sample["order"]], device=device)

        with torch.no_grad():
            g_logits, slot_raw = model(
                wave,
                filter_type=scalar[0:1].long(),
                fc_hz=scalar[1:2],
                f_min_hz=f_min_hz,
                f_max_hz=f_max_hz,
                bw_frac=bw_frac,
                ripple_db=ripple_db,
                stopband_max_db=stopband_max_db,
                order=order,
            )
        g_logits = g_logits[0]
        slot_raw = slot_raw[0]

        macro_ids = torch.argmax(g_logits, dim=-1)
        macro_ids = _enforce_non_empty(macro_ids, g_logits, skip_id)
        slot_mask = macro_slot_mask[macro_ids].to(device=device, dtype=slot_raw.dtype)
        slot_raw_clamped = slot_raw.clamp(min=-32.0, max=-12.0)
        values_flat = torch.exp(slot_raw_clamped.reshape(-1)) * slot_mask.reshape(-1) + 1e-30

        circuit, slot_idx = _build_circuit_and_indices(
            macro_ids,
            id_to_macro=id_to_macro,
            skip_id=skip_id,
            slot_count=slot_count,
            assembler=assembler,
            device=device,
            dtype=slot_raw.dtype,
        )
        values_vec = values_flat.index_select(0, slot_idx)
        pred_s21 = circuit(freq, values=values_vec, output="s21_db")

        if args.target_wave == "real":
            target = sample["real_s21_db"]
        elif args.target_wave == "ideal":
            target = sample["ideal_s21_db"]
        else:
            target = sample["real_s21_db"] if raw_dict.get("real_s21_db") is not None else sample["ideal_s21_db"]

        freq_np = sample["freq"].cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = pred_s21.detach().cpu().numpy()

        mse = float(np.mean((pred_np - target_np) ** 2))
        sample_id = str(raw_dict.get("sample_id", idx))
        gt_macros = sample["macro_ir_macros"]
        pred_macros = [id_to_macro[int(m)] for m in macro_ids.tolist() if int(m) != skip_id]

        print(
            f"[sample {idx}] id={sample_id} type={raw_dict.get('filter_type')} "
            f"fc={float(raw_dict.get('fc_hz', 0.0)):.3e} mse={mse:.4f} "
            f"pred_macros={pred_macros} gt_macros={gt_macros}"
        )

        out_name = f"{_safe_name(sample_id)}_mse_{mse:.3f}.png"
        out_path = args.output_dir / out_name
        _plot_sample(freq_np, target_np, pred_np, out_path)


if __name__ == "__main__":
    main()
