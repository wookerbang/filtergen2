from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import (
    MACRO_LIBRARY,
    SERIES_MACROS,
    VALUE_SLOTS,
    dsl_tokens_to_macro_values,
    make_dsl_prefix_allowed_tokens_fn,
    make_macro_ir_prefix_allowed_tokens_fn,
)  # noqa: E402
from src.data.token_decode import build_label_value_map, decode_components_from_token_ids  # noqa: E402
from src.models import VACTT5  # noqa: E402
from src.physics.differentiable_rf import (  # noqa: E402
    calc_violation_max,
    calc_violation_quantile,
    DynamicCircuitAssembler,
    barrier_loss,
    DifferentiablePhysicsKernel,
    unroll_refine_slots,
)


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


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


class Seq2SeqEvalDataset(Dataset):
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
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))
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

        mask_min = s.get("mask_min_db")
        mask_max = s.get("mask_max_db")
        if mask_min is None:
            mask_min = [float("nan")] * len(freq)
        if mask_max is None:
            mask_max = [float("nan")] * len(freq)
        mask_min = torch.tensor(mask_min, dtype=torch.float32)
        mask_max = torch.tensor(mask_max, dtype=torch.float32)

        return {
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
            "ideal_s11_db": ideal_s11,
            "real_s21_db": real_s21,
            "real_s11_db": real_s11,
            "mask_min_db": mask_min,
            "mask_max_db": mask_max,
        }


def _expand_macros_with_placeholders(macro_seq: List[Tuple[int, str]], slot_count: int) -> Tuple[list, List[int]]:
    comps = []
    slot_indices: List[int] = []
    base = 1_000_000.0
    series_positions = [i for i, (_, macro) in enumerate(macro_seq) if macro in SERIES_MACROS]
    last_series_pos = series_positions[-1] if series_positions else None
    current = "in"
    node_idx = 0
    for seq_idx, (cell_pos, macro) in enumerate(macro_seq):
        macro_def = MACRO_LIBRARY[macro]
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
        placeholder_vals = [base + cell_pos * slot_count + j for j in range(len(macro_def.slot_types))]
        macro_comps = macro_def.expand_fn(a, b, "gnd", placeholder_vals, cell_pos)
        for c in macro_comps:
            slot_global = int(round(float(c.value_si) - base))
            slot_indices.append(slot_global)
        comps.extend(macro_comps)
    return comps, slot_indices


def _build_circuit_and_indices(
    macros: List[str],
    *,
    slot_count: int,
    assembler: DynamicCircuitAssembler,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[object, torch.Tensor]:
    macro_seq = [(i, m) for i, m in enumerate(macros)]
    comps, slot_indices = _expand_macros_with_placeholders(macro_seq, slot_count)
    circuit, _ = assembler.assemble(comps, trainable=False, device=device, dtype=dtype)
    value_comp_indices = getattr(circuit, "value_comp_indices", None)
    if value_comp_indices is None:
        slot_idx_order = slot_indices
    else:
        slot_idx_order = [slot_indices[int(i)] for i in value_comp_indices]
    return circuit, torch.tensor(slot_idx_order, device=device, dtype=torch.long)


def _has_constraints(mask_min: torch.Tensor, mask_max: torch.Tensor) -> bool:
    return bool(torch.isfinite(mask_min).any().item() or torch.isfinite(mask_max).any().item())


def _dynamic_envelope_masks(
    target_db: torch.Tensor,
    *,
    pass_drop_db: float,
    stop_rel_db: float,
    delta_pass: float,
    delta_stop: float,
    peak_quantile: float,
    pass_max_db: float | None,
) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
    target = target_db.reshape(-1)
    mask_min = torch.full_like(target, float("nan"))
    mask_max = torch.full_like(target, float("nan"))
    finite = torch.isfinite(target)
    if not bool(finite.any().item()):
        return mask_min, mask_max, False, False
    vals = target[finite]
    q = float(peak_quantile)
    if q >= 1.0:
        peak = vals.max()
    else:
        peak = torch.quantile(vals, q)
    pass_thresh = peak - float(pass_drop_db)
    stop_thresh = peak - float(stop_rel_db)
    pass_mask = finite & (target >= pass_thresh)
    stop_mask = finite & (target <= stop_thresh)
    stop_mask = stop_mask & ~pass_mask
    if bool(pass_mask.any().item()):
        mask_min[pass_mask] = target[pass_mask] - float(delta_pass)
        pass_max = target[pass_mask] + float(delta_pass)
        if pass_max_db is not None and math.isfinite(float(pass_max_db)):
            pass_max = torch.minimum(
                pass_max,
                torch.tensor(float(pass_max_db), device=target.device, dtype=target.dtype),
            )
        mask_max[pass_mask] = pass_max
    if bool(stop_mask.any().item()):
        mask_max[stop_mask] = target[stop_mask] + float(delta_stop)
    return mask_min, mask_max, bool(pass_mask.any().item()), bool(stop_mask.any().item())


def _masked_mse_components(
    pred_db: torch.Tensor,
    target_db: torch.Tensor,
    mask_min_db: torch.Tensor,
    mask_max_db: torch.Tensor,
    *,
    w_pass: float,
    w_stop: float,
) -> tuple[float, float, float]:
    pred = pred_db.reshape(-1)
    target = target_db.reshape(-1)
    full_mse = float(F.mse_loss(pred, target).item())
    min_mask = torch.isfinite(mask_min_db.reshape(-1))
    max_mask = torch.isfinite(mask_max_db.reshape(-1))
    constrained_mask = min_mask | max_mask
    if bool(constrained_mask.any()):
        constrained = torch.mean((pred[constrained_mask] - target[constrained_mask]) ** 2)
        constrained_mse = float(constrained.item())
    else:
        constrained_mse = full_mse
    pass_mask = min_mask
    stop_mask = max_mask & ~pass_mask
    if bool(pass_mask.any()) or bool(stop_mask.any()):
        err = (pred - target) ** 2
        num = 0.0
        denom = 0.0
        if bool(pass_mask.any()):
            num += float(w_pass) * float(err[pass_mask].sum().item())
            denom += float(w_pass) * float(pass_mask.sum().item())
        if bool(stop_mask.any()):
            num += float(w_stop) * float(err[stop_mask].sum().item())
            denom += float(w_stop) * float(stop_mask.sum().item())
        weighted_mse = num / denom if denom > 0.0 else full_mse
    else:
        weighted_mse = full_mse
    return full_mse, constrained_mse, weighted_mse


def _circuit_s11_db(
    circuit: object,
    freq_hz: torch.Tensor,
    values_vec: torch.Tensor,
) -> torch.Tensor:
    s11, _, _, _ = circuit(freq_hz, values=values_vec, output="sparams")
    return DifferentiablePhysicsKernel.s11_db(s11)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate seq2seq DSL model with unroll refinement.")
    p.add_argument("--data", required=True, type=Path, help="Path to dataset jsonl (val/test).")
    p.add_argument("--ckpt", required=True, type=Path, help="Checkpoint dir (trainer save).")
    p.add_argument("--tokenizer", type=Path, help="Tokenizer path (defaults to --ckpt).")
    p.add_argument("--t5-name", type=str, default="t5-base", help="Base T5 model name (for raw state_dict load).")
    p.add_argument("--repr", choices=["dsl", "dsl_value", "macro_ir", "sfci"], default="macro_ir")
    p.add_argument("--num", type=int, default=2000, help="Number of samples to eval.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample selection.")
    p.add_argument("--use-wave", default="real", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"])
    p.add_argument("--target-wave", choices=["ideal", "real"], default="real")
    p.add_argument("--wave-norm", action="store_true", help="Normalize waveforms (must match training if enabled).")
    p.add_argument(
        "--freq-mode",
        choices=["none", "log_fc", "linear_fc", "log_f", "log_f_centered"],
        default="log_f_centered",
    )
    p.add_argument(
        "--freq-scale",
        choices=["none", "log_fc", "log_f_mean"],
        default="log_f_mean",
    )
    p.add_argument(
        "--spec-mode",
        choices=["none", "type_fc"],
        default="type_fc",
        help="Spec token usage: none (wave-only) or type_fc (prepend filter type + fc token).",
    )
    p.add_argument("--no-s11", dest="include_s11", action="store_false", help="Drop S11 channels.")
    p.set_defaults(include_s11=True)
    p.add_argument("--allow-input-mismatch", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")

    # generation
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of beam search.")
    p.add_argument("--num-beams", type=int, default=4, help="Beam size when not sampling.")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--syntax-mask", action="store_true", help="Apply DSL grammar mask during decoding.")
    p.add_argument("--value-mode", choices=["standard", "precision"], default="precision")
    p.add_argument("--predict-values", action="store_true", help="Predict continuous values for SFCI tokens.")

    # unroll refinement
    p.add_argument("--unroll-steps", type=int, default=10)
    p.add_argument("--inner-lr", type=float, default=5e-2)
    p.add_argument("--inner-max-step", type=float, default=0.5)
    p.add_argument("--inner-raw-min", type=float, default=-32.0)
    p.add_argument("--inner-raw-max", type=float, default=-12.0)
    p.add_argument("--inner-nan-backoff", type=float, default=0.5)
    p.add_argument("--inner-nan-tries", type=int, default=3)
    p.add_argument(
        "--loss-mode",
        choices=["full_mse", "constrained_mse", "weighted_mse", "barrier_only"],
        default=None,
    )
    p.add_argument("--w-pass", type=float, default=None)
    p.add_argument("--w-stop", type=float, default=None)
    p.add_argument("--barrier-weight", type=float, default=None)
    p.add_argument(
        "--yield-taus",
        type=str,
        default="0,0.25,0.5,1.0",
        help="Comma-separated tau (dB) slack for yield reporting.",
    )
    p.add_argument(
        "--yield-alphas",
        type=str,
        default="0.01,0.03,0.05",
        help="Comma-separated alpha for robust yield reporting.",
    )
    p.add_argument(
        "--yield-s11-max-db",
        type=float,
        default=None,
        help="S11 max (dB) for yield guard (disabled by default).",
    )
    p.add_argument(
        "--mask-mode",
        choices=["dataset", "dynamic"],
        default="dataset",
        help="Mask source for yield/loss: dataset masks or dynamic envelope from target.",
    )
    p.add_argument("--mask-pass-drop-db", type=float, default=3.0)
    p.add_argument("--mask-stop-rel-db", type=float, default=20.0)
    p.add_argument("--mask-delta-pass", type=float, default=1.0)
    p.add_argument("--mask-delta-stop", type=float, default=3.0)
    p.add_argument("--mask-peak-quantile", type=float, default=1.0)
    p.add_argument("--mask-pass-max-db", type=float, default=0.0)

    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--output", type=Path, help="Optional JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    yield_taus = sorted({t for t in _parse_csv_floats(args.yield_taus) if math.isfinite(t) and t >= 0.0})
    if not yield_taus:
        yield_taus = [0.0]
    yield_alphas = sorted({a for a in _parse_csv_floats(args.yield_alphas) if math.isfinite(a) and 0.0 < a < 1.0})
    primary_tau = yield_taus[0]
    use_s11_yield = args.yield_s11_max_db is not None and math.isfinite(float(args.yield_s11_max_db))
    yield_s11_max_db = float(args.yield_s11_max_db) if use_s11_yield else None
    mask_mode = str(args.mask_mode)
    mask_pass_drop_db = float(args.mask_pass_drop_db)
    mask_stop_rel_db = float(args.mask_stop_rel_db)
    mask_delta_pass = float(args.mask_delta_pass)
    mask_delta_stop = float(args.mask_delta_stop)
    mask_peak_quantile = float(args.mask_peak_quantile)
    mask_pass_max_db = float(args.mask_pass_max_db)

    tok_path = args.tokenizer or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_map = build_label_value_map(tokenizer)

    dataset = Seq2SeqEvalDataset(
        str(args.data),
        use_wave=args.use_wave,
        normalize_wave=bool(args.wave_norm),
        freq_mode=args.freq_mode,
        freq_scale=args.freq_scale,
        include_s11=bool(args.include_s11),
    )
    if args.num and args.num > 0:
        max_n = min(int(args.num), len(dataset.samples))
        if max_n < len(dataset.samples):
            rng = random.Random(args.seed)
            idxs = rng.sample(range(len(dataset.samples)), max_n)
            dataset.samples = [dataset.samples[i] for i in idxs]
        else:
            dataset.samples = dataset.samples[:max_n]

    def collate(batch: List[dict]) -> dict:
        target_key = "real_s21_db" if args.target_wave == "real" else "ideal_s21_db"
        target_s11_key = "real_s11_db" if args.target_wave == "real" else "ideal_s11_db"
        targets = torch.stack([b[target_key] for b in batch])
        targets_s11 = torch.stack([b[target_s11_key] for b in batch])
        return {
            "wave": torch.stack([b["wave"] for b in batch]),
            "freq": torch.stack([b["freq"] for b in batch]),
            "target_s21_db": targets,
            "target_s11_db": targets_s11,
            "mask_min_db": torch.stack([b["mask_min_db"] for b in batch]),
            "mask_max_db": torch.stack([b["mask_max_db"] for b in batch]),
            "scalar": torch.stack([b["scalar"] for b in batch]),
        }

    device = torch.device(args.device)
    num_workers = max(0, int(args.num_workers))
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        **loader_kwargs,
    )
    non_blocking = bool(pin_memory)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    def _build_value_token_info(tok):
        vocab = tok.get_vocab()
        val_ids = []
        slot_map = {}
        slot_order = {t: i for i, t in enumerate(VALUE_SLOTS)}
        for t, tid in vocab.items():
            if t.startswith("<VAL_"):
                val_ids.append(int(tid))
                if t in slot_order:
                    slot_map[int(tid)] = int(slot_order[t])
        return val_ids, slot_map

    value_token_ids, slot_type_map = _build_value_token_info(tokenizer)
    slot_count = max(len(m.slot_types) for m in MACRO_LIBRARY.values())

    in_channels = dataset[0]["wave"].shape[0]
    cfg_path = args.ckpt / "input_config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        mismatches = []
        for key, expected in (
            ("freq_mode", args.freq_mode),
            ("freq_scale", args.freq_scale),
            ("include_s11", bool(args.include_s11)),
            ("spec_mode", args.spec_mode),
        ):
            if key in cfg and cfg[key] != expected:
                mismatches.append((key, cfg[key], expected))
        if "in_channels" in cfg and int(cfg["in_channels"]) != int(in_channels):
            mismatches.append(("in_channels", cfg["in_channels"], int(in_channels)))
        if mismatches:
            lines = ["Input config mismatch with checkpoint:"]
            lines.extend([f"- {k}: ckpt={v_ckpt} current={v_cur}" for k, v_ckpt, v_cur in mismatches])
            lines.append("Align flags or pass --allow-input-mismatch.")
            msg = "\n".join(lines)
            if args.allow_input_mismatch:
                print(f"[warn] {msg}")
            else:
                raise ValueError(msg)
    else:
        cfg = {}

    loss_mode = args.loss_mode or cfg.get("loss_mode", "full_mse")
    w_pass = float(args.w_pass) if args.w_pass is not None else float(cfg.get("w_pass", 1.0))
    w_stop = float(args.w_stop) if args.w_stop is not None else float(cfg.get("w_stop", 5.0))
    barrier_weight = (
        float(args.barrier_weight)
        if args.barrier_weight is not None
        else float(cfg.get("barrier_weight", 0.0))
    )

    model = VACTT5(
        t5_name=args.t5_name,
        waveform_in_channels=in_channels,
        vocab_size=len(tokenizer),
        spec_mode=args.spec_mode,
        value_token_ids=value_token_ids,
        slot_type_token_to_idx=slot_type_map,
        macro_slot_count=slot_count if args.repr == "macro_ir" else None,
    )
    state_path = args.ckpt / "pytorch_model.bin"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.t5.config.eos_token_id = tokenizer.eos_token_id
    model.t5.config.pad_token_id = tokenizer.pad_token_id
    model.t5.config.decoder_start_token_id = tokenizer.pad_token_id
    model.to(device).eval()

    assembler = DynamicCircuitAssembler(z0=50.0)

    prefix_allowed = None
    if args.syntax_mask:
        if args.repr == "macro_ir":
            prefix_allowed = make_macro_ir_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr in ("dsl", "dsl_value"):
            prefix_allowed = make_dsl_prefix_allowed_tokens_fn(tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    total = 0
    failed = 0
    pre_mse_sum = 0.0
    post_mse_sum = 0.0
    pre_loss_sum = 0.0
    post_loss_sum = 0.0
    pre_constrained_sum = 0.0
    post_constrained_sum = 0.0
    pre_weighted_sum = 0.0
    post_weighted_sum = 0.0
    yield_total = 0
    yield_pre_pass = {tau: 0 for tau in yield_taus}
    yield_post_pass = {tau: 0 for tau in yield_taus}
    yield_oracle_pass = {tau: 0 for tau in yield_taus}
    yield_pre_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    yield_post_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    yield_oracle_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    mask_pass_empty = 0
    mask_stop_empty = 0
    mask_empty = 0

    for batch in loader:
        wave = batch["wave"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        freq = batch["freq"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        target = batch["target_s21_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        target_s11 = None
        if use_s11_yield:
            target_s11 = batch["target_s11_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        mask_min = batch["mask_min_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        mask_max = batch["mask_max_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        scalar = batch["scalar"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        filter_type = scalar[:, 0].long()
        fc_hz = scalar[:, 1]

        gen_kwargs = dict(
            wave=wave,
            filter_type=filter_type,
            fc_hz=fc_hz,
            max_new_tokens=int(args.max_new),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed,
            num_return_sequences=1,
            repetition_penalty=float(args.repetition_penalty),
        )
        if args.do_sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    num_beams=1,
                    top_p=float(args.top_p),
                    temperature=float(args.temperature),
                )
            )
        else:
            gen_kwargs.update(dict(do_sample=False, num_beams=max(1, int(args.num_beams))))

        with torch.no_grad():
            outs = model.generate(**gen_kwargs)
        seqs = outs.cpu().tolist()

        slot_values_seqs = None
        macro_raw_seqs = None
        predict_values = args.repr == "dsl" or (args.repr == "sfci" and args.predict_values)
        if args.repr == "macro_ir" or predict_values:
            seq_lens = [len(s) for s in seqs]
            max_len = max(seq_lens) if seq_lens else 0
            if max_len > 0:
                pad_id = int(tokenizer.pad_token_id)
                seq_tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    seq_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                if args.repr == "macro_ir":
                    with torch.no_grad():
                        pred_raw = model.predict_macro_values(
                            wave,
                            filter_type,
                            fc_hz,
                            token_ids=seq_tensor,
                        )
                    pred_raw = pred_raw.detach().cpu().tolist()
                    macro_raw_seqs = [pred_raw[i][: seq_lens[i]] for i in range(len(seqs))]
                else:
                    with torch.no_grad():
                        pred_vals = model.predict_values(
                            wave,
                            filter_type,
                            fc_hz,
                            token_ids=seq_tensor,
                            mode=args.value_mode,
                        )
                    pred_vals = pred_vals.detach().cpu().tolist()
                    slot_values_seqs = [pred_vals[i][: seq_lens[i]] for i in range(len(seqs))]

        for b, seq in enumerate(seqs):
            total += 1
            try:
                if not torch.isfinite(target[b]).all():
                    failed += 1
                    continue
                if mask_mode == "dynamic":
                    mask_min_b, mask_max_b, pass_any, stop_any = _dynamic_envelope_masks(
                        target[b],
                        pass_drop_db=mask_pass_drop_db,
                        stop_rel_db=mask_stop_rel_db,
                        delta_pass=mask_delta_pass,
                        delta_stop=mask_delta_stop,
                        peak_quantile=mask_peak_quantile,
                        pass_max_db=mask_pass_max_db,
                    )
                    if not pass_any:
                        mask_pass_empty += 1
                    if not stop_any:
                        mask_stop_empty += 1
                    if not (pass_any or stop_any):
                        mask_empty += 1
                else:
                    mask_min_b = mask_min[b]
                    mask_max_b = mask_max[b]
                tokens_full = tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=False)
                if args.repr == "macro_ir":
                    macros = []
                    macro_positions = []
                    for i_tok, (tid, tok) in enumerate(zip(seq, tokens_full)):
                        if int(tid) in special_ids:
                            continue
                        if tok in MACRO_LIBRARY:
                            macros.append(tok)
                            macro_positions.append(i_tok)
                    if not macros:
                        failed += 1
                        continue
                    slot_mask = torch.zeros((len(macros), slot_count), dtype=dtype, device=device)
                    slot_raw = torch.full((len(macros), slot_count), float(args.inner_raw_min), dtype=dtype, device=device)
                    pred_seq = macro_raw_seqs[b] if macro_raw_seqs is not None else None
                    for i_m, (macro, pos) in enumerate(zip(macros, macro_positions)):
                        slen = len(MACRO_LIBRARY[macro].slot_types)
                        slot_mask[i_m, :slen] = 1.0
                        if pred_seq is None or pos >= len(pred_seq):
                            continue
                        row = pred_seq[pos]
                        for j in range(min(slen, len(row))):
                            v = float(row[j])
                            if math.isfinite(v):
                                slot_raw[i_m, j] = v
                    circuit, slot_idx = _build_circuit_and_indices(
                        macros,
                        slot_count=slot_count,
                        assembler=assembler,
                        device=device,
                        dtype=dtype,
                    )
                elif args.repr == "sfci":
                    slot_vals = slot_values_seqs[b] if slot_values_seqs is not None else None
                    comps, _ = decode_components_from_token_ids(
                        seq,
                        tokenizer,
                        repr_kind="sfci",
                        label_to_value=label_map,
                        slot_values=slot_vals,
                    )
                    if not comps:
                        failed += 1
                        continue
                    circuit, comps = assembler.assemble(comps, trainable=False, device=device, dtype=dtype)
                    value_comp_indices = getattr(circuit, "value_comp_indices", None)
                    if value_comp_indices is None:
                        value_comp_indices = list(range(len(comps)))
                    if not value_comp_indices:
                        failed += 1
                        continue
                    slot_raw = torch.full((len(value_comp_indices),), float(args.inner_raw_min), dtype=dtype, device=device)
                    for i_v, comp_idx in enumerate(value_comp_indices):
                        v = float(comps[comp_idx].value_si)
                        if math.isfinite(v) and v > 0.0:
                            slot_raw[i_v] = math.log(v)
                    slot_mask = torch.ones_like(slot_raw)
                    slot_idx = torch.arange(len(value_comp_indices), device=device, dtype=torch.long)
                else:
                    slot_vals = slot_values_seqs[b] if slot_values_seqs is not None else None
                    tokens = []
                    vals = []
                    for i_tok, (tid, tok) in enumerate(zip(seq, tokens_full)):
                        if int(tid) in special_ids:
                            continue
                        tokens.append(tok)
                        if slot_vals is not None and i_tok < len(slot_vals):
                            vals.append(float(slot_vals[i_tok]))
                    segments = dsl_tokens_to_macro_values(
                        tokens,
                        slot_values=vals if slot_vals is not None else None,
                        strict=False,
                    )
                    if not segments:
                        failed += 1
                        continue
                    macros = [m for m, _ in segments]
                    slot_vals_by_macro = [vals_m for _, vals_m in segments]

                    slot_mask = torch.zeros((len(macros), slot_count), dtype=dtype, device=device)
                    slot_raw = torch.full((len(macros), slot_count), float(args.inner_raw_min), dtype=dtype, device=device)
                    for i_m, macro in enumerate(macros):
                        slen = len(MACRO_LIBRARY[macro].slot_types)
                        slot_mask[i_m, :slen] = 1.0
                        vals_m = slot_vals_by_macro[i_m]
                        for j in range(min(slen, len(vals_m))):
                            v = float(vals_m[j])
                            if math.isfinite(v) and v > 0.0:
                                slot_raw[i_m, j] = math.log(v)

                    circuit, slot_idx = _build_circuit_and_indices(
                        macros,
                        slot_count=slot_count,
                        assembler=assembler,
                        device=device,
                        dtype=dtype,
                    )
                raw_pre = slot_raw.clamp(float(args.inner_raw_min), float(args.inner_raw_max))
                values_flat = torch.exp(raw_pre.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                values_vec = values_flat.index_select(0, slot_idx)
                pred_pre = circuit(freq[b], values=values_vec, output="s21_db")
                pred_pre_s11 = None
                if use_s11_yield:
                    pred_pre_s11 = _circuit_s11_db(circuit, freq[b], values_vec)
                if not torch.isfinite(pred_pre).all():
                    failed += 1
                    continue
                pre_mse, pre_constrained, pre_weighted = _masked_mse_components(
                    pred_pre,
                    target[b],
                    mask_min_b,
                    mask_max_b,
                    w_pass=w_pass,
                    w_stop=w_stop,
                )
                if not math.isfinite(pre_mse):
                    failed += 1
                    continue
                if loss_mode == "constrained_mse":
                    pre_loss = pre_constrained
                elif loss_mode == "weighted_mse":
                    pre_loss = pre_weighted
                elif loss_mode == "barrier_only":
                    pre_loss = float(barrier_weight) * float(
                        barrier_loss(pred_pre, mask_min_b, mask_max_b).item()
                    )
                else:
                    pre_loss = pre_mse

                raw_init = slot_raw.detach().clone().requires_grad_(True)
                _, raw_post = unroll_refine_slots(
                    raw_init,
                    slot_mask,
                    slot_idx,
                    circuit,
                    freq[b],
                    target[b],
                    steps=int(args.unroll_steps),
                    lr=float(args.inner_lr),
                    max_step=float(args.inner_max_step),
                    raw_min=float(args.inner_raw_min),
                    raw_max=float(args.inner_raw_max),
                    nan_backoff=float(args.inner_nan_backoff),
                    max_backoff=int(args.inner_nan_tries),
                    create_graph=False,
                    return_raw=True,
                    mask_min_db=mask_min_b,
                    mask_max_db=mask_max_b,
                    barrier_weight=barrier_weight,
                    loss_mode=loss_mode,
                    w_pass=w_pass,
                    w_stop=w_stop,
                )
                values_flat_post = torch.exp(raw_post.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                values_vec_post = values_flat_post.index_select(0, slot_idx)
                pred_post = circuit(freq[b], values=values_vec_post, output="s21_db")
                pred_post_s11 = None
                if use_s11_yield:
                    pred_post_s11 = _circuit_s11_db(circuit, freq[b], values_vec_post)
                if not torch.isfinite(pred_post).all():
                    failed += 1
                    continue
                post_mse, post_constrained, post_weighted = _masked_mse_components(
                    pred_post,
                    target[b],
                    mask_min_b,
                    mask_max_b,
                    w_pass=w_pass,
                    w_stop=w_stop,
                )
                if not math.isfinite(post_mse):
                    failed += 1
                    continue
                if loss_mode == "constrained_mse":
                    post_loss = post_constrained
                elif loss_mode == "weighted_mse":
                    post_loss = post_weighted
                elif loss_mode == "barrier_only":
                    post_loss = float(barrier_weight) * float(
                        barrier_loss(pred_post, mask_min_b, mask_max_b).item()
                    )
                else:
                    post_loss = post_mse

                pre_mse_sum += pre_mse
                post_mse_sum += post_mse
                pre_constrained_sum += pre_constrained
                post_constrained_sum += post_constrained
                pre_weighted_sum += pre_weighted
                post_weighted_sum += post_weighted
                pre_loss_sum += pre_loss
                post_loss_sum += post_loss

                if _has_constraints(mask_min_b, mask_max_b):
                    yield_total += 1
                    oracle_s11 = target_s11[b] if use_s11_yield and target_s11 is not None else None
                    oracle_max = float(
                        calc_violation_max(
                            target[b],
                            mask_min_b,
                            mask_max_b,
                            pred_s11_db=oracle_s11,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    pre_max = float(
                        calc_violation_max(
                            pred_pre,
                            mask_min_b,
                            mask_max_b,
                            pred_s11_db=pred_pre_s11,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    post_max = float(
                        calc_violation_max(
                            pred_post,
                            mask_min_b,
                            mask_max_b,
                            pred_s11_db=pred_post_s11,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    for tau_val in yield_taus:
                        if oracle_max <= tau_val:
                            yield_oracle_pass[tau_val] += 1
                        if pre_max <= tau_val:
                            yield_pre_pass[tau_val] += 1
                        if post_max <= tau_val:
                            yield_post_pass[tau_val] += 1
                    for alpha_val in yield_alphas:
                        oracle_q = float(
                            calc_violation_quantile(
                                target[b],
                                mask_min_b,
                                mask_max_b,
                                alpha=alpha_val,
                                pred_s11_db=oracle_s11,
                                s11_max_db=yield_s11_max_db,
                            ).item()
                        )
                        pre_q = float(
                            calc_violation_quantile(
                                pred_pre,
                                mask_min_b,
                                mask_max_b,
                                alpha=alpha_val,
                                pred_s11_db=pred_pre_s11,
                                s11_max_db=yield_s11_max_db,
                            ).item()
                        )
                        post_q = float(
                            calc_violation_quantile(
                                pred_post,
                                mask_min_b,
                                mask_max_b,
                                alpha=alpha_val,
                                pred_s11_db=pred_post_s11,
                                s11_max_db=yield_s11_max_db,
                            ).item()
                        )
                        for tau_val in yield_taus:
                            if oracle_q <= tau_val:
                                yield_oracle_robust[(tau_val, alpha_val)] += 1
                            if pre_q <= tau_val:
                                yield_pre_robust[(tau_val, alpha_val)] += 1
                            if post_q <= tau_val:
                                yield_post_robust[(tau_val, alpha_val)] += 1
            except Exception:
                failed += 1
                continue

    def _rate(val: int) -> float | None:
        return (val / yield_total) if yield_total else None

    yield_oracle_tight = {f"{tau:g}": _rate(yield_oracle_pass[tau]) for tau in yield_taus}
    yield_pre_tight = {f"{tau:g}": _rate(yield_pre_pass[tau]) for tau in yield_taus}
    yield_post_tight = {f"{tau:g}": _rate(yield_post_pass[tau]) for tau in yield_taus}
    yield_oracle_robust_out = {
        f"{alpha:g}": {f"{tau:g}": _rate(yield_oracle_robust[(tau, alpha)]) for tau in yield_taus}
        for alpha in yield_alphas
    }
    yield_pre_robust_out = {
        f"{alpha:g}": {f"{tau:g}": _rate(yield_pre_robust[(tau, alpha)]) for tau in yield_taus}
        for alpha in yield_alphas
    }
    yield_post_robust_out = {
        f"{alpha:g}": {f"{tau:g}": _rate(yield_post_robust[(tau, alpha)]) for tau in yield_taus}
        for alpha in yield_alphas
    }

    results = {
        "num_samples": total,
        "failed_samples": failed,
        "mse_pre": pre_mse_sum / max(1, total - failed),
        "mse_post": post_mse_sum / max(1, total - failed),
        "constrained_mse_pre": pre_constrained_sum / max(1, total - failed),
        "constrained_mse_post": post_constrained_sum / max(1, total - failed),
        "weighted_mse_pre": pre_weighted_sum / max(1, total - failed),
        "weighted_mse_post": post_weighted_sum / max(1, total - failed),
        "loss_mode": loss_mode,
        "loss_pre": pre_loss_sum / max(1, total - failed),
        "loss_post": post_loss_sum / max(1, total - failed),
        "yield_total": yield_total,
        "yield_oracle": _rate(yield_oracle_pass[primary_tau]),
        "yield_pre": _rate(yield_pre_pass[primary_tau]),
        "yield_post": _rate(yield_post_pass[primary_tau]),
        "yield_taus_db": yield_taus,
        "yield_alphas": yield_alphas,
        "yield_s11_max_db": yield_s11_max_db,
        "yield_oracle_tight_by_tau": yield_oracle_tight,
        "yield_pre_tight_by_tau": yield_pre_tight,
        "yield_post_tight_by_tau": yield_post_tight,
        "yield_oracle_robust_by_tau": yield_oracle_robust_out,
        "yield_pre_robust_by_tau": yield_pre_robust_out,
        "yield_post_robust_by_tau": yield_post_robust_out,
        "mask_mode": mask_mode,
        "mask_pass_drop_db": mask_pass_drop_db,
        "mask_stop_rel_db": mask_stop_rel_db,
        "mask_delta_pass": mask_delta_pass,
        "mask_delta_stop": mask_delta_stop,
        "mask_peak_quantile": mask_peak_quantile,
        "mask_pass_max_db": mask_pass_max_db,
        "mask_pass_empty": mask_pass_empty,
        "mask_stop_empty": mask_stop_empty,
        "mask_empty": mask_empty,
        "target_wave": str(args.target_wave),
        "checkpoint": str(args.ckpt),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
