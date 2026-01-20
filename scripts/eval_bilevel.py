"""Evaluate bilevel model with pre-/post-refine metrics and yield."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import (
    MACRO_LIBRARY,
    MACRO_SER_C,
    MACRO_SER_L,
    MACRO_SHUNT_C,
    MACRO_SHUNT_L,
    SERIES_MACROS,
    dsl_tokens_to_macro_sequence,
)
from src.utils.macro_transition import build_transition_matrices, viterbi_decode
from src.models import Wave2StructureModel
from src.physics.differentiable_rf import (
    calc_violation_max,
    calc_violation_quantile,
    DynamicCircuitAssembler,
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


def _band_metrics(
    pred_db: torch.Tensor,
    mask_min_db: torch.Tensor,
    mask_max_db: torch.Tensor,
) -> tuple[float | None, float | None]:
    pred = pred_db.reshape(-1)
    pass_mask = torch.isfinite(mask_min_db.reshape(-1))
    stop_mask = torch.isfinite(mask_max_db.reshape(-1)) & ~pass_mask
    ripple = None
    if bool(pass_mask.any()):
        pb = pred[pass_mask]
        ripple = float((pb.max() - pb.min()).item())
    stop_max = None
    if bool(stop_mask.any()):
        stop_max = float(pred[stop_mask].max().item())
    return ripple, stop_max


def _circuit_s11_db(
    circuit: object,
    freq_hz: torch.Tensor,
    values_vec: torch.Tensor,
) -> torch.Tensor:
    s11, _, _, _ = circuit(freq_hz, values=values_vec, output="sparams")
    return DifferentiablePhysicsKernel.s11_db(s11)


def _interp_target(
    freq_src: torch.Tensor,
    target_src: torch.Tensor,
    freq_dst: torch.Tensor,
) -> torch.Tensor:
    src = freq_src.detach().cpu().numpy().astype(float)
    dst = freq_dst.detach().cpu().numpy().astype(float)
    tgt = target_src.detach().cpu().numpy().astype(float)
    finite = np.isfinite(tgt)
    if finite.sum() < 2:
        return torch.full_like(freq_dst, float("nan"))
    out = np.interp(dst, src[finite], tgt[finite])
    return torch.tensor(out, device=target_src.device, dtype=target_src.dtype)


def _interp_mask_nearest(
    freq_src: torch.Tensor,
    mask_src: torch.Tensor,
    freq_dst: torch.Tensor,
) -> torch.Tensor:
    src = freq_src.detach().cpu().numpy().astype(float)
    dst = freq_dst.detach().cpu().numpy().astype(float)
    vals = mask_src.detach().cpu().numpy().astype(float)
    idx = np.searchsorted(src, dst)
    idx = np.clip(idx, 1, len(src) - 1)
    left = idx - 1
    right = idx
    choose_right = (dst - src[left]) > (src[right] - dst)
    nearest = np.where(choose_right, right, left)
    out = vals[nearest]
    return torch.tensor(out, device=mask_src.device, dtype=mask_src.dtype)


class BilevelEvalDataset(Dataset):
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

        mask_min = s.get("mask_min_db")
        mask_max = s.get("mask_max_db")
        if mask_min is None:
            mask_min = [float("nan")] * len(freq)
        if mask_max is None:
            mask_max = [float("nan")] * len(freq)
        mask_min = torch.tensor(mask_min, dtype=torch.float32)
        mask_max = torch.tensor(mask_max, dtype=torch.float32)

        return {
            "dsl_tokens": s.get("dsl_tokens", []),
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
            "ideal_s11_db": ideal_s11,
            "real_s21_db": real_s21,
            "real_s11_db": real_s11,
            "mask_min_db": mask_min,
            "mask_max_db": mask_max,
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


def _has_constraints(mask_min: torch.Tensor, mask_max: torch.Tensor) -> bool:
    return bool(torch.isfinite(mask_min).any().item() or torch.isfinite(mask_max).any().item())


def _apply_order_length_mask(
    logits: torch.Tensor,
    order: torch.Tensor,
    *,
    skip_id: int,
) -> torch.Tensor:
    """
    Enforce exact macro length = order (non-skip before order, skip after).
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (B,K,V), got {tuple(logits.shape)}")
    bsz, k_max, _ = logits.shape
    if order.ndim == 0:
        order = order.unsqueeze(0)
    order = order.to(device=logits.device)
    finite = torch.isfinite(order)
    order_int = torch.round(order).to(torch.long).clamp(min=1, max=k_max)
    pos = torch.arange(k_max, device=logits.device).unsqueeze(0).expand(bsz, k_max)
    before = (pos < order_int.unsqueeze(1)) & finite.unsqueeze(1)
    after = (pos >= order_int.unsqueeze(1)) & finite.unsqueeze(1)
    masked = logits.clone()
    if bool(before.any()):
        masked[..., skip_id] = masked[..., skip_id].masked_fill(before, -1e9)
    if bool(after.any()):
        masked[..., :skip_id] = masked[..., :skip_id].masked_fill(after.unsqueeze(-1), -1e9)
    return masked


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate bilevel model (pre/post refine + yield).")
    p.add_argument("--data", type=Path, required=True, help="Path to eval jsonl.")
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint file or directory.")
    p.add_argument("--config", type=Path, help="Optional input_config.json; auto-located if omitted.")
    p.add_argument("--output", type=Path, help="Optional JSON output path.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (num_workers>0).")
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory for faster H2D copies.")
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive.")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of eval samples (0 disables).")
    p.add_argument("--use-wave", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"], default=None)
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
    p.add_argument("--no-s11", dest="include_s11", action="store_false")
    p.set_defaults(include_s11=None)
    p.add_argument("--wave-norm", dest="wave_norm", action="store_true")
    p.add_argument("--no-wave-norm", dest="wave_norm", action="store_false")
    p.set_defaults(wave_norm=False)
    p.add_argument("--unroll-steps", type=int, default=None)
    p.add_argument("--inner-lr", type=float, default=None)
    p.add_argument("--inner-max-step", type=float, default=None)
    p.add_argument("--inner-raw-min", type=float, default=None)
    p.add_argument("--inner-raw-max", type=float, default=None)
    p.add_argument("--inner-nan-backoff", type=float, default=None)
    p.add_argument("--inner-nan-tries", type=int, default=None)
    p.add_argument(
        "--loss-mode",
        choices=["full_mse", "constrained_mse", "weighted_mse", "barrier_only"],
        default=None,
    )
    p.add_argument("--w-pass", type=float, default=None, help="Weighted MSE passband weight.")
    p.add_argument("--w-stop", type=float, default=None, help="Weighted MSE stopband weight.")
    p.add_argument("--barrier-weight", type=float, default=None, help="Barrier loss weight for refine.")
    p.add_argument(
        "--yield-taus",
        type=str,
        default="0,0.25,0.5,1.0",
        help="Comma-separated tau (dB) slack for yield reporting.",
    )
    p.add_argument(
        "--yield-alphas",
        type=str,
        default="0.01,0.05",
        help="Comma-separated alpha for robust yield reporting.",
    )
    p.add_argument(
        "--yield-s11-max-db",
        type=float,
        default=None,
        help="S11 max (dB) for yield guard (disabled by default).",
    )
    p.add_argument(
        "--uniform-grid",
        action="store_true",
        help="Evaluate additional metrics on a shared wide log-frequency grid.",
    )
    p.add_argument("--uniform-grid-points", type=int, default=256, help="Number of points for uniform grid.")
    p.add_argument("--uniform-grid-mult", type=float, default=10.0, help="Span as fc/mult to fc*mult.")
    p.add_argument(
        "--target-wave",
        choices=["ideal", "real"],
        default=None,
        help="Target S21 curve for refine/MSE (ideal_s21_db or real_s21_db).",
    )
    p.add_argument(
        "--force-order-length",
        dest="force_order_length",
        action="store_true",
        help="Force macro length to match filter order (assumes 1 macro per component).",
    )
    p.add_argument(
        "--no-force-order-length",
        dest="force_order_length",
        action="store_false",
        help="Disable order-length constraint.",
    )
    p.set_defaults(force_order_length=None)
    p.add_argument("--use-viterbi", action="store_true", help="Decode macros with Viterbi + hard transitions.")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return p.parse_args()


def _new_group() -> dict:
    return {
        "count": 0,
        "mse_pre": 0.0,
        "mse_post": 0.0,
        "constrained_mse_pre": 0.0,
        "constrained_mse_post": 0.0,
        "weighted_mse_pre": 0.0,
        "weighted_mse_post": 0.0,
        "ripple_pre_sum": 0.0,
        "ripple_post_sum": 0.0,
        "ripple_count": 0,
        "stop_pre_sum": 0.0,
        "stop_post_sum": 0.0,
        "stop_count": 0,
        "yield_total": 0,
        "yield_pre": 0,
        "yield_post": 0,
        "yield_oracle": 0,
        "failed": 0,
        "macro_slot_total": 0,
        "macro_slot_correct": 0,
        "macro_non_skip_total": 0,
        "macro_non_skip_correct": 0,
        "len_abs_sum": 0.0,
        "len_bias_sum": 0.0,
        "len_exact": 0,
    }


def _new_uniform_group() -> dict:
    return {
        "count": 0,
        "mse_pre": 0.0,
        "mse_post": 0.0,
        "constrained_mse_pre": 0.0,
        "constrained_mse_post": 0.0,
        "weighted_mse_pre": 0.0,
        "weighted_mse_post": 0.0,
        "ripple_pre_sum": 0.0,
        "ripple_post_sum": 0.0,
        "ripple_count": 0,
        "stop_pre_sum": 0.0,
        "stop_post_sum": 0.0,
        "stop_count": 0,
        "failed": 0,
    }


def main() -> None:
    args = parse_args()
    yield_taus = sorted({t for t in _parse_csv_floats(args.yield_taus) if math.isfinite(t) and t >= 0.0})
    if not yield_taus:
        yield_taus = [0.0]
    yield_alphas = sorted({a for a in _parse_csv_floats(args.yield_alphas) if math.isfinite(a) and 0.0 < a < 1.0})
    primary_tau = yield_taus[0]
    use_s11_yield = args.yield_s11_max_db is not None and math.isfinite(float(args.yield_s11_max_db))
    yield_s11_max_db = float(args.yield_s11_max_db) if use_s11_yield else None
    ckpt_path = _resolve_ckpt(args.ckpt)
    cfg_path = args.config or _find_config(ckpt_path.parent)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    use_wave = args.use_wave or cfg.get("use_wave", "ideal")
    freq_mode = args.freq_mode or cfg.get("freq_mode", "log_fc")
    freq_scale = args.freq_scale or cfg.get("freq_scale", "none")
    spec_mode = args.spec_mode or cfg.get("spec_mode", "type_fc")
    target_wave = args.target_wave or cfg.get("target_wave", "ideal")
    force_order_length = bool(cfg.get("force_order_length", False)) if args.force_order_length is None else bool(args.force_order_length)
    include_s11 = cfg.get("include_s11", True) if args.include_s11 is None else bool(args.include_s11)
    loss_mode = args.loss_mode or cfg.get("loss_mode", "full_mse")
    w_pass = float(args.w_pass) if args.w_pass is not None else float(cfg.get("w_pass", 1.0))
    w_stop = float(args.w_stop) if args.w_stop is not None else float(cfg.get("w_stop", 5.0))
    barrier_weight = float(args.barrier_weight) if args.barrier_weight is not None else float(cfg.get("barrier_weight", 0.0))
    d_model = int(cfg.get("d_model", 512))
    hidden_mult = int(cfg.get("hidden_mult", 2))
    dropout = float(cfg.get("dropout", 0.1))
    use_role_queries = bool(cfg.get("use_role_queries", False))
    role_input_frac = float(cfg.get("role_input_frac", 0.2))
    role_output_frac = float(cfg.get("role_output_frac", 0.2))
    unroll_steps = int(args.unroll_steps or cfg.get("unroll_steps", 5))
    inner_lr = float(args.inner_lr or cfg.get("inner_lr", 5e-2))
    inner_max_step = float(args.inner_max_step or cfg.get("inner_max_step", 0.5))
    inner_raw_min = float(args.inner_raw_min or cfg.get("inner_raw_min", -32.0))
    inner_raw_max = float(args.inner_raw_max or cfg.get("inner_raw_max", -12.0))
    inner_nan_backoff = float(args.inner_nan_backoff or cfg.get("inner_nan_backoff", 0.5))
    inner_nan_tries = int(args.inner_nan_tries or cfg.get("inner_nan_tries", 3))

    macro_vocab = list(cfg.get("macro_vocab") or [])
    if not macro_vocab:
        raise ValueError("macro_vocab missing in input_config.json.")
    macro_to_id = {m: i for i, m in enumerate(macro_vocab)}
    k_max = int(cfg.get("k_max", 0))
    if k_max <= 0:
        raise ValueError("k_max missing in input_config.json.")
    slot_count = int(cfg.get("slot_count", 0))
    if slot_count <= 0:
        raise ValueError("slot_count missing in input_config.json.")

    dataset = BilevelEvalDataset(
        str(args.data),
        use_wave=use_wave,
        normalize_wave=bool(args.wave_norm),
        freq_mode=freq_mode,
        freq_scale=freq_scale,
        include_s11=include_s11,
    )
    if args.max_samples and args.max_samples > 0:
        max_n = int(args.max_samples)
        dataset.samples = dataset.samples[:max_n]
        dataset.macro_ir_macros = dataset.macro_ir_macros[:max_n]

    def collate(batch: List[dict]) -> dict:
        if target_wave == "real":
            targets = torch.stack([b["real_s21_db"] for b in batch])
            targets_s11 = torch.stack([b["real_s11_db"] for b in batch])
        else:
            targets = torch.stack([b["ideal_s21_db"] for b in batch])
            targets_s11 = torch.stack([b["ideal_s11_db"] for b in batch])
        return {
            "wave": torch.stack([b["wave"] for b in batch]),
            "freq": torch.stack([b["freq"] for b in batch]),
            "target_s21_db": targets,
            "target_s11_db": targets_s11,
            "mask_min_db": torch.stack([b["mask_min_db"] for b in batch]),
            "mask_max_db": torch.stack([b["mask_max_db"] for b in batch]),
            "scalar": torch.stack([b["scalar"] for b in batch]),
            "f_min_hz": torch.tensor([b["f_min_hz"] for b in batch], dtype=torch.float32),
            "f_max_hz": torch.tensor([b["f_max_hz"] for b in batch], dtype=torch.float32),
            "bw_frac": torch.tensor([b["bw_frac"] for b in batch], dtype=torch.float32),
            "ripple_db": torch.tensor([b["ripple_db"] for b in batch], dtype=torch.float32),
            "stopband_max_db": torch.tensor([b["stopband_max_db"] for b in batch], dtype=torch.float32),
            "order": torch.tensor([b["order"] for b in batch], dtype=torch.float32),
            "dsl_tokens": [b["dsl_tokens"] for b in batch],
            "macro_ir_macros": [b.get("macro_ir_macros") for b in batch],
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

    model = Wave2StructureModel(
        k_max=k_max,
        macro_vocab_size=len(macro_vocab),
        slot_count=slot_count,
        waveform_in_channels=dataset[0]["wave"].shape[0],
        d_model=d_model,
        hidden_mult=hidden_mult,
        dropout=dropout,
        spec_mode=spec_mode,
        use_role_queries=use_role_queries,
        role_input_frac=role_input_frac,
        role_output_frac=role_output_frac,
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
    assembler = DynamicCircuitAssembler(z0=50.0)
    use_viterbi = bool(args.use_viterbi)
    c_skip_penalty = float(cfg.get("c_skip_penalty", 100.0))
    c_redundant_penalty = float(cfg.get("c_redundant_penalty", 1.0))
    if use_viterbi:
        redundant_macros = [MACRO_SER_L, MACRO_SER_C, MACRO_SHUNT_L, MACRO_SHUNT_C]
        c_hard, _ = build_transition_matrices(
            id_to_macro=id_to_macro,
            skip_id=skip_id,
            soft_skip_penalty=c_skip_penalty,
            soft_redundant_penalty=c_redundant_penalty,
            redundant_macros=redundant_macros,
            hard_ban_skip_to_non_skip=True,
        )
        c_hard = c_hard.to(device=device, dtype=dtype)

    type_names = {0: "lowpass", 1: "highpass", 2: "bandpass", 3: "bandstop"}
    total = 0
    pre_mse_sum = 0.0
    post_mse_sum = 0.0
    pre_constrained_sum = 0.0
    post_constrained_sum = 0.0
    pre_weighted_sum = 0.0
    post_weighted_sum = 0.0
    ripple_pre_sum = 0.0
    ripple_post_sum = 0.0
    ripple_count = 0
    stop_pre_sum = 0.0
    stop_post_sum = 0.0
    stop_count = 0
    uniform_enabled = bool(args.uniform_grid)
    uni_pre_mse_sum = 0.0
    uni_post_mse_sum = 0.0
    uni_pre_constrained_sum = 0.0
    uni_post_constrained_sum = 0.0
    uni_pre_weighted_sum = 0.0
    uni_post_weighted_sum = 0.0
    uni_ripple_pre_sum = 0.0
    uni_ripple_post_sum = 0.0
    uni_ripple_count = 0
    uni_stop_pre_sum = 0.0
    uni_stop_post_sum = 0.0
    uni_stop_count = 0
    uni_total = 0
    per_type_uniform: dict[str, dict] = {name: _new_uniform_group() for name in type_names.values()}
    yield_total = 0
    yield_pre_pass = {tau: 0 for tau in yield_taus}
    yield_post_pass = {tau: 0 for tau in yield_taus}
    yield_oracle_pass = {tau: 0 for tau in yield_taus}
    yield_pre_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    yield_post_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    yield_oracle_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    failed = 0
    nonfinite_logits = 0
    nonfinite_slot = 0
    nonfinite_target = 0
    nonfinite_pred_pre = 0
    nonfinite_pred_post = 0
    macro_slot_total = 0
    macro_slot_correct = 0
    macro_non_skip_total = 0
    macro_non_skip_correct = 0
    len_abs_sum = 0.0
    len_bias_sum = 0.0
    len_exact = 0
    per_type: dict[str, dict] = {name: _new_group() for name in type_names.values()}

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
        f_min_hz = batch["f_min_hz"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        f_max_hz = batch["f_max_hz"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        bw_frac = batch["bw_frac"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        ripple_db = batch["ripple_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        stopband_max_db = batch["stopband_max_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        order = batch["order"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        dsl_tokens = batch["dsl_tokens"]
        macro_ir_macros = batch.get("macro_ir_macros")
        filter_type = scalar[:, 0].long()
        fc_hz = scalar[:, 1]

        with torch.no_grad():
            g_logits, slot_raw = model(
                wave,
                filter_type=filter_type,
                fc_hz=fc_hz,
                f_min_hz=f_min_hz,
                f_max_hz=f_max_hz,
                bw_frac=bw_frac,
                ripple_db=ripple_db,
                stopband_max_db=stopband_max_db,
                order=order,
            )
        g_logits = g_logits.float()
        slot_raw = slot_raw.float()
        if force_order_length:
            g_logits = _apply_order_length_mask(g_logits, order, skip_id=skip_id)

        if use_viterbi:
            macro_ids = viterbi_decode(g_logits, c_hard)
        else:
            macro_ids = torch.argmax(g_logits, dim=-1)
        for b in range(wave.shape[0]):
            try:
                if not torch.isfinite(target[b]).all():
                    nonfinite_target += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue
                if not torch.isfinite(g_logits[b]).all():
                    nonfinite_logits += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue
                if not torch.isfinite(slot_raw[b]).all():
                    nonfinite_slot += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue
                macro_ids_b = _enforce_non_empty(macro_ids[b], g_logits[b], skip_id)
                slot_mask = macro_slot_mask[macro_ids_b].to(dtype)
                circuit, slot_idx = _build_circuit_and_indices(
                    macro_ids_b,
                    id_to_macro=id_to_macro,
                    skip_id=skip_id,
                    slot_count=slot_count,
                    assembler=assembler,
                    device=device,
                    dtype=dtype,
                )
                raw_pre = slot_raw[b].to(dtype).clamp(inner_raw_min, inner_raw_max)
                values_flat = torch.exp(raw_pre.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                values_vec = values_flat.index_select(0, slot_idx)
                pred_pre = circuit(freq[b], values=values_vec, output="s21_db")
                pred_pre_s11 = None
                if use_s11_yield:
                    pred_pre_s11 = _circuit_s11_db(circuit, freq[b], values_vec)
                if not torch.isfinite(pred_pre).all():
                    nonfinite_pred_pre += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                        per_type[ft_name]["failed"] += 1
                    continue
                pre_mse, pre_constrained, pre_weighted = _masked_mse_components(
                    pred_pre,
                    target[b],
                    mask_min[b],
                    mask_max[b],
                    w_pass=w_pass,
                    w_stop=w_stop,
                )
                if not math.isfinite(pre_mse):
                    nonfinite_pred_pre += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                        per_type[ft_name]["failed"] += 1
                    continue
                ripple_pre, stop_pre = _band_metrics(pred_pre, mask_min[b], mask_max[b])

                raw_init = slot_raw[b].to(dtype).detach().requires_grad_(True)
                loss_post, raw_post = unroll_refine_slots(
                    raw_init,
                    slot_mask,
                    slot_idx,
                    circuit,
                    freq[b],
                    target[b],
                    steps=unroll_steps,
                    lr=inner_lr,
                    max_step=inner_max_step,
                    raw_min=inner_raw_min,
                    raw_max=inner_raw_max,
                    nan_backoff=inner_nan_backoff,
                    max_backoff=inner_nan_tries,
                    create_graph=False,
                    return_raw=True,
                    mask_min_db=mask_min[b],
                    mask_max_db=mask_max[b],
                    barrier_weight=barrier_weight,
                    loss_mode=loss_mode,
                    w_pass=w_pass,
                    w_stop=w_stop,
                )
                raw_post = raw_post.to(dtype)
                values_flat_post = torch.exp(raw_post.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                values_vec_post = values_flat_post.index_select(0, slot_idx)
                pred_post = circuit(freq[b], values=values_vec_post, output="s21_db")
                pred_post_s11 = None
                if use_s11_yield:
                    pred_post_s11 = _circuit_s11_db(circuit, freq[b], values_vec_post)
                if not torch.isfinite(pred_post).all():
                    nonfinite_pred_post += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                        per_type[ft_name]["failed"] += 1
                    continue
                post_mse, post_constrained, post_weighted = _masked_mse_components(
                    pred_post,
                    target[b],
                    mask_min[b],
                    mask_max[b],
                    w_pass=w_pass,
                    w_stop=w_stop,
                )
                if not math.isfinite(post_mse):
                    nonfinite_pred_post += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                        per_type[ft_name]["failed"] += 1
                    continue
                ripple_post, stop_post = _band_metrics(pred_post, mask_min[b], mask_max[b])

                macros = None
                if macro_ir_macros is not None:
                    macros = macro_ir_macros[b]
                if not macros:
                    try:
                        macros = dsl_tokens_to_macro_sequence(dsl_tokens[b], strict=True)
                    except Exception:
                        failed += 1
                        ft_id = int(filter_type[b].item())
                        ft_name = type_names.get(ft_id, "unknown")
                        if ft_name not in per_type:
                            per_type[ft_name] = _new_group()
                        per_type[ft_name]["failed"] += 1
                        continue
                if len(macros) > k_max:
                    macros = macros[:k_max]
                macro_ids_gt = torch.full((k_max,), skip_id, dtype=torch.long)
                for j, m in enumerate(macros):
                    if m not in macro_to_id:
                        failed += 1
                        ft_id = int(filter_type[b].item())
                        ft_name = type_names.get(ft_id, "unknown")
                        if ft_name not in per_type:
                            per_type[ft_name] = _new_group()
                        per_type[ft_name]["failed"] += 1
                        macro_ids_gt = None
                        break
                    macro_ids_gt[j] = int(macro_to_id[m])
                if macro_ids_gt is None:
                    continue

                pred_ids = macro_ids[b].to(torch.long).cpu()
                gt_ids = macro_ids_gt
                slot_correct = int((pred_ids == gt_ids).sum().item())
                slot_total = int(k_max)
                gt_non_skip = gt_ids != skip_id
                non_skip_total = int(gt_non_skip.sum().item())
                non_skip_correct = int((pred_ids[gt_non_skip] == gt_ids[gt_non_skip]).sum().item())
                pred_len = int((pred_ids != skip_id).sum().item())
                gt_len = int((gt_ids != skip_id).sum().item())
                len_abs = abs(pred_len - gt_len)
                len_bias = pred_len - gt_len
                len_exact_flag = int(pred_len == gt_len)

                pre_mse_sum += pre_mse
                post_mse_sum += post_mse
                pre_constrained_sum += pre_constrained
                post_constrained_sum += post_constrained
                pre_weighted_sum += pre_weighted
                post_weighted_sum += post_weighted
                if ripple_pre is not None:
                    ripple_pre_sum += ripple_pre
                    ripple_count += 1
                if ripple_post is not None:
                    ripple_post_sum += ripple_post
                if stop_pre is not None:
                    stop_pre_sum += stop_pre
                    stop_count += 1
                if stop_post is not None:
                    stop_post_sum += stop_post

                if uniform_enabled:
                    fc_val = float(fc_hz[b].item())
                    mult = float(args.uniform_grid_mult)
                    if not math.isfinite(fc_val) or fc_val <= 0.0:
                        f_min = float(freq[b].min().item())
                        f_max = float(freq[b].max().item())
                    else:
                        f_min = fc_val / mult if mult > 0 else fc_val / 10.0
                        f_max = fc_val * mult if mult > 0 else fc_val * 10.0
                        if f_min <= 0.0 or not math.isfinite(f_min):
                            f_min = fc_val / 10.0
                        if f_max <= 0.0 or not math.isfinite(f_max):
                            f_max = fc_val * 10.0
                    if f_min > f_max:
                        f_min, f_max = f_max, f_min
                    freq_u_np = np.logspace(np.log10(f_min), np.log10(f_max), int(args.uniform_grid_points))
                    freq_u = torch.tensor(freq_u_np, device=freq.device, dtype=freq.dtype)
                    target_u = _interp_target(freq[b], target[b], freq_u)
                    mask_min_u = _interp_mask_nearest(freq[b], mask_min[b], freq_u)
                    mask_max_u = _interp_mask_nearest(freq[b], mask_max[b], freq_u)
                    pred_pre_u = circuit(freq_u, values=values_vec, output="s21_db")
                    values_vec_post_u = values_vec_post
                    pred_post_u = circuit(freq_u, values=values_vec_post_u, output="s21_db")
                    if torch.isfinite(target_u).all() and torch.isfinite(pred_pre_u).all() and torch.isfinite(pred_post_u).all():
                        uni_pre_mse, uni_pre_con, uni_pre_w = _masked_mse_components(
                            pred_pre_u,
                            target_u,
                            mask_min_u,
                            mask_max_u,
                            w_pass=w_pass,
                            w_stop=w_stop,
                        )
                        uni_post_mse, uni_post_con, uni_post_w = _masked_mse_components(
                            pred_post_u,
                            target_u,
                            mask_min_u,
                            mask_max_u,
                            w_pass=w_pass,
                            w_stop=w_stop,
                        )
                        uni_pre_ripple, uni_pre_stop = _band_metrics(pred_pre_u, mask_min_u, mask_max_u)
                        uni_post_ripple, uni_post_stop = _band_metrics(pred_post_u, mask_min_u, mask_max_u)
                        uni_pre_mse_sum += uni_pre_mse
                        uni_post_mse_sum += uni_post_mse
                        uni_pre_constrained_sum += uni_pre_con
                        uni_post_constrained_sum += uni_post_con
                        uni_pre_weighted_sum += uni_pre_w
                        uni_post_weighted_sum += uni_post_w
                        if uni_pre_ripple is not None:
                            uni_ripple_pre_sum += uni_pre_ripple
                            uni_ripple_count += 1
                        if uni_post_ripple is not None:
                            uni_ripple_post_sum += uni_post_ripple
                        if uni_pre_stop is not None:
                            uni_stop_pre_sum += uni_pre_stop
                            uni_stop_count += 1
                        if uni_post_stop is not None:
                            uni_stop_post_sum += uni_post_stop
                        uni_total += 1
                        ft_id_u = int(filter_type[b].item())
                        ft_name_u = type_names.get(ft_id_u, "unknown")
                        if ft_name_u not in per_type_uniform:
                            per_type_uniform[ft_name_u] = _new_uniform_group()
                        ugroup = per_type_uniform[ft_name_u]
                        ugroup["count"] += 1
                        ugroup["mse_pre"] += uni_pre_mse
                        ugroup["mse_post"] += uni_post_mse
                        ugroup["constrained_mse_pre"] += uni_pre_con
                        ugroup["constrained_mse_post"] += uni_post_con
                        ugroup["weighted_mse_pre"] += uni_pre_w
                        ugroup["weighted_mse_post"] += uni_post_w
                        if uni_pre_ripple is not None:
                            ugroup["ripple_pre_sum"] += uni_pre_ripple
                            ugroup["ripple_count"] += 1
                        if uni_post_ripple is not None:
                            ugroup["ripple_post_sum"] += uni_post_ripple
                        if uni_pre_stop is not None:
                            ugroup["stop_pre_sum"] += uni_pre_stop
                            ugroup["stop_count"] += 1
                        if uni_post_stop is not None:
                            ugroup["stop_post_sum"] += uni_post_stop
                    else:
                        ft_id_u = int(filter_type[b].item())
                        ft_name_u = type_names.get(ft_id_u, "unknown")
                        if ft_name_u not in per_type_uniform:
                            per_type_uniform[ft_name_u] = _new_uniform_group()
                        per_type_uniform[ft_name_u]["failed"] += 1

                total += 1
                ft_id = int(filter_type[b].item())
                ft_name = type_names.get(ft_id, "unknown")
                if ft_name not in per_type:
                    per_type[ft_name] = _new_group()
                group = per_type[ft_name]
                group["count"] += 1
                group["mse_pre"] += pre_mse
                group["mse_post"] += post_mse
                group["constrained_mse_pre"] += pre_constrained
                group["constrained_mse_post"] += post_constrained
                group["weighted_mse_pre"] += pre_weighted
                group["weighted_mse_post"] += post_weighted
                if ripple_pre is not None:
                    group["ripple_pre_sum"] += ripple_pre
                    group["ripple_count"] += 1
                if ripple_post is not None:
                    group["ripple_post_sum"] += ripple_post
                if stop_pre is not None:
                    group["stop_pre_sum"] += stop_pre
                    group["stop_count"] += 1
                if stop_post is not None:
                    group["stop_post_sum"] += stop_post
                macro_slot_total += slot_total
                macro_slot_correct += slot_correct
                macro_non_skip_total += non_skip_total
                macro_non_skip_correct += non_skip_correct
                len_abs_sum += len_abs
                len_bias_sum += len_bias
                len_exact += len_exact_flag
                group["macro_slot_total"] += slot_total
                group["macro_slot_correct"] += slot_correct
                group["macro_non_skip_total"] += non_skip_total
                group["macro_non_skip_correct"] += non_skip_correct
                group["len_abs_sum"] += len_abs
                group["len_bias_sum"] += len_bias
                group["len_exact"] += len_exact_flag

                if _has_constraints(mask_min[b], mask_max[b]):
                    yield_total += 1
                    oracle_s11 = target_s11[b] if use_s11_yield and target_s11 is not None else None
                    oracle_max = float(
                        calc_violation_max(
                            target[b],
                            mask_min[b],
                            mask_max[b],
                            pred_s11_db=oracle_s11,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    pre_max = float(
                        calc_violation_max(
                            pred_pre,
                            mask_min[b],
                            mask_max[b],
                            pred_s11_db=pred_pre_s11,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    post_max = float(
                        calc_violation_max(
                            pred_post,
                            mask_min[b],
                            mask_max[b],
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
                                mask_min[b],
                                mask_max[b],
                                alpha=alpha_val,
                                pred_s11_db=oracle_s11,
                                s11_max_db=yield_s11_max_db,
                            ).item()
                        )
                        pre_q = float(
                            calc_violation_quantile(
                                pred_pre,
                                mask_min[b],
                                mask_max[b],
                                alpha=alpha_val,
                                pred_s11_db=pred_pre_s11,
                                s11_max_db=yield_s11_max_db,
                            ).item()
                        )
                        post_q = float(
                            calc_violation_quantile(
                                pred_post,
                                mask_min[b],
                                mask_max[b],
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
                    group["yield_total"] += 1
                    if oracle_max <= primary_tau:
                        group["yield_oracle"] += 1
                    if pre_max <= primary_tau:
                        group["yield_pre"] += 1
                    if post_max <= primary_tau:
                        group["yield_post"] += 1
            except Exception:
                failed += 1
                ft_id = int(filter_type[b].item())
                ft_name = type_names.get(ft_id, "unknown")
                if ft_name not in per_type:
                    per_type[ft_name] = _new_group()
                per_type[ft_name]["failed"] += 1
                continue

    per_type_out = {}
    for name, group in per_type.items():
        count = group["count"]
        ytot = group["yield_total"]
        slot_total = group["macro_slot_total"]
        non_skip_total = group["macro_non_skip_total"]
        per_type_out[name] = {
            "count": count,
            "failed": group["failed"],
            "mse_pre": group["mse_pre"] / max(1, count),
            "mse_post": group["mse_post"] / max(1, count),
            "constrained_mse_pre": group["constrained_mse_pre"] / max(1, count),
            "constrained_mse_post": group["constrained_mse_post"] / max(1, count),
            "weighted_mse_pre": group["weighted_mse_pre"] / max(1, count),
            "weighted_mse_post": group["weighted_mse_post"] / max(1, count),
            "ripple_pre": (group["ripple_pre_sum"] / group["ripple_count"]) if group["ripple_count"] else None,
            "ripple_post": (group["ripple_post_sum"] / group["ripple_count"]) if group["ripple_count"] else None,
            "stopband_max_pre": (group["stop_pre_sum"] / group["stop_count"]) if group["stop_count"] else None,
            "stopband_max_post": (group["stop_post_sum"] / group["stop_count"]) if group["stop_count"] else None,
            "macro_acc": (group["macro_slot_correct"] / slot_total) if slot_total else None,
            "macro_non_skip_acc": (group["macro_non_skip_correct"] / non_skip_total) if non_skip_total else None,
            "len_mae": (group["len_abs_sum"] / max(1, count)),
            "len_bias": (group["len_bias_sum"] / max(1, count)),
            "len_exact": (group["len_exact"] / max(1, count)),
            "yield_total": ytot,
            "yield_oracle": (group["yield_oracle"] / ytot) if ytot else None,
            "yield_pre": (group["yield_pre"] / ytot) if ytot else None,
            "yield_post": (group["yield_post"] / ytot) if ytot else None,
        }

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
        "nonfinite_logits": nonfinite_logits,
        "nonfinite_slot": nonfinite_slot,
        "nonfinite_target": nonfinite_target,
        "nonfinite_pred_pre": nonfinite_pred_pre,
        "nonfinite_pred_post": nonfinite_pred_post,
        "mse_pre": pre_mse_sum / max(1, total),
        "mse_post": post_mse_sum / max(1, total),
        "constrained_mse_pre": pre_constrained_sum / max(1, total),
        "constrained_mse_post": post_constrained_sum / max(1, total),
        "weighted_mse_pre": pre_weighted_sum / max(1, total),
        "weighted_mse_post": post_weighted_sum / max(1, total),
        "ripple_pre": (ripple_pre_sum / ripple_count) if ripple_count else None,
        "ripple_post": (ripple_post_sum / ripple_count) if ripple_count else None,
        "stopband_max_pre": (stop_pre_sum / stop_count) if stop_count else None,
        "stopband_max_post": (stop_post_sum / stop_count) if stop_count else None,
        "macro_acc": (macro_slot_correct / macro_slot_total) if macro_slot_total else None,
        "macro_non_skip_acc": (macro_non_skip_correct / macro_non_skip_total) if macro_non_skip_total else None,
        "len_mae": (len_abs_sum / max(1, total)),
        "len_bias": (len_bias_sum / max(1, total)),
        "len_exact": (len_exact / max(1, total)),
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
        "per_filter_type": per_type_out,
        "target_wave": str(target_wave),
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
    }

    if uniform_enabled:
        per_type_uniform_out = {}
        for name, group in per_type_uniform.items():
            count = group["count"]
            per_type_uniform_out[name] = {
                "count": count,
                "failed": group["failed"],
                "mse_pre": group["mse_pre"] / max(1, count),
                "mse_post": group["mse_post"] / max(1, count),
                "constrained_mse_pre": group["constrained_mse_pre"] / max(1, count),
                "constrained_mse_post": group["constrained_mse_post"] / max(1, count),
                "weighted_mse_pre": group["weighted_mse_pre"] / max(1, count),
                "weighted_mse_post": group["weighted_mse_post"] / max(1, count),
                "ripple_pre": (group["ripple_pre_sum"] / group["ripple_count"]) if group["ripple_count"] else None,
                "ripple_post": (group["ripple_post_sum"] / group["ripple_count"]) if group["ripple_count"] else None,
                "stopband_max_pre": (group["stop_pre_sum"] / group["stop_count"]) if group["stop_count"] else None,
                "stopband_max_post": (group["stop_post_sum"] / group["stop_count"]) if group["stop_count"] else None,
            }
        results["uniform_grid"] = {
            "num_samples": uni_total,
            "mse_pre": (uni_pre_mse_sum / uni_total) if uni_total else None,
            "mse_post": (uni_post_mse_sum / uni_total) if uni_total else None,
            "constrained_mse_pre": (uni_pre_constrained_sum / uni_total) if uni_total else None,
            "constrained_mse_post": (uni_post_constrained_sum / uni_total) if uni_total else None,
            "weighted_mse_pre": (uni_pre_weighted_sum / uni_total) if uni_total else None,
            "weighted_mse_post": (uni_post_weighted_sum / uni_total) if uni_total else None,
            "ripple_pre": (uni_ripple_pre_sum / uni_ripple_count) if uni_ripple_count else None,
            "ripple_post": (uni_ripple_post_sum / uni_ripple_count) if uni_ripple_count else None,
            "stopband_max_pre": (uni_stop_pre_sum / uni_stop_count) if uni_stop_count else None,
            "stopband_max_post": (uni_stop_post_sum / uni_stop_count) if uni_stop_count else None,
            "per_filter_type": per_type_uniform_out,
        }

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
