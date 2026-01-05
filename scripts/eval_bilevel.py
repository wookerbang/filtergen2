"""Evaluate bilevel model with pre-/post-refine metrics and yield."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import MACRO_LIBRARY, SERIES_MACROS, dsl_tokens_to_macro_sequence
from src.models import Wave2StructureModel
from src.physics.differentiable_rf import DynamicCircuitAssembler, unroll_refine_slots


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
        self.dsl_macros = []
        with open(jsonl_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                tokens = sample.get("dsl_tokens") or []
                if not tokens:
                    raise ValueError(f"Missing dsl_tokens at line {line_no} in {jsonl_path}.")
                macros = dsl_tokens_to_macro_sequence(tokens, strict=True)
                if not macros:
                    raise ValueError(f"Empty macro sequence at line {line_no} in {jsonl_path}.")
                self.samples.append(sample)
                self.dsl_macros.append(macros)
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
            "dsl_macros": self.dsl_macros[idx],
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            "ideal_s21_db": ideal_s21,
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


def _mask_satisfied(pred_db: torch.Tensor, mask_min: torch.Tensor, mask_max: torch.Tensor) -> bool:
    ok = torch.ones_like(pred_db, dtype=torch.bool)
    ok &= torch.isnan(mask_min) | (pred_db >= mask_min)
    ok &= torch.isnan(mask_max) | (pred_db <= mask_max)
    return bool(ok.all().item())


def _has_constraints(mask_min: torch.Tensor, mask_max: torch.Tensor) -> bool:
    return bool(torch.isfinite(mask_min).any().item() or torch.isfinite(mask_max).any().item())


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
    p.add_argument("--spec-mode", choices=["none", "type_fc"], default=None)
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
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return p.parse_args()


def _new_group() -> dict:
    return {
        "count": 0,
        "mse_pre": 0.0,
        "mse_post": 0.0,
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
    d_model = int(cfg.get("d_model", 512))
    hidden_mult = int(cfg.get("hidden_mult", 2))
    dropout = float(cfg.get("dropout", 0.1))
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
        dataset.dsl_macros = dataset.dsl_macros[:max_n]

    def collate(batch: List[dict]) -> dict:
        return {
            "wave": torch.stack([b["wave"] for b in batch]),
            "freq": torch.stack([b["freq"] for b in batch]),
            "target_s21_db": torch.stack([b["ideal_s21_db"] for b in batch]),
            "mask_min_db": torch.stack([b["mask_min_db"] for b in batch]),
            "mask_max_db": torch.stack([b["mask_max_db"] for b in batch]),
            "scalar": torch.stack([b["scalar"] for b in batch]),
            "dsl_tokens": [b["dsl_tokens"] for b in batch],
            "dsl_macros": [b.get("dsl_macros") for b in batch],
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

    type_names = {0: "lowpass", 1: "highpass", 2: "bandpass", 3: "bandstop"}
    total = 0
    pre_mse_sum = 0.0
    post_mse_sum = 0.0
    yield_total = 0
    yield_pre_pass = 0
    yield_post_pass = 0
    yield_oracle_pass = 0
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
        mask_min = batch["mask_min_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        mask_max = batch["mask_max_db"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        scalar = batch["scalar"].to(device=device, dtype=dtype, non_blocking=non_blocking)
        dsl_tokens = batch["dsl_tokens"]
        dsl_macros = batch.get("dsl_macros")
        filter_type = scalar[:, 0].long()
        fc_hz = scalar[:, 1]

        with torch.no_grad():
            g_logits, slot_raw = model(wave, filter_type=filter_type, fc_hz=fc_hz)
        g_logits = g_logits.float()
        slot_raw = slot_raw.float()

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
                if not torch.isfinite(pred_pre).all():
                    nonfinite_pred_pre += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue
                pre_mse = F.mse_loss(pred_pre, target[b]).item()
                if not math.isfinite(pre_mse):
                    nonfinite_pred_pre += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue

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
                )
                raw_post = raw_post.to(dtype)
                values_flat_post = torch.exp(raw_post.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                values_vec_post = values_flat_post.index_select(0, slot_idx)
                pred_post = circuit(freq[b], values=values_vec_post, output="s21_db")
                if not torch.isfinite(pred_post).all():
                    nonfinite_pred_post += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue
                post_mse = F.mse_loss(pred_post, target[b]).item()
                if not math.isfinite(post_mse):
                    nonfinite_pred_post += 1
                    failed += 1
                    ft_id = int(filter_type[b].item())
                    ft_name = type_names.get(ft_id, "unknown")
                    if ft_name not in per_type:
                        per_type[ft_name] = _new_group()
                    per_type[ft_name]["failed"] += 1
                    continue

                macros = None
                if dsl_macros is not None:
                    macros = dsl_macros[b]
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
                total += 1
                ft_id = int(filter_type[b].item())
                ft_name = type_names.get(ft_id, "unknown")
                if ft_name not in per_type:
                    per_type[ft_name] = _new_group()
                group = per_type[ft_name]
                group["count"] += 1
                group["mse_pre"] += pre_mse
                group["mse_post"] += post_mse
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
                    if _mask_satisfied(target[b], mask_min[b], mask_max[b]):
                        yield_oracle_pass += 1
                    if _mask_satisfied(pred_pre, mask_min[b], mask_max[b]):
                        yield_pre_pass += 1
                    if _mask_satisfied(pred_post, mask_min[b], mask_max[b]):
                        yield_post_pass += 1
                    group["yield_total"] += 1
                    if _mask_satisfied(target[b], mask_min[b], mask_max[b]):
                        group["yield_oracle"] += 1
                    if _mask_satisfied(pred_pre, mask_min[b], mask_max[b]):
                        group["yield_pre"] += 1
                    if _mask_satisfied(pred_post, mask_min[b], mask_max[b]):
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
        "macro_acc": (macro_slot_correct / macro_slot_total) if macro_slot_total else None,
        "macro_non_skip_acc": (macro_non_skip_correct / macro_non_skip_total) if macro_non_skip_total else None,
        "len_mae": (len_abs_sum / max(1, total)),
        "len_bias": (len_bias_sum / max(1, total)),
        "len_exact": (len_exact / max(1, total)),
        "yield_total": yield_total,
        "yield_oracle": (yield_oracle_pass / yield_total) if yield_total else None,
        "yield_pre": (yield_pre_pass / yield_total) if yield_total else None,
        "yield_post": (yield_post_pass / yield_total) if yield_total else None,
        "per_filter_type": per_type_out,
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
