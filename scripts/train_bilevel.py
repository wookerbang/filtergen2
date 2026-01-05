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
from torch.utils.data import DataLoader, Dataset

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import MACRO_IDS, MACRO_LIBRARY, SERIES_MACROS, dsl_tokens_to_macro_sequence
from src.models import Wave2StructureModel
from src.physics.differentiable_rf import DynamicCircuitAssembler, unroll_refine_slots


class BilevelDataset(Dataset):
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

        dsl_tokens = s.get("dsl_tokens") or []
        return {
            "freq": freq,
            "wave": wave,
            "scalar": scalar,
            "ideal_s21_db": ideal_s21,
            "dsl_tokens": dsl_tokens,
            "dsl_macros": self.dsl_macros[idx],
        }


def _scan_macro_vocab_and_k(
    jsonl_path: str,
    *,
    k_percentile: float,
    k_cap: int,
    k_min: int,
) -> Tuple[List[str], int]:
    macro_set = set()
    cell_counts: List[int] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            toks = sample.get("dsl_tokens") or []
            if not toks:
                raise ValueError("DSL tokens missing in dataset; bilevel training requires dsl_tokens.")
            macros = dsl_tokens_to_macro_sequence(toks, strict=True)
            if not macros:
                raise ValueError("Empty macro sequence parsed from dsl_tokens.")
            macro_set.update(macros)
            cell_counts.append(len(macros))
    if not cell_counts:
        raise ValueError("No valid DSL samples found for macro/vocab scan.")

    macro_vocab = [m for m in MACRO_IDS if m in macro_set]
    if not macro_vocab:
        raise ValueError("Macro vocab is empty after scanning dataset.")

    p = float(np.percentile(np.asarray(cell_counts, dtype=float), float(k_percentile)))
    k_est = int(math.ceil(p))
    k_max = min(int(k_cap), max(int(k_min), k_est))
    return macro_vocab, k_max


def _enforce_non_empty(macro_ids: torch.Tensor, g_logits: torch.Tensor, skip_id: int) -> torch.Tensor:
    if bool((macro_ids != skip_id).any()):
        return macro_ids
    # pick the best non-skip macro for one cell
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


def make_collate_fn(macro_to_id: dict, *, skip_id: int, k_max: int):
    def collate(batch: List[dict]) -> dict:
        waves = torch.stack([b["wave"] for b in batch])
        scalars = torch.stack([b["scalar"] for b in batch])
        freq = torch.stack([b["freq"] for b in batch])
        target = torch.stack([b["ideal_s21_db"] for b in batch])

        macro_ids = torch.full((len(batch), k_max), int(skip_id), dtype=torch.long)
        for i, b in enumerate(batch):
            macros = b.get("dsl_macros")
            if macros is None:
                macros = dsl_tokens_to_macro_sequence(b["dsl_tokens"], strict=True)
            if len(macros) > k_max:
                macros = macros[:k_max]
            for j, m in enumerate(macros):
                if m not in macro_to_id:
                    raise ValueError(f"Macro {m} missing from macro vocab.")
                macro_ids[i, j] = int(macro_to_id[m])

        return {
            "wave": waves,
            "filter_type": scalars[:, 0].long(),
            "fc_hz": scalars[:, 1],
            "freq": freq,
            "target_s21_db": target,
            "macro_ids": macro_ids,
        }

    return collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bilevel training with DSL projection.")
    p.add_argument("--data", type=Path, required=True, help="Path to train jsonl.")
    p.add_argument("--eval-data", type=Path, help="Optional eval jsonl.")
    p.add_argument("--output", type=Path, default=Path("checkpoints/bilevel"), help="Checkpoint dir.")
    p.add_argument("--init-from", type=Path, help="Initialize from checkpoint (pytorch_model.bin or dir).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp-bf16", action="store_true", help="Enable BF16 autocast (model forward only).")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (num_workers>0).")
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory for faster H2D copies.")
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive.")
    p.add_argument("--clip-grad", type=float, default=5.0, help="Clip gradient norm (<=0 disables).")
    p.add_argument("--skip-nonfinite", dest="skip_nonfinite", action="store_true", help="Skip updates on non-finite batches.")
    p.add_argument("--no-skip-nonfinite", dest="skip_nonfinite", action="store_false")
    p.set_defaults(skip_nonfinite=False)

    # input config
    p.add_argument("--use-wave", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"], default="ideal")
    p.add_argument("--wave-norm", action="store_true")
    p.add_argument("--freq-mode", choices=["none", "log_fc", "linear_fc", "log_f", "log_f_centered"], default="log_f_centered")
    p.add_argument("--freq-scale", choices=["none", "log_fc", "log_f_mean"], default="log_f_mean")
    p.add_argument("--spec-mode", choices=["none", "type_fc"], default="type_fc")
    p.add_argument("--no-s11", dest="include_s11", action="store_false")
    p.set_defaults(include_s11=True)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--hidden-mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # bilevel config
    p.add_argument("--k-percentile", type=float, default=95.0)
    p.add_argument("--k-cap", type=int, default=12)
    p.add_argument("--k-min", type=int, default=12)
    p.add_argument("--unroll-steps", type=int, default=5)
    p.add_argument("--inner-lr", type=float, default=1e-2)
    p.add_argument("--inner-max-step", type=float, default=0.5)
    p.add_argument("--inner-raw-min", type=float, default=-32.0)
    p.add_argument("--inner-raw-max", type=float, default=-12.0)
    p.add_argument("--inner-nan-backoff", type=float, default=0.5)
    p.add_argument("--inner-nan-tries", type=int, default=3)
    p.add_argument("--phys-weight", type=float, default=1e-4)
    p.add_argument("--len-weight", type=float, default=1e-3)
    p.add_argument("--gumbel-tau", type=float, default=1.0)
    p.add_argument("--alpha-start", type=float, default=1.0)
    p.add_argument("--alpha-min", type=float, default=0.1)
    p.add_argument("--alpha-decay-frac", type=float, default=0.3)
    p.add_argument("--use-token-loss", action="store_true", help="(Reserved) include token loss during bilevel.")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return p.parse_args()


def _alpha_schedule(step: int, total_steps: int, *, alpha_start: float, alpha_min: float, decay_frac: float) -> float:
    if total_steps <= 0:
        return float(alpha_min)
    decay_steps = max(1, int(total_steps * float(decay_frac)))
    if step >= decay_steps:
        return float(alpha_min)
    t = 1.0 - float(step) / float(decay_steps)
    return float(alpha_min) + (float(alpha_start) - float(alpha_min)) * t


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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    macro_vocab, k_max = _scan_macro_vocab_and_k(
        str(args.data), k_percentile=args.k_percentile, k_cap=args.k_cap, k_min=args.k_min
    )
    macro_to_id = {m: i for i, m in enumerate(macro_vocab)}
    id_to_macro = list(macro_vocab)
    skip_id = len(macro_vocab)
    slot_count = max(len(MACRO_LIBRARY[m].slot_types) for m in macro_vocab)
    macro_slot_lens = [len(MACRO_LIBRARY[m].slot_types) for m in macro_vocab] + [0]
    macro_slot_mask = torch.zeros((len(macro_vocab) + 1, slot_count), dtype=torch.float32)
    for mid, slen in enumerate(macro_slot_lens):
        if slen > 0:
            macro_slot_mask[mid, :slen] = 1.0

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = {
        "k_max": k_max,
        "k_cap": args.k_cap,
        "k_percentile": args.k_percentile,
        "k_min": args.k_min,
        "macro_vocab": macro_vocab,
        "macro_vocab_size": len(macro_vocab),
        "slot_count": slot_count,
        "use_wave": args.use_wave,
        "freq_mode": args.freq_mode,
        "freq_scale": args.freq_scale,
        "include_s11": bool(args.include_s11),
        "spec_mode": args.spec_mode,
        "d_model": args.d_model,
        "hidden_mult": args.hidden_mult,
        "dropout": args.dropout,
        "amp_bf16": bool(args.amp_bf16),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers),
        "clip_grad": args.clip_grad,
        "skip_nonfinite": bool(args.skip_nonfinite),
        "alpha_start": args.alpha_start,
        "alpha_min": args.alpha_min,
        "alpha_decay_frac": args.alpha_decay_frac,
        "len_weight": args.len_weight,
        "unroll_steps": args.unroll_steps,
        "inner_lr": args.inner_lr,
        "inner_max_step": args.inner_max_step,
        "inner_raw_min": args.inner_raw_min,
        "inner_raw_max": args.inner_raw_max,
        "inner_nan_backoff": args.inner_nan_backoff,
        "inner_nan_tries": args.inner_nan_tries,
        "phys_weight": args.phys_weight,
        "gumbel_tau": args.gumbel_tau,
        "init_from": str(args.init_from) if args.init_from else None,
    }
    with (args.output / "input_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    dataset = BilevelDataset(
        str(args.data),
        use_wave=args.use_wave,
        normalize_wave=args.wave_norm,
        freq_mode=args.freq_mode,
        freq_scale=args.freq_scale,
        include_s11=args.include_s11,
    )
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
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=make_collate_fn(macro_to_id, skip_id=skip_id, k_max=k_max),
        **loader_kwargs,
    )
    non_blocking = bool(pin_memory)

    in_channels = dataset[0]["wave"].shape[0]
    model = Wave2StructureModel(
        k_max=k_max,
        macro_vocab_size=len(macro_vocab),
        slot_count=slot_count,
        waveform_in_channels=in_channels,
        d_model=args.d_model,
        hidden_mult=args.hidden_mult,
        dropout=args.dropout,
        spec_mode=args.spec_mode,
    ).to(device=device)
    if args.init_from:
        ckpt_path = _resolve_ckpt(args.init_from)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        print(f"[init] loaded weights from {ckpt_path}")
    macro_slot_mask = macro_slot_mask.to(device=args.device)

    if args.use_token_loss:
        raise ValueError("--use-token-loss is reserved; bilevel model has no decoder.")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    amp_enabled = bool(args.amp_bf16 and device.type == "cuda" and dtype == torch.float32)
    if args.amp_bf16 and not amp_enabled:
        print("[warn] BF16 autocast disabled (requires CUDA + float32 dtype).")
    assembler = DynamicCircuitAssembler(z0=50.0)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    step = 0
    total_steps = int(args.epochs) * max(1, math.ceil(len(dataset) / max(1, args.batch_size)))

    model.train()
    skipped_nonfinite = 0
    for epoch in range(int(args.epochs)):
        for batch in loader:
            step += 1
            wave = batch["wave"].to(device, dtype=dtype, non_blocking=non_blocking)
            filter_type = batch["filter_type"].to(device, non_blocking=non_blocking)
            fc_hz = batch["fc_hz"].to(device, dtype=dtype, non_blocking=non_blocking)
            freq = batch["freq"].to(device, dtype=dtype, non_blocking=non_blocking)
            target = batch["target_s21_db"].to(device, dtype=dtype, non_blocking=non_blocking)
            macro_targets = batch["macro_ids"].to(device, non_blocking=non_blocking)
            if args.skip_nonfinite:
                if not (torch.isfinite(wave).all() and torch.isfinite(freq).all() and torch.isfinite(target).all()):
                    skipped_nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    continue

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                g_logits, slot_raw = model(wave, filter_type=filter_type, fc_hz=fc_hz)
            if amp_enabled:
                g_logits = g_logits.float()
                slot_raw = slot_raw.float()
            slot_raw = slot_raw.to(dtype)
            if args.skip_nonfinite and not (torch.isfinite(g_logits).all() and torch.isfinite(slot_raw).all()):
                skipped_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                continue

            g_soft = F.gumbel_softmax(g_logits, tau=float(args.gumbel_tau), hard=False, dim=-1)
            g_hard = F.gumbel_softmax(g_logits, tau=float(args.gumbel_tau), hard=True, dim=-1)

            macro_ce = F.cross_entropy(g_logits.view(-1, skip_id + 1), macro_targets.view(-1))
            p_skip = g_soft[..., skip_id]
            len_loss = (1.0 - p_skip).mean()

            physics_losses = []
            for b in range(wave.shape[0]):
                macro_ids_raw = torch.argmax(g_hard[b], dim=-1)
                was_empty = not bool((macro_ids_raw != skip_id).any())
                macro_ids_hard = _enforce_non_empty(macro_ids_raw, g_logits[b], skip_id)
                hard_mask = torch.matmul(g_hard[b], macro_slot_mask)
                if was_empty:
                    hard_mask = macro_slot_mask[macro_ids_hard]
                soft_mask = torch.matmul(g_soft[b], macro_slot_mask)
                slot_mask = (hard_mask - soft_mask.detach() + soft_mask).to(dtype)
                circuit, slot_idx = _build_circuit_and_indices(
                    macro_ids_hard,
                    id_to_macro=id_to_macro,
                    skip_id=skip_id,
                    slot_count=slot_count,
                    assembler=assembler,
                    device=wave.device,
                    dtype=dtype,
                )
                loss_b = unroll_refine_slots(
                    slot_raw[b],
                    slot_mask,
                    slot_idx,
                    circuit,
                    freq[b],
                    target[b],
                    steps=args.unroll_steps,
                    lr=args.inner_lr,
                    max_step=args.inner_max_step,
                    raw_min=args.inner_raw_min,
                    raw_max=args.inner_raw_max,
                    nan_backoff=args.inner_nan_backoff,
                    max_backoff=args.inner_nan_tries,
                )
                physics_losses.append(loss_b)
            physics_loss = torch.stack(physics_losses).mean()

            alpha = _alpha_schedule(step, total_steps, alpha_start=args.alpha_start, alpha_min=args.alpha_min, decay_frac=args.alpha_decay_frac)
            phys_weight = float(args.phys_weight)
            loss = phys_weight * physics_loss + float(alpha) * macro_ce + float(args.len_weight) * len_loss
            if args.skip_nonfinite and not torch.isfinite(loss):
                skipped_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()

            if step % int(args.grad_accum) == 0:
                if float(args.clip_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % int(args.log_steps) == 0:
                print(
                    f"[epoch {epoch+1}] step={step} loss={loss.item():.4f} "
                    f"phys={physics_loss.item():.4f} phys_w={phys_weight:.1e} "
                    f"macro_ce={macro_ce.item():.4f} len={len_loss.item():.4f} alpha={alpha:.3f}"
                    + (f" skipped={skipped_nonfinite}" if skipped_nonfinite else "")
                )

            if int(args.save_steps) > 0 and step % int(args.save_steps) == 0:
                ckpt = args.output / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt / "pytorch_model.bin")

        # epoch checkpoint
        ckpt = args.output / f"epoch_{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / "pytorch_model.bin")


if __name__ == "__main__":
    main()
