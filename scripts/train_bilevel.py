from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict
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
    MACRO_IDS,
    MACRO_LIBRARY,
    MACRO_SER_C,
    MACRO_SER_L,
    MACRO_SER_RESO,
    MACRO_SER_TANK,
    MACRO_SHUNT_C,
    MACRO_SHUNT_L,
    MACRO_SHUNT_NOTCH,
    MACRO_SHUNT_RESO,
    SERIES_MACROS,
    dsl_tokens_to_macro_sequence,
)
from src.utils.macro_transition import build_transition_matrices, expected_transition_penalty
from src.models import Wave2StructureModel
from src.physics.differentiable_rf import (
    DynamicCircuitAssembler,
    DifferentiablePhysicsKernel,
    MacroBankEntry,
    mixed_s21_db,
    unroll_refine_slots,
    unroll_refine_slots_mixed,
)


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
        log_every: int = 0,
    ) -> None:
        self.samples = []
        self.macro_ir_macros = []
        if log_every:
            print(f"[load] reading dataset {jsonl_path}", flush=True)
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
                if log_every and line_no % int(log_every) == 0:
                    print(f"[load] parsed {line_no} lines", flush=True)
        self.use_wave = use_wave
        self.mix_real_prob = mix_real_prob
        self.normalize_wave = normalize_wave
        self.freq_mode = freq_mode
        self.freq_scale = freq_scale
        self.include_s11 = include_s11
        if log_every:
            print(f"[load] finished: {len(self.samples)} samples", flush=True)

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
            "macro_ir_macros": self.macro_ir_macros[idx],
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
            macros = sample.get("macro_ir_macros") or []
            if not macros:
                toks = sample.get("dsl_tokens") or []
                if not toks:
                    raise ValueError("macro_ir_macros/dsl_tokens missing in dataset; bilevel training requires macros.")
                macros = dsl_tokens_to_macro_sequence(toks, strict=True)
            if not macros:
                raise ValueError("Empty macro sequence parsed from dataset.")
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


def _scan_macro_vocab_and_k_from_sequences(
    sequences: List[List[str]],
    *,
    k_percentile: float,
    k_cap: int,
    k_min: int,
) -> Tuple[List[str], int]:
    macro_set = set()
    cell_counts: List[int] = []
    for macros in sequences:
        if not macros:
            continue
        macro_set.update(macros)
        cell_counts.append(len(macros))
    if not cell_counts:
        raise ValueError("No valid macro sequences found for macro/vocab scan.")

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


def _build_macro_bank(
    *,
    id_to_macro: List[str],
    slot_count: int,
    assembler: DynamicCircuitAssembler,
    device: torch.device,
    dtype: torch.dtype,
) -> List[MacroBankEntry]:
    entries: List[MacroBankEntry] = []
    op_map = {
        MACRO_SER_L: ([DifferentiablePhysicsKernel.OP_SERIES_L], [1]),
        MACRO_SER_C: ([DifferentiablePhysicsKernel.OP_SERIES_C], [1]),
        MACRO_SHUNT_L: ([DifferentiablePhysicsKernel.OP_SHUNT_L], [1]),
        MACRO_SHUNT_C: ([DifferentiablePhysicsKernel.OP_SHUNT_C], [1]),
        MACRO_SER_RESO: ([DifferentiablePhysicsKernel.OP_SERIES_L, DifferentiablePhysicsKernel.OP_SERIES_C], [1, 1]),
        MACRO_SER_TANK: ([DifferentiablePhysicsKernel.OP_SERIES_PARALLEL_LC], [2]),
        MACRO_SHUNT_RESO: ([DifferentiablePhysicsKernel.OP_SHUNT_L, DifferentiablePhysicsKernel.OP_SHUNT_C], [1, 1]),
        MACRO_SHUNT_NOTCH: ([DifferentiablePhysicsKernel.OP_SHUNT_SERIES_LC], [2]),
    }
    base = 1_000_000.0
    for macro in id_to_macro:
        if macro not in MACRO_LIBRARY:
            raise ValueError(f"Macro {macro} missing from MACRO_LIBRARY.")
        macro_def = MACRO_LIBRARY[macro]
        if macro in op_map:
            op_codes, op_param_counts = op_map[macro]
            slot_idx_order = list(range(len(macro_def.slot_types)))
        else:
            placeholder_vals = [base + j for j in range(len(macro_def.slot_types))]
            if macro in SERIES_MACROS:
                a, b = "in", "out"
            else:
                a, b = "in", "in"
            comps = macro_def.expand_fn(a, b, "gnd", placeholder_vals, 0)
            slot_indices: List[int] = []
            for c in comps:
                slot_idx = int(round(float(c.value_si) - base))
                slot_indices.append(slot_idx)
            circuit, _ = assembler.assemble(comps, trainable=False, device=device, dtype=dtype)
            value_comp_indices = getattr(circuit, "value_comp_indices", None)
            if value_comp_indices is None:
                slot_idx_order = slot_indices
            else:
                slot_idx_order = [slot_indices[int(i)] for i in value_comp_indices]
            op_codes = [int(x) for x in getattr(circuit, "op_codes").detach().cpu().tolist()]
            op_param_counts = getattr(circuit, "_op_param_counts", None)
        entries.append(
            MacroBankEntry(
                op_codes=op_codes,
                op_param_counts=op_param_counts,
                slot_idx=torch.tensor(slot_idx_order, device=device, dtype=torch.long),
            )
        )
    return entries


class CircuitCache:
    def __init__(
        self,
        *,
        max_size: int,
        assembler: DynamicCircuitAssembler,
        id_to_macro: List[str],
        skip_id: int,
        slot_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_size = max(0, int(max_size))
        self.assembler = assembler
        self.id_to_macro = id_to_macro
        self.skip_id = skip_id
        self.slot_count = slot_count
        self.device = device
        self.dtype = dtype
        self._cache: OrderedDict[tuple[int, ...], tuple[object, torch.Tensor]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, macro_ids: torch.Tensor) -> tuple[object, torch.Tensor]:
        if self.max_size <= 0:
            return _build_circuit_and_indices(
                macro_ids,
                id_to_macro=self.id_to_macro,
                skip_id=self.skip_id,
                slot_count=self.slot_count,
                assembler=self.assembler,
                device=self.device,
                dtype=self.dtype,
            )
        key = tuple(int(x) for x in macro_ids.tolist())
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return hit
        self.misses += 1
        circuit, slot_idx = _build_circuit_and_indices(
            macro_ids,
            id_to_macro=self.id_to_macro,
            skip_id=self.skip_id,
            slot_count=self.slot_count,
            assembler=self.assembler,
            device=self.device,
            dtype=self.dtype,
        )
        self._cache[key] = (circuit, slot_idx)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
            self.evictions += 1
        return circuit, slot_idx

def make_collate_fn(macro_to_id: dict, *, skip_id: int, k_max: int):
    def collate(batch: List[dict]) -> dict:
        waves = torch.stack([b["wave"] for b in batch])
        scalars = torch.stack([b["scalar"] for b in batch])
        freq = torch.stack([b["freq"] for b in batch])
        target = torch.stack([b["ideal_s21_db"] for b in batch])

        macro_ids = torch.full((len(batch), k_max), int(skip_id), dtype=torch.long)
        for i, b in enumerate(batch):
            macros = b.get("macro_ir_macros")
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
    p = argparse.ArgumentParser(description="Bilevel training with Macro-IR slot filling.")
    p.add_argument("--data", type=Path, required=True, help="Path to train jsonl.")
    p.add_argument("--eval-data", type=Path, help="Optional eval jsonl.")
    p.add_argument("--output", type=Path, default=Path("checkpoints/bilevel"), help="Checkpoint dir.")
    p.add_argument("--init-from", type=Path, help="Initialize from checkpoint (pytorch_model.bin or dir).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--log-train-metrics", action="store_true", help="Log macro/length metrics on training batches.")
    p.add_argument("--log-epoch-metrics", action="store_true", help="Log averaged training metrics at epoch end.")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp-bf16", action="store_true", help="Enable BF16 autocast (model forward only).")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (num_workers>0).")
    p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory for faster H2D copies.")
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive.")
    p.add_argument("--circuit-cache-size", type=int, default=2048, help="LRU cache size for compiled circuits (0 disables).")
    p.add_argument("--load-log-steps", type=int, default=10000, help="Log dataset loading progress every N lines (0 disables).")
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
    p.add_argument("--gate-skip-bias", type=float, default=1.0, help="Initial bias for SKIP gate logit.")
    p.add_argument("--use-role-queries", action="store_true", help="Add role-aware embeddings to slot queries.")
    p.add_argument("--role-input-frac", type=float, default=0.2, help="Fraction of slots tagged as input-match.")
    p.add_argument("--role-output-frac", type=float, default=0.2, help="Fraction of slots tagged as output-match.")
    p.add_argument("--sym-weight", type=float, default=0.0, help="Optional symmetry regularizer weight for queries.")
    p.add_argument("--sym-core-only", action="store_true", help="Apply symmetry regularizer only to core slots.")

    # bilevel config
    p.add_argument("--k-percentile", type=float, default=95.0)
    p.add_argument("--k-cap", type=int, default=12)
    p.add_argument("--k-min", type=int, default=12)
    p.add_argument("--matrix-mix", action="store_true", help="Use mixed ABCD relaxation for structure gradients.")
    p.add_argument("--mix-topk", type=int, default=0, help="Use top-k sparse mixing for matrix mix (0 disables).")
    p.add_argument("--unroll-steps", type=int, default=5)
    p.add_argument("--no-unroll", dest="use_unroll", action="store_false", help="Disable inner unroll refinement.")
    p.set_defaults(use_unroll=True)
    p.add_argument(
        "--unroll-create-graph",
        dest="unroll_create_graph",
        action="store_true",
        help="Retain graph for unroll hypergradient (default: on).",
    )
    p.add_argument(
        "--no-unroll-create-graph",
        dest="unroll_create_graph",
        action="store_false",
        help="Disable second-order unroll gradients for stability.",
    )
    p.set_defaults(unroll_create_graph=True)
    p.add_argument("--inner-lr", type=float, default=1e-2)
    p.add_argument("--inner-max-step", type=float, default=0.5)
    p.add_argument("--inner-raw-min", type=float, default=-32.0)
    p.add_argument("--inner-raw-max", type=float, default=-12.0)
    p.add_argument("--inner-nan-backoff", type=float, default=0.5)
    p.add_argument("--inner-nan-tries", type=int, default=3)
    p.add_argument("--phys-weight", type=float, default=1e-4)
    p.add_argument("--len-weight", type=float, default=1e-3)
    p.add_argument("--gumbel-tau", type=float, default=1.0)
    p.add_argument("--gumbel-tau-min", type=float, default=None)
    p.add_argument("--gumbel-tau-decay-frac", type=float, default=0.5)
    p.add_argument("--alpha-start", type=float, default=1.0)
    p.add_argument("--alpha-min", type=float, default=0.1)
    p.add_argument("--alpha-decay-frac", type=float, default=0.3)
    p.add_argument("--use-token-loss", action="store_true", help="(Reserved) include token loss during bilevel.")
    p.add_argument("--c-reg-weight", type=float, default=0.0, help="Weight for transition regularizer (0 disables).")
    p.add_argument("--c-skip-penalty", type=float, default=100.0, help="Soft penalty for SKIP->nonSKIP transitions.")
    p.add_argument(
        "--c-redundant-penalty",
        type=float,
        default=1.0,
        help="Soft penalty for redundant self-transitions (e.g., SER_L->SER_L).",
    )

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


def _gumbel_tau_schedule(step: int, total_steps: int, *, tau_start: float, tau_min: float, decay_frac: float) -> float:
    if total_steps <= 0:
        return float(tau_min)
    decay_steps = max(1, int(total_steps * float(decay_frac)))
    if step >= decay_steps:
        return float(tau_min)
    t = 1.0 - float(step) / float(decay_steps)
    return float(tau_min) + (float(tau_start) - float(tau_min)) * t


def _sparse_topk_probs(probs: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= probs.shape[-1]:
        return probs
    vals, idx = torch.topk(probs, k=k, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, idx, 1.0)
    pruned = probs * mask
    denom = pruned.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return pruned / denom


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
    if args.gumbel_tau_min is None:
        args.gumbel_tau_min = float(args.gumbel_tau)
    if float(args.gumbel_tau_min) <= 0.0:
        raise ValueError("gumbel_tau_min must be > 0.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = BilevelDataset(
        str(args.data),
        use_wave=args.use_wave,
        normalize_wave=args.wave_norm,
        freq_mode=args.freq_mode,
        freq_scale=args.freq_scale,
        include_s11=args.include_s11,
        log_every=args.load_log_steps,
    )
    macro_vocab, k_max = _scan_macro_vocab_and_k_from_sequences(
        dataset.macro_ir_macros,
        k_percentile=args.k_percentile,
        k_cap=args.k_cap,
        k_min=args.k_min,
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
        "matrix_mix": bool(args.matrix_mix),
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
        "gate_skip_bias": args.gate_skip_bias,
        "use_role_queries": bool(args.use_role_queries),
        "role_input_frac": args.role_input_frac,
        "role_output_frac": args.role_output_frac,
        "sym_weight": args.sym_weight,
        "sym_core_only": bool(args.sym_core_only),
        "amp_bf16": bool(args.amp_bf16),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers),
        "log_train_metrics": bool(args.log_train_metrics),
        "log_epoch_metrics": bool(args.log_epoch_metrics),
        "circuit_cache_size": int(args.circuit_cache_size),
        "clip_grad": args.clip_grad,
        "skip_nonfinite": bool(args.skip_nonfinite),
        "alpha_start": args.alpha_start,
        "alpha_min": args.alpha_min,
        "alpha_decay_frac": args.alpha_decay_frac,
        "len_weight": args.len_weight,
        "unroll_steps": args.unroll_steps,
        "use_unroll": bool(args.use_unroll),
        "unroll_create_graph": bool(args.unroll_create_graph),
        "inner_lr": args.inner_lr,
        "inner_max_step": args.inner_max_step,
        "inner_raw_min": args.inner_raw_min,
        "inner_raw_max": args.inner_raw_max,
        "inner_nan_backoff": args.inner_nan_backoff,
        "inner_nan_tries": args.inner_nan_tries,
        "phys_weight": args.phys_weight,
        "gumbel_tau": args.gumbel_tau,
        "gumbel_tau_min": args.gumbel_tau_min,
        "gumbel_tau_decay_frac": args.gumbel_tau_decay_frac,
        "mix_topk": args.mix_topk,
        "c_reg_weight": args.c_reg_weight,
        "c_skip_penalty": args.c_skip_penalty,
        "c_redundant_penalty": args.c_redundant_penalty,
        "init_from": str(args.init_from) if args.init_from else None,
    }
    with (args.output / "input_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
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
        gate_skip_bias=args.gate_skip_bias,
        use_role_queries=bool(args.use_role_queries),
        role_input_frac=float(args.role_input_frac),
        role_output_frac=float(args.role_output_frac),
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
    redundant_macros = [MACRO_SER_L, MACRO_SER_C, MACRO_SHUNT_L, MACRO_SHUNT_C]
    c_hard, p_soft = build_transition_matrices(
        id_to_macro=id_to_macro,
        skip_id=skip_id,
        soft_skip_penalty=float(args.c_skip_penalty),
        soft_redundant_penalty=float(args.c_redundant_penalty),
        redundant_macros=redundant_macros,
        hard_ban_skip_to_non_skip=True,
    )
    c_hard = c_hard.to(device=device, dtype=dtype)
    p_soft = p_soft.to(device=device, dtype=dtype)
    assembler = DynamicCircuitAssembler(z0=50.0)
    circuit_cache = CircuitCache(
        max_size=args.circuit_cache_size,
        assembler=assembler,
        id_to_macro=id_to_macro,
        skip_id=skip_id,
        slot_count=slot_count,
        device=device,
        dtype=dtype,
    )
    macro_bank = None
    if args.matrix_mix:
        macro_bank = _build_macro_bank(
            id_to_macro=id_to_macro,
            slot_count=slot_count,
            assembler=assembler,
            device=device,
            dtype=dtype,
        )

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    step = 0
    total_steps = int(args.epochs) * max(1, math.ceil(len(dataset) / max(1, args.batch_size)))

    model.train()
    skipped_nonfinite = 0
    for epoch in range(int(args.epochs)):
        epoch_samples = 0
        epoch_tokens = 0
        epoch_correct = 0
        epoch_non_skip = 0
        epoch_non_skip_correct = 0
        epoch_len_abs = 0.0
        epoch_len_exact = 0.0
        epoch_loss = 0.0
        epoch_phys = 0.0
        epoch_macro_ce = 0.0
        epoch_len = 0.0
        epoch_c_reg = 0.0
        epoch_sym = 0.0
        epoch_alpha = 0.0
        epoch_tau = 0.0
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

            tau = _gumbel_tau_schedule(
                step,
                total_steps,
                tau_start=float(args.gumbel_tau),
                tau_min=float(args.gumbel_tau_min),
                decay_frac=float(args.gumbel_tau_decay_frac),
            )
            g_soft = F.gumbel_softmax(g_logits, tau=float(tau), hard=False, dim=-1)
            g_soft = g_soft.to(dtype)
            g_phys = g_soft
            if args.matrix_mix and int(args.mix_topk) > 0:
                g_phys = _sparse_topk_probs(g_soft, int(args.mix_topk))

            macro_ce = F.cross_entropy(g_logits.view(-1, skip_id + 1), macro_targets.view(-1))
            p_skip = g_soft[..., skip_id]
            len_loss = (1.0 - p_skip).mean()
            c_reg = torch.tensor(0.0, device=device, dtype=dtype)
            if float(args.c_reg_weight) > 0.0:
                probs = F.softmax(g_logits, dim=-1).to(dtype)
                c_reg = expected_transition_penalty(probs, p_soft)
            sym_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if float(args.sym_weight) > 0.0:
                sym_loss = model.query_symmetry_loss(core_only=bool(args.sym_core_only)).to(dtype)

            metric_note = ""
            if args.log_train_metrics or args.log_epoch_metrics:
                with torch.no_grad():
                    pred_ids = torch.argmax(g_logits, dim=-1)
                    correct = pred_ids.eq(macro_targets)
                    tokens = int(correct.numel())
                    correct_count = int(correct.sum().item())
                    non_skip = macro_targets.ne(skip_id)
                    non_skip_count = int(non_skip.sum().item())
                    non_skip_correct = int((correct & non_skip).sum().item())
                    len_pred = pred_ids.ne(skip_id).sum(dim=1)
                    len_tgt = macro_targets.ne(skip_id).sum(dim=1)
                    len_abs = float((len_pred.float() - len_tgt.float()).abs().sum().item())
                    len_exact = float(len_pred.eq(len_tgt).float().sum().item())
                    if args.log_train_metrics:
                        macro_acc = correct_count / tokens if tokens else 0.0
                        macro_non_skip_acc = non_skip_correct / non_skip_count if non_skip_count else 0.0
                        len_mae = len_abs / float(len_pred.numel())
                        len_exact_rate = len_exact / float(len_pred.numel())
                        metric_note = (
                            f" mac_acc={macro_acc:.3f}"
                            f" mac_ns={macro_non_skip_acc:.3f}"
                            f" len_mae={len_mae:.3f}"
                            f" len_exact={len_exact_rate:.3f}"
                        )

            physics_losses = []
            for b in range(wave.shape[0]):
                if args.matrix_mix:
                    if macro_bank is None:
                        raise ValueError("matrix_mix enabled but macro_bank not initialized.")
                    if args.use_unroll:
                        loss_b = unroll_refine_slots_mixed(
                            slot_raw[b],
                            g_phys[b],
                            macro_bank,
                            freq[b],
                            target[b],
                            steps=args.unroll_steps,
                            lr=args.inner_lr,
                            max_step=args.inner_max_step,
                            raw_min=args.inner_raw_min,
                            raw_max=args.inner_raw_max,
                            nan_backoff=args.inner_nan_backoff,
                            max_backoff=args.inner_nan_tries,
                            create_graph=args.unroll_create_graph,
                        )
                    else:
                        pred = mixed_s21_db(
                            slot_raw[b],
                            g_phys[b],
                            macro_bank,
                            freq[b],
                            raw_min=args.inner_raw_min,
                            raw_max=args.inner_raw_max,
                        )
                        loss_b = F.mse_loss(pred, target[b])
                else:
                    g_hard = F.gumbel_softmax(g_logits[b], tau=float(tau), hard=True, dim=-1)
                    macro_ids_raw = torch.argmax(g_hard, dim=-1)
                    was_empty = not bool((macro_ids_raw != skip_id).any())
                    macro_ids_hard = _enforce_non_empty(macro_ids_raw, g_logits[b], skip_id)
                    hard_mask = torch.matmul(g_hard, macro_slot_mask)
                    if was_empty:
                        hard_mask = macro_slot_mask[macro_ids_hard]
                    soft_mask = torch.matmul(g_soft[b], macro_slot_mask)
                    slot_mask = (hard_mask - soft_mask.detach() + soft_mask).to(dtype)
                    circuit, slot_idx = circuit_cache.get(macro_ids_hard)
                    if args.use_unroll:
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
                            create_graph=args.unroll_create_graph,
                        )
                    else:
                        raw = slot_raw[b].clamp(min=float(args.inner_raw_min), max=float(args.inner_raw_max))
                        values_flat = torch.exp(raw.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
                        values_vec = values_flat.index_select(0, slot_idx)
                        pred = circuit(freq[b], values=values_vec, output="s21_db")
                        loss_b = F.mse_loss(pred, target[b])
                physics_losses.append(loss_b)
            physics_loss = torch.stack(physics_losses).mean()

            alpha = _alpha_schedule(step, total_steps, alpha_start=args.alpha_start, alpha_min=args.alpha_min, decay_frac=args.alpha_decay_frac)
            phys_weight = float(args.phys_weight)
            loss = phys_weight * physics_loss + float(alpha) * macro_ce + float(args.len_weight) * len_loss
            if float(args.c_reg_weight) > 0.0:
                loss = loss + float(args.c_reg_weight) * c_reg
            if float(args.sym_weight) > 0.0:
                loss = loss + float(args.sym_weight) * sym_loss
            if args.skip_nonfinite and not torch.isfinite(loss):
                skipped_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                continue

            if args.log_epoch_metrics:
                batch_size = int(macro_targets.shape[0])
                epoch_samples += batch_size
                epoch_tokens += tokens
                epoch_correct += correct_count
                epoch_non_skip += non_skip_count
                epoch_non_skip_correct += non_skip_correct
                epoch_len_abs += len_abs
                epoch_len_exact += len_exact
                epoch_loss += float(loss.item()) * batch_size
                epoch_phys += float(physics_loss.item()) * batch_size
                epoch_macro_ce += float(macro_ce.item()) * batch_size
                epoch_len += float(len_loss.item()) * batch_size
                epoch_c_reg += float(c_reg.item()) * batch_size
                epoch_sym += float(sym_loss.item()) * batch_size
                epoch_alpha += float(alpha) * batch_size
                epoch_tau += float(tau) * batch_size
            loss.backward()

            if step % int(args.grad_accum) == 0:
                if float(args.clip_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                opt.step()
                opt.zero_grad(set_to_none=True)

            if step % int(args.log_steps) == 0:
                cache_note = ""
                reg_note = ""
                if float(args.c_reg_weight) > 0.0:
                    reg_note += f" c_reg={c_reg.item():.4f}"
                if float(args.sym_weight) > 0.0:
                    reg_note += f" sym={sym_loss.item():.4f}"
                if circuit_cache.max_size > 0:
                    total_cache = circuit_cache.hits + circuit_cache.misses
                    if total_cache > 0:
                        hit_rate = float(circuit_cache.hits) / float(total_cache)
                        cache_note = f" cache_hit={hit_rate:.2f}"
                print(
                    f"[epoch {epoch+1}] step={step} loss={loss.item():.4f} "
                    f"phys={physics_loss.item():.4f} phys_w={phys_weight:.1e} "
                    f"macro_ce={macro_ce.item():.4f} len={len_loss.item():.4f} alpha={alpha:.3f} tau={tau:.3f}"
                    + (f" skipped={skipped_nonfinite}" if skipped_nonfinite else "")
                    + metric_note
                    + reg_note
                    + cache_note
                )

            if int(args.save_steps) > 0 and step % int(args.save_steps) == 0:
                ckpt = args.output / f"step_{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt / "pytorch_model.bin")

        # epoch checkpoint
        if args.log_epoch_metrics and epoch_samples > 0:
            macro_acc = epoch_correct / epoch_tokens if epoch_tokens else 0.0
            macro_non_skip_acc = epoch_non_skip_correct / epoch_non_skip if epoch_non_skip else 0.0
            len_mae = epoch_len_abs / float(epoch_samples)
            len_exact = epoch_len_exact / float(epoch_samples)
            avg_loss = epoch_loss / float(epoch_samples)
            avg_phys = epoch_phys / float(epoch_samples)
            avg_macro_ce = epoch_macro_ce / float(epoch_samples)
            avg_len = epoch_len / float(epoch_samples)
            avg_c_reg = epoch_c_reg / float(epoch_samples)
            avg_sym = epoch_sym / float(epoch_samples)
            avg_alpha = epoch_alpha / float(epoch_samples)
            avg_tau = epoch_tau / float(epoch_samples)
            reg_note = ""
            if float(args.c_reg_weight) > 0.0:
                reg_note += f" c_reg={avg_c_reg:.4f}"
            if float(args.sym_weight) > 0.0:
                reg_note += f" sym={avg_sym:.4f}"
            print(
                f"[epoch {epoch+1}] avg loss={avg_loss:.4f} phys={avg_phys:.4f} "
                f"macro_ce={avg_macro_ce:.4f} len={avg_len:.4f} alpha={avg_alpha:.3f} tau={avg_tau:.3f} "
                f"mac_acc={macro_acc:.3f} mac_ns={macro_non_skip_acc:.3f} "
                f"len_mae={len_mae:.3f} len_exact={len_exact:.3f}" + reg_note
            )
        ckpt = args.output / f"epoch_{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / "pytorch_model.bin")


if __name__ == "__main__":
    main()
