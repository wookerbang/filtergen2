"""Random-topology + refinement baseline for bilevel tasks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import MACRO_LIBRARY, SERIES_MACROS, dsl_tokens_to_macro_sequence
from src.physics.differentiable_rf import DynamicCircuitAssembler, barrier_loss, calc_yield


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


def _sample_random_macros(
    gt_macros: List[str],
    *,
    macro_vocab: List[str],
    series_macros: List[str],
    shunt_macros: List[str],
    mode: str,
) -> List[str]:
    if mode == "uniform":
        macros = [random.choice(macro_vocab) for _ in gt_macros]
        if not any(m in SERIES_MACROS for m in macros) and series_macros:
            macros[random.randrange(len(macros))] = random.choice(series_macros)
        return macros
    # role-matched random
    macros = []
    for m in gt_macros:
        pool = series_macros if m in SERIES_MACROS else shunt_macros
        if not pool:
            pool = macro_vocab
        macros.append(random.choice(pool))
    if not any(m in SERIES_MACROS for m in macros) and series_macros:
        macros[random.randrange(len(macros))] = random.choice(series_macros)
    return macros


def _refine_slots(
    slot_raw: torch.Tensor,
    slot_mask: torch.Tensor,
    slot_idx: torch.Tensor,
    circuit: object,
    freq: torch.Tensor,
    target: torch.Tensor,
    mask_min: torch.Tensor,
    mask_max: torch.Tensor,
    *,
    steps: int,
    lr: float,
    max_step: float,
    raw_min: float,
    raw_max: float,
    barrier_weight: float,
    yield_threshold: float,
) -> Tuple[torch.Tensor, List[float], int]:
    raw_flat = slot_raw.reshape(-1).detach()
    mask_flat = slot_mask.reshape(-1)
    steps_to_success = -1
    loss_hist: List[float] = []
    for step in range(int(steps)):
        raw_flat = raw_flat.clamp(min=raw_min, max=raw_max).detach().requires_grad_(True)
        values_flat = torch.exp(raw_flat) * mask_flat + 1e-30
        values_vec = values_flat.index_select(0, slot_idx)
        pred = circuit(freq, values=values_vec, output="s21_db")
        loss = F.mse_loss(pred, target)
        if barrier_weight > 0.0:
            loss = loss + float(barrier_weight) * barrier_loss(pred, mask_min, mask_max)
        loss_hist.append(float(loss.detach().cpu().item()))
        _, y = calc_yield(pred, mask_min, mask_max)
        if steps_to_success < 0 and float(y.item()) >= float(yield_threshold):
            steps_to_success = step + 1
        grad = torch.autograd.grad(loss, raw_flat, create_graph=False)[0]
        step_delta = lr * grad
        if max_step > 0:
            step_delta = step_delta.clamp(-max_step, max_step)
        raw_flat = (raw_flat - step_delta * mask_flat).detach()
    return raw_flat.view_as(slot_raw).detach(), loss_hist, steps_to_success


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-topology + refine baseline.")
    p.add_argument("--data", type=Path, required=True, help="Path to eval jsonl.")
    p.add_argument("--num", type=int, default=200, help="Number of samples to evaluate (0=all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-wave", choices=["auto", "ideal", "real"], default="auto")
    p.add_argument("--random-mode", choices=["role", "uniform"], default="role")
    p.add_argument("--length-mode", choices=["gt", "kmax"], default="gt", help="Macro length: gt or kmax+skip.")
    p.add_argument("--k-max", type=int, default=12, help="k_max when length-mode=kmax.")
    p.add_argument("--skip-prob", type=float, default=0.3, help="Skip probability when length-mode=kmax.")
    p.add_argument("--steps", type=int, default=15, help="Refinement steps.")
    p.add_argument("--inner-lr", type=float, default=1e-2)
    p.add_argument("--inner-max-step", type=float, default=0.5)
    p.add_argument("--raw-min", type=float, default=-32.0)
    p.add_argument("--raw-max", type=float, default=-12.0)
    p.add_argument("--barrier-weight", type=float, default=1.0)
    p.add_argument("--yield-threshold", type=float, default=1.0)
    p.add_argument("--output", type=Path, help="Optional JSONL output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples: List[dict] = []
    macro_seqs: List[List[str]] = []
    with open(args.data, "r") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            s = json.loads(line)
            macros = s.get("macro_ir_macros") or []
            if not macros:
                tokens = s.get("dsl_tokens") or []
                if not tokens:
                    raise ValueError(f"Missing macro_ir_macros/dsl_tokens at line {line_no} in {args.data}.")
                macros = dsl_tokens_to_macro_sequence(tokens, strict=True)
            if not macros:
                continue
            samples.append(s)
            macro_seqs.append(macros)

    if not samples:
        raise ValueError(f"No valid samples in {args.data}")

    if args.num and int(args.num) > 0:
        idxs = list(range(len(samples)))
        random.shuffle(idxs)
        idxs = idxs[: int(args.num)]
    else:
        idxs = list(range(len(samples)))

    macro_vocab = sorted({m for seq in macro_seqs for m in seq})
    macro_to_id = {m: i for i, m in enumerate(macro_vocab)}
    id_to_macro = list(macro_vocab)
    skip_id = len(macro_vocab)
    slot_count = max(len(MACRO_LIBRARY[m].slot_types) for m in macro_vocab)
    macro_slot_mask = torch.zeros((len(macro_vocab) + 1, slot_count), dtype=torch.float32, device=device)
    for mid, macro in enumerate(macro_vocab):
        slen = len(MACRO_LIBRARY[macro].slot_types)
        if slen > 0:
            macro_slot_mask[mid, :slen] = 1.0

    series_macros = [m for m in macro_vocab if m in SERIES_MACROS]
    shunt_macros = [m for m in macro_vocab if m not in SERIES_MACROS]

    assembler = DynamicCircuitAssembler(z0=50.0)
    device = torch.device(args.device)

    results = []
    steps_to_success_list: List[int] = []
    success_count = 0
    success_pre = 0
    total_sims = 0

    for idx in idxs:
        s = samples[idx]
        gt_macros = macro_seqs[idx]
        if str(args.length_mode) == "kmax":
            k_max = max(1, int(args.k_max))
            macros = _sample_random_macros(
                [random.choice(series_macros or macro_vocab) for _ in range(k_max)],
                macro_vocab=macro_vocab,
                series_macros=series_macros,
                shunt_macros=shunt_macros,
                mode=str(args.random_mode),
            )
            macro_ids = torch.tensor([macro_to_id[m] for m in macros], dtype=torch.long, device=device)
            skip_prob = float(args.skip_prob)
            if skip_prob > 0.0:
                mask = torch.rand(k_max, device=device) < skip_prob
                macro_ids = macro_ids.clone()
                macro_ids[mask] = skip_id
        else:
            macros = _sample_random_macros(
                gt_macros,
                macro_vocab=macro_vocab,
                series_macros=series_macros,
                shunt_macros=shunt_macros,
                mode=str(args.random_mode),
            )
            macro_ids = torch.tensor([macro_to_id[m] for m in macros], dtype=torch.long, device=device)
        slot_mask = macro_slot_mask[macro_ids].to(device=device)

        circuit, slot_idx = _build_circuit_and_indices(
            macro_ids,
            id_to_macro=id_to_macro,
            skip_id=skip_id,
            slot_count=slot_count,
            assembler=assembler,
            device=device,
            dtype=torch.float32,
        )

        freq = torch.tensor(s["freq_hz"], dtype=torch.float32, device=device)
        ideal = torch.tensor(s["ideal_s21_db"], dtype=torch.float32, device=device)
        real = torch.tensor(s["real_s21_db"], dtype=torch.float32, device=device)
        if args.target_wave == "ideal":
            target = ideal
        elif args.target_wave == "real":
            target = real
        else:
            target = real if s.get("real_s21_db") is not None else ideal

        mask_min = torch.tensor(s.get("mask_min_db") or [float("nan")] * len(freq), dtype=torch.float32, device=device)
        mask_max = torch.tensor(s.get("mask_max_db") or [float("nan")] * len(freq), dtype=torch.float32, device=device)

        slot_raw = torch.empty((len(macros), slot_count), device=device).uniform_(
            float(args.raw_min), float(args.raw_max)
        )
        values_flat = torch.exp(slot_raw.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
        values_vec = values_flat.index_select(0, slot_idx)
        pred_pre = circuit(freq, values=values_vec, output="s21_db")
        pre_mse = float(F.mse_loss(pred_pre, target).detach().cpu().item())
        _, pre_yield = calc_yield(pred_pre, mask_min, mask_max)
        if float(pre_yield.item()) >= float(args.yield_threshold):
            success_pre += 1

        refined_raw, loss_hist, steps_to_success = _refine_slots(
            slot_raw,
            slot_mask,
            slot_idx,
            circuit,
            freq,
            target,
            mask_min,
            mask_max,
            steps=int(args.steps),
            lr=float(args.inner_lr),
            max_step=float(args.inner_max_step),
            raw_min=float(args.raw_min),
            raw_max=float(args.raw_max),
            barrier_weight=float(args.barrier_weight),
            yield_threshold=float(args.yield_threshold),
        )
        total_sims += int(args.steps)
        if steps_to_success > 0:
            success_count += 1
            steps_to_success_list.append(steps_to_success)
        values_flat = torch.exp(refined_raw.reshape(-1)) * slot_mask.reshape(-1) + 1e-30
        values_vec = values_flat.index_select(0, slot_idx)
        pred_post = circuit(freq, values=values_vec, output="s21_db")
        post_mse = float(F.mse_loss(pred_post, target).detach().cpu().item())
        _, post_yield = calc_yield(pred_post, mask_min, mask_max)

        results.append(
            {
                "idx": idx,
                "sample_id": s.get("sample_id"),
                "filter_type": s.get("filter_type"),
                "fc_hz": float(s.get("fc_hz", 0.0) or 0.0),
                "pre_mse": pre_mse,
                "post_mse": post_mse,
                "pre_yield": float(pre_yield.item()),
                "post_yield": float(post_yield.item()),
                "steps_to_success": steps_to_success,
            }
        )

    avg_pre = float(np.mean([r["pre_mse"] for r in results])) if results else 0.0
    avg_post = float(np.mean([r["post_mse"] for r in results])) if results else 0.0
    avg_pre_y = float(np.mean([r["pre_yield"] for r in results])) if results else 0.0
    avg_post_y = float(np.mean([r["post_yield"] for r in results])) if results else 0.0
    success_rate = success_count / max(1, len(results))
    success_per_sim = success_count / max(1, total_sims)
    step_mean = float(np.mean(steps_to_success_list)) if steps_to_success_list else None

    print(f"samples={len(results)}")
    print(f"pre_mse_avg={avg_pre:.6f} post_mse_avg={avg_post:.6f}")
    print(f"pre_yield_avg={avg_pre_y:.4f} post_yield_avg={avg_post_y:.4f}")
    print(f"success_rate={success_rate:.4f} success_per_sim={success_per_sim:.6f}")
    if step_mean is not None:
        print(f"steps_to_success_mean={step_mean:.2f}")
    else:
        print("steps_to_success_mean=None")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
