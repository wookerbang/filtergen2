"""Black-box DE sizing baseline with fixed template topologies."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import MACRO_LIBRARY, SERIES_MACROS, components_to_macro_ir  # noqa: E402
from src.data.gen_prototype import (  # noqa: E402
    denormalize_bandstop_to_LC,
    denormalize_highpass_to_LC,
    denormalize_lowpass_to_LC,
    get_g_values,
    synthesize_cascade_bandpass,
)
from src.physics.differentiable_rf import (  # noqa: E402
    barrier_loss,
    calc_violation_max,
    calc_violation_quantile,
    DynamicCircuitAssembler,
)


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


def _resolve_bw_frac(sample: dict) -> float:
    bw = sample.get("bw_frac") or sample.get("stopband_bw_frac")
    try:
        return float(bw) if bw is not None else 0.2
    except Exception:
        return 0.2


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


def _choose_topology(mode: str, rng: random.Random) -> str:
    mode = mode.lower()
    if mode == "pi":
        return "pi"
    if mode in ("t", "tee"):
        return "t"
    return rng.choice(["pi", "t"])


def _template_macros_for_sample(
    sample: dict,
    *,
    rng: random.Random,
    bp_rng: np.random.Generator,
    topology_mode: str,
    bp_topology_mode: str,
    bp_cascade_order: str,
) -> List[List[str]]:
    ftype = str(sample.get("filter_type") or "lowpass")
    order = int(sample.get("order") or 4)
    fc_hz = float(sample.get("fc_hz") or 1.0)
    ripple_db = float(sample.get("ripple_db") or 0.5)
    proto = str(sample.get("prototype_type") or "butter")
    z0 = float(sample.get("z0") or 50.0)
    bw_frac = _resolve_bw_frac(sample)
    freq_range = sample.get("freq_range")
    if freq_range and len(freq_range) >= 2:
        f_min, f_max = float(freq_range[0]), float(freq_range[1])
    else:
        f_min = fc_hz * (1.0 - 0.5 * bw_frac)
        f_max = fc_hz * (1.0 + 0.5 * bw_frac)

    topo_list: List[str] = []
    if ftype == "bandpass":
        mode = bp_topology_mode.lower()
        if mode == "both":
            topo_list = ["pi", "t"]
        else:
            topo_list = [_choose_topology(mode, rng)]
    else:
        topo_list = [_choose_topology(topology_mode, rng)]

    macros_list: List[List[str]] = []
    for topo in topo_list:
        if ftype == "lowpass":
            g = get_g_values(order, ripple_db, prototype_type=proto)
            comps = denormalize_lowpass_to_LC(g, fc_hz, z0, topo)
        elif ftype == "highpass":
            g = get_g_values(order, ripple_db, prototype_type=proto)
            comps = denormalize_highpass_to_LC(g, fc_hz, z0, topo)
        elif ftype == "bandstop":
            g = get_g_values(order, ripple_db, prototype_type=proto)
            comps = denormalize_bandstop_to_LC(g, fc_hz, z0, bw_frac, topo)
        else:
            spec = {
                "filter_type": "bandpass",
                "prototype_type": proto,
                "topology_type": topo,
                "order": int(order),
                "ripple_db": ripple_db,
                "fc_hz": fc_hz,
                "bw_frac": bw_frac,
                "freq_range": [f_min, f_max],
                "bp_cascade_order": bp_cascade_order,
                "z0": z0,
            }
            comps = synthesize_cascade_bandpass(spec, int(order), z0, rng=bp_rng)
        macros = components_to_macro_ir(comps)
        if macros:
            macros_list.append(macros)
    return macros_list


def _masked_mse_components(
    pred_db: torch.Tensor,
    target_db: torch.Tensor,
    mask_min_db: torch.Tensor,
    mask_max_db: torch.Tensor,
    *,
    w_pass: float,
    w_stop: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred_db.reshape(-1)
    target = target_db.reshape(-1)
    full_mse = F.mse_loss(pred, target)
    min_mask = torch.isfinite(mask_min_db.reshape(-1))
    max_mask = torch.isfinite(mask_max_db.reshape(-1))
    constrained_mask = min_mask | max_mask
    if bool(constrained_mask.any()):
        constrained_mse = torch.mean((pred[constrained_mask] - target[constrained_mask]) ** 2)
    else:
        constrained_mse = full_mse
    pass_mask = min_mask
    stop_mask = max_mask & ~pass_mask
    if bool(pass_mask.any()) or bool(stop_mask.any()):
        err = (pred - target) ** 2
        num = torch.zeros((), device=pred.device, dtype=pred.dtype)
        denom = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if bool(pass_mask.any()):
            num = num + float(w_pass) * err[pass_mask].sum()
            denom = denom + float(w_pass) * pass_mask.sum()
        if bool(stop_mask.any()):
            num = num + float(w_stop) * err[stop_mask].sum()
            denom = denom + float(w_stop) * stop_mask.sum()
        if bool(denom.item() > 0):
            weighted_mse = num / denom
        else:
            weighted_mse = full_mse
    else:
        weighted_mse = full_mse
    return full_mse, constrained_mse, weighted_mse


def _de_optimize(
    obj_fn,
    *,
    bounds: Tuple[float, float],
    pop_size: int,
    iters: int,
    f_mut: float,
    cr: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, int, np.ndarray, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    dim = int(obj_fn.dim)
    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    scores = np.full((pop_size,), np.inf, dtype=float)
    evals = 0
    for i in range(pop_size):
        scores[i] = obj_fn(pop[i])
        evals += 1
    init_best_idx = int(np.argmin(scores))
    init_best = pop[init_best_idx].copy()
    init_best_score = float(scores[init_best_idx])
    for _ in range(int(iters)):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = pop[a] + f_mut * (pop[b] - pop[c])
            mutant = np.clip(mutant, lo, hi)
            cross = rng.random(dim) < cr
            if not cross.any():
                cross[rng.integers(dim)] = True
            trial = np.where(cross, mutant, pop[i])
            score = obj_fn(trial)
            evals += 1
            if score < scores[i]:
                pop[i] = trial
                scores[i] = score
    best_idx = int(np.argmin(scores))
    return pop[best_idx], float(scores[best_idx]), evals, init_best, init_best_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black-box DE sizing baseline with fixed templates.")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--num", type=int, default=0, help="Max samples (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-wave", choices=["ideal", "real"], default="real")
    p.add_argument(
        "--loss-mode",
        choices=["full_mse", "constrained_mse", "weighted_mse", "barrier_only"],
        default="barrier_only",
    )
    p.add_argument("--w-pass", type=float, default=1.0)
    p.add_argument("--w-stop", type=float, default=5.0)
    p.add_argument("--barrier-weight", type=float, default=0.0)
    p.add_argument(
        "--mask-mode",
        choices=["dataset", "dynamic"],
        default="dataset",
        help="Mask source: dataset masks or dynamic envelope from target.",
    )
    p.add_argument("--mask-pass-drop-db", type=float, default=3.0)
    p.add_argument("--mask-stop-rel-db", type=float, default=20.0)
    p.add_argument("--mask-delta-pass", type=float, default=1.0)
    p.add_argument("--mask-delta-stop", type=float, default=3.0)
    p.add_argument("--mask-peak-quantile", type=float, default=1.0)
    p.add_argument("--mask-pass-max-db", type=float, default=0.0)
    p.add_argument("--raw-min", type=float, default=-32.0)
    p.add_argument("--raw-max", type=float, default=-12.0)
    p.add_argument("--pop-size", type=int, default=32)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--de-f", type=float, default=0.8)
    p.add_argument("--de-cr", type=float, default=0.9)
    p.add_argument("--topology-mode", choices=["pi", "t", "random"], default="pi")
    p.add_argument("--bp-topology-mode", choices=["pi", "t", "random", "both"], default="both")
    p.add_argument("--bp-cascade-order", choices=["random", "lp_hp", "hp_lp"], default="random")
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
    p.add_argument("--yield-s11-max-db", type=float, default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, help="Optional JSON output path.")
    return p.parse_args()


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
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

    with open(args.data, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.num and args.num > 0:
        samples = rng.sample(samples, min(int(args.num), len(samples)))

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    assembler = DynamicCircuitAssembler(z0=50.0)

    total = 0
    failed = 0
    pre_mse_sum = 0.0
    post_mse_sum = 0.0
    pre_constrained_sum = 0.0
    post_constrained_sum = 0.0
    pre_weighted_sum = 0.0
    post_weighted_sum = 0.0
    pre_loss_sum = 0.0
    post_loss_sum = 0.0
    yield_total = 0
    yield_pre_pass = {tau: 0 for tau in yield_taus}
    yield_post_pass = {tau: 0 for tau in yield_taus}
    yield_oracle_pass = {tau: 0 for tau in yield_taus}
    yield_pre_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    yield_post_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    yield_oracle_robust = {(tau, alpha): 0 for tau in yield_taus for alpha in yield_alphas}
    sim_calls = 0
    mask_pass_empty = 0
    mask_stop_empty = 0
    mask_empty = 0

    for sample in samples:
        total += 1
        try:
            freq = torch.tensor(sample["freq_hz"], dtype=dtype, device=device)
            target_key = "real_s21_db" if args.target_wave == "real" else "ideal_s21_db"
            target = torch.tensor(sample[target_key], dtype=dtype, device=device)
            if mask_mode == "dynamic":
                mask_min, mask_max, pass_any, stop_any = _dynamic_envelope_masks(
                    target,
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
                mask_min_raw = sample.get("mask_min_db")
                mask_max_raw = sample.get("mask_max_db")
                if mask_min_raw is None:
                    mask_min_raw = [float("nan")] * len(sample["freq_hz"])
                if mask_max_raw is None:
                    mask_max_raw = [float("nan")] * len(sample["freq_hz"])
                mask_min = torch.tensor(mask_min_raw, dtype=dtype, device=device)
                mask_max = torch.tensor(mask_max_raw, dtype=dtype, device=device)
            if not torch.isfinite(target).all():
                failed += 1
                continue

            bp_rng = np.random.default_rng(np_rng.integers(0, 1_000_000))
            macros_list = _template_macros_for_sample(
                sample,
                rng=rng,
                bp_rng=bp_rng,
                topology_mode=args.topology_mode,
                bp_topology_mode=args.bp_topology_mode,
                bp_cascade_order=args.bp_cascade_order,
            )
            if not macros_list:
                failed += 1
                continue

            best_candidate = None
            best_loss = float("inf")
            best_pre_pred = None

            for macros in macros_list:
                slot_count = max(len(MACRO_LIBRARY[m].slot_types) for m in macros)
                slot_mask = torch.zeros((len(macros), slot_count), dtype=dtype, device=device)
                for i_m, macro in enumerate(macros):
                    slen = len(MACRO_LIBRARY[macro].slot_types)
                    slot_mask[i_m, :slen] = 1.0
                circuit, slot_idx = _build_circuit_and_indices(
                    macros,
                    slot_count=slot_count,
                    assembler=assembler,
                    device=device,
                    dtype=dtype,
                )
                mask_flat = slot_mask.reshape(-1)
                active_idx = torch.nonzero(mask_flat, as_tuple=False).reshape(-1)
                if active_idx.numel() == 0:
                    continue
                raw_fill = torch.full_like(mask_flat, float(args.raw_min))

                class ObjFn:
                    def __init__(self) -> None:
                        self.dim = int(active_idx.numel())

                    def __call__(self, x: np.ndarray) -> float:
                        raw_flat = raw_fill.clone()
                        raw_flat[active_idx] = torch.tensor(x, dtype=dtype, device=device)
                        values_flat = torch.exp(raw_flat) * mask_flat + 1e-30
                        values_vec = values_flat.index_select(0, slot_idx)
                        pred = circuit(freq, values=values_vec, output="s21_db")
                        full_mse, constrained_mse, weighted_mse = _masked_mse_components(
                            pred,
                            target,
                            mask_min,
                            mask_max,
                            w_pass=args.w_pass,
                            w_stop=args.w_stop,
                        )
                        if args.loss_mode == "constrained_mse":
                            loss = constrained_mse
                        elif args.loss_mode == "weighted_mse":
                            loss = weighted_mse
                        elif args.loss_mode == "barrier_only":
                            loss = pred.new_zeros(())
                        else:
                            loss = full_mse
                        if args.barrier_weight > 0.0:
                            loss = loss + float(args.barrier_weight) * barrier_loss(pred, mask_min, mask_max)
                        if not torch.isfinite(loss):
                            return float("inf")
                        return float(loss.item())

                obj_fn = ObjFn()
                x_best, loss_best, evals, x_init, loss_init = _de_optimize(
                    obj_fn,
                    bounds=(args.raw_min, args.raw_max),
                    pop_size=int(args.pop_size),
                    iters=int(args.iters),
                    f_mut=float(args.de_f),
                    cr=float(args.de_cr),
                    rng=np_rng,
                )
                sim_calls += int(evals)

                raw_flat_best = raw_fill.clone()
                raw_flat_best[active_idx] = torch.tensor(x_best, dtype=dtype, device=device)
                values_flat_best = torch.exp(raw_flat_best) * mask_flat + 1e-30
                values_vec_best = values_flat_best.index_select(0, slot_idx)
                pred_best = circuit(freq, values=values_vec_best, output="s21_db")
                full_mse, constrained_mse, weighted_mse = _masked_mse_components(
                    pred_best,
                    target,
                    mask_min,
                    mask_max,
                    w_pass=args.w_pass,
                    w_stop=args.w_stop,
                )
                if args.loss_mode == "constrained_mse":
                    final_loss = constrained_mse
                elif args.loss_mode == "weighted_mse":
                    final_loss = weighted_mse
                elif args.loss_mode == "barrier_only":
                    final_loss = pred_best.new_zeros(())
                else:
                    final_loss = full_mse
                if args.barrier_weight > 0.0:
                    final_loss = final_loss + float(args.barrier_weight) * barrier_loss(pred_best, mask_min, mask_max)

                raw_flat_init = raw_fill.clone()
                raw_flat_init[active_idx] = torch.tensor(x_init, dtype=dtype, device=device)
                values_flat_init = torch.exp(raw_flat_init) * mask_flat + 1e-30
                values_vec_init = values_flat_init.index_select(0, slot_idx)
                pred_init = circuit(freq, values=values_vec_init, output="s21_db")
                if not torch.isfinite(pred_init).all():
                    pred_init = pred_best

                if float(final_loss.item()) < best_loss:
                    best_loss = float(final_loss.item())
                    best_candidate = (pred_best, full_mse, constrained_mse, weighted_mse)
                    best_pre_pred = pred_init

            if best_candidate is None:
                failed += 1
                continue

            pred_post, post_mse, post_constrained, post_weighted = best_candidate
            pred_pre = best_pre_pred if best_pre_pred is not None else pred_post
            pre_mse, pre_constrained, pre_weighted = _masked_mse_components(
                pred_pre,
                target,
                mask_min,
                mask_max,
                w_pass=args.w_pass,
                w_stop=args.w_stop,
            )
            if args.loss_mode == "constrained_mse":
                pre_loss = pre_constrained
                post_loss = post_constrained
            elif args.loss_mode == "weighted_mse":
                pre_loss = pre_weighted
                post_loss = post_weighted
            elif args.loss_mode == "barrier_only":
                pre_loss = pred_pre.new_zeros(())
                post_loss = pred_post.new_zeros(())
            else:
                pre_loss = pre_mse
                post_loss = post_mse
            if args.barrier_weight > 0.0:
                pre_loss = pre_loss + float(args.barrier_weight) * barrier_loss(pred_pre, mask_min, mask_max)
                post_loss = post_loss + float(args.barrier_weight) * barrier_loss(pred_post, mask_min, mask_max)

            pre_mse_sum += float(pre_mse.item())
            post_mse_sum += float(post_mse.item())
            pre_constrained_sum += float(pre_constrained.item())
            post_constrained_sum += float(post_constrained.item())
            pre_weighted_sum += float(pre_weighted.item())
            post_weighted_sum += float(post_weighted.item())
            pre_loss_sum += float(pre_loss.item())
            post_loss_sum += float(post_loss.item())

            if torch.isfinite(mask_min).any() or torch.isfinite(mask_max).any():
                yield_total += 1
                oracle_max = float(
                    calc_violation_max(
                        target,
                        mask_min,
                        mask_max,
                        pred_s11_db=None,
                        s11_max_db=yield_s11_max_db,
                    ).item()
                )
                pre_max = float(
                    calc_violation_max(
                        pred_pre,
                        mask_min,
                        mask_max,
                        pred_s11_db=None,
                        s11_max_db=yield_s11_max_db,
                    ).item()
                )
                post_max = float(
                    calc_violation_max(
                        pred_post,
                        mask_min,
                        mask_max,
                        pred_s11_db=None,
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
                            target,
                            mask_min,
                            mask_max,
                            alpha=alpha_val,
                            pred_s11_db=None,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    pre_q = float(
                        calc_violation_quantile(
                            pred_pre,
                            mask_min,
                            mask_max,
                            alpha=alpha_val,
                            pred_s11_db=None,
                            s11_max_db=yield_s11_max_db,
                        ).item()
                    )
                    post_q = float(
                        calc_violation_quantile(
                            pred_post,
                            mask_min,
                            mask_max,
                            alpha=alpha_val,
                            pred_s11_db=None,
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
        "loss_mode": args.loss_mode,
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
        "sim_calls": int(sim_calls),
        "sim_calls_per_sample": (sim_calls / max(1, total - failed)),
        "topology_mode": args.topology_mode,
        "bp_topology_mode": args.bp_topology_mode,
        "bp_cascade_order": args.bp_cascade_order,
        "target_wave": args.target_wave,
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
    }

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
