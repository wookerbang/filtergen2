from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Literal, Sequence, Tuple

import numpy as np

from src.data.schema import ComponentSpec
from src.data.scenarios import build_spec_masks as _build_spec_masks
from src.physics import FastTrackEngine

YieldCIMethod = Literal["wilson", "agresti_coull"]
TolDist = Literal["uniform", "normal"]
TolUnknownPolicy = Literal["zero", "error"]


@dataclass(frozen=True)
class YieldSpec:
    mask_min_db: np.ndarray
    mask_max_db: np.ndarray
    passband_mask: np.ndarray
    stopband_mask: np.ndarray
    passband_ripple_max_db: float | None = None
    return_loss_min_db: float | None = None


@dataclass(frozen=True)
class YieldEstimate:
    yield_hat: float
    ci_low: float
    ci_high: float
    num_pass: int
    num_total: int
    fail_counts: Dict[str, int]
    ci_method: YieldCIMethod = "wilson"


def build_spec_masks(spec: dict, freq_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    return _build_spec_masks(spec, freq_hz)


def prepare_yield_spec(
    *,
    freq_hz: Sequence[float] | np.ndarray,
    mask_min_db: Sequence[float] | np.ndarray | None = None,
    mask_max_db: Sequence[float] | np.ndarray | None = None,
    spec: dict | None = None,
    passband_ripple_max_db: float | None = None,
    return_loss_min_db: float | None = None,
    require_masks: bool = False,
) -> YieldSpec:
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    if mask_min_db is None or mask_max_db is None:
        if spec is None:
            raise ValueError("Either mask_min_db/mask_max_db or spec must be provided for yield evaluation.")
        mask_min_db, mask_max_db, _, _ = build_spec_masks(spec, f)
        if passband_ripple_max_db is None and spec.get("ripple_db") is not None:
            passband_ripple_max_db = abs(float(spec.get("ripple_db")))
    mask_min = np.asarray(mask_min_db, dtype=float).reshape(-1)
    mask_max = np.asarray(mask_max_db, dtype=float).reshape(-1)
    if mask_min.shape != f.shape or mask_max.shape != f.shape:
        raise ValueError(
            f"mask_min_db/mask_max_db must match freq_hz shape {tuple(f.shape)}, "
            f"got {tuple(mask_min.shape)} and {tuple(mask_max.shape)}"
        )
    passband_mask = np.isfinite(mask_min)
    stopband_mask = np.isfinite(mask_max)
    if require_masks:
        if not bool(passband_mask.any()):
            raise ValueError("Passband mask is empty; check freq grid and spec alignment.")
        if not bool(stopband_mask.any()):
            raise ValueError("Stopband mask is empty; check freq grid and spec alignment.")
    rl_db = abs(float(return_loss_min_db)) if return_loss_min_db is not None else None
    return YieldSpec(
        mask_min_db=mask_min,
        mask_max_db=mask_max,
        passband_mask=passband_mask,
        stopband_mask=stopband_mask,
        passband_ripple_max_db=passband_ripple_max_db,
        return_loss_min_db=rl_db,
    )


def _sample_scale(
    rng: np.random.Generator,
    tol_frac: float,
    *,
    dist: TolDist,
    sigma_frac: float | None,
    trunc_sigma: float,
) -> float:
    t = float(tol_frac)
    if t <= 0:
        return 1.0
    if dist == "uniform":
        return float(rng.uniform(1.0 - t, 1.0 + t))
    if dist == "normal":
        sigma = float(sigma_frac) if sigma_frac is not None else float(t / max(float(trunc_sigma), 1e-6))
        if sigma <= 0:
            return 1.0
        eps = 0.0
        for _ in range(64):
            eps = float(rng.normal(0.0, sigma))
            if abs(eps) <= t:
                return 1.0 + eps
        return 1.0 + float(np.clip(eps, -t, t))
    raise ValueError(f"Unknown tolerance distribution: {dist}")


def _sample_global_shift(
    rng: np.random.Generator,
    sigma_frac: float | None,
    trunc_sigma: float,
) -> float:
    if sigma_frac is None:
        return 0.0
    sigma = float(sigma_frac)
    if sigma <= 0:
        return 0.0
    limit = abs(float(trunc_sigma)) * sigma
    if limit <= 0:
        return float(rng.normal(0.0, sigma))
    eps = 0.0
    for _ in range(64):
        eps = float(rng.normal(0.0, sigma))
        if abs(eps) <= limit:
            return eps
    return float(np.clip(eps, -limit, limit))


def _contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        ii = int(i)
        if ii == prev + 1:
            prev = ii
            continue
        segments.append((start, prev + 1))
        start = ii
        prev = ii
    segments.append((start, prev + 1))
    return segments


def apply_tolerance(
    comps: Sequence[ComponentSpec],
    *,
    rng: np.random.Generator,
    tol_L: float,
    tol_C: float,
    tol_map: Dict[str, float] | None = None,
    unknown_policy: TolUnknownPolicy = "zero",
    global_sigma_L: float | None = None,
    global_sigma_C: float | None = None,
    global_sigma_map: Dict[str, float] | None = None,
    global_trunc_sigma: float = 3.0,
    dist: TolDist = "uniform",
    sigma_frac: float | None = None,
    trunc_sigma: float = 3.0,
) -> list[ComponentSpec]:
    if tol_map is None:
        tol_map = {"L": float(tol_L), "C": float(tol_C)}
    global_sigma_map = dict(global_sigma_map or {})
    if global_sigma_L is not None:
        global_sigma_map["L"] = float(global_sigma_L)
    if global_sigma_C is not None:
        global_sigma_map["C"] = float(global_sigma_C)
    global_scale_map: Dict[str, float] = {}
    for ctype, sigma in global_sigma_map.items():
        sigma_val = float(sigma)
        if sigma_val > 0:
            shift = _sample_global_shift(rng, sigma_val, global_trunc_sigma)
            global_scale_map[str(ctype)] = 1.0 + shift
    out: list[ComponentSpec] = []
    for c in comps:
        tol = tol_map.get(str(c.ctype))
        if tol is None:
            if unknown_policy == "error":
                raise ValueError(f"Unknown component type for tolerance: {c.ctype}")
            tol = 0.0
        scale = _sample_scale(rng, tol, dist=dist, sigma_frac=sigma_frac, trunc_sigma=trunc_sigma)
        scale = float(global_scale_map.get(str(c.ctype), 1.0)) * float(scale)
        out.append(
            ComponentSpec(
                ctype=c.ctype,
                role=c.role,
                value_si=float(c.value_si) * float(scale),
                std_label=c.std_label,
                node1=c.node1,
                node2=c.node2,
            )
        )
    return out


def _check_specs(
    s21_db: np.ndarray,
    s11_db: np.ndarray | None,
    spec: YieldSpec,
) -> Tuple[bool, Tuple[str, ...]]:
    s21 = np.asarray(s21_db, dtype=float).reshape(-1)
    if s21.shape != spec.mask_min_db.shape:
        raise ValueError(f"s21_db shape {tuple(s21.shape)} != mask shape {tuple(spec.mask_min_db.shape)}")

    reasons: list[str] = []
    if bool(spec.passband_mask.any()):
        if np.any(s21[spec.passband_mask] < spec.mask_min_db[spec.passband_mask]):
            reasons.append("passband_min")
    else:
        reasons.append("passband_mask_empty")

    if bool(spec.stopband_mask.any()):
        if np.any(s21[spec.stopband_mask] > spec.mask_max_db[spec.stopband_mask]):
            reasons.append("stopband_max")
    else:
        reasons.append("stopband_mask_empty")

    if spec.passband_ripple_max_db is not None and bool(spec.passband_mask.any()):
        max_ripple = None
        for start, end in _contiguous_true_segments(spec.passband_mask):
            seg = s21[start:end]
            if seg.size == 0:
                continue
            ripple = float(np.max(seg) - np.min(seg))
            max_ripple = ripple if max_ripple is None else max(max_ripple, ripple)
        if max_ripple is not None and max_ripple > float(spec.passband_ripple_max_db):
            reasons.append("passband_ripple")

    if spec.return_loss_min_db is not None:
        if s11_db is None:
            reasons.append("s11_missing")
        elif bool(spec.passband_mask.any()):
            s11 = np.asarray(s11_db, dtype=float).reshape(-1)
            if s11.shape != s21.shape:
                raise ValueError(f"s11_db shape {tuple(s11.shape)} != s21_db shape {tuple(s21.shape)}")
            limit = -abs(float(spec.return_loss_min_db))
            if np.any(s11[spec.passband_mask] > limit):
                reasons.append("return_loss")

    ok = len(reasons) == 0
    return ok, tuple(reasons)


def _binom_ci(k: int, n: int, *, alpha: float = 0.05, method: YieldCIMethod = "wilson") -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    k = max(0, min(int(k), int(n)))
    p = float(k) / float(n)
    z = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    if method == "wilson":
        denom = 1.0 + (z * z) / float(n)
        center = (p + (z * z) / (2.0 * float(n))) / denom
        half = (z * np.sqrt((p * (1.0 - p) / float(n)) + (z * z) / (4.0 * float(n) * float(n)))) / denom
        return float(max(0.0, center - half)), float(min(1.0, center + half))
    if method == "agresti_coull":
        n_tilde = float(n) + (z * z)
        p_tilde = (float(k) + (z * z) / 2.0) / n_tilde
        half = z * np.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
        return float(max(0.0, p_tilde - half)), float(min(1.0, p_tilde + half))
    raise ValueError(f"Unknown CI method: {method}")


def _simulate_yield_batch(
    *,
    components: Sequence[ComponentSpec],
    freq_hz: np.ndarray,
    spec: YieldSpec,
    n: int,
    rng: np.random.Generator,
    engine: FastTrackEngine,
    q_L: float | None,
    q_C: float | None,
    q_model: Literal["freq_dependent", "fixed_ref"],
    ref_freq_hz: float | None,
    tol_L: float,
    tol_C: float,
    tol_map: Dict[str, float] | None,
    unknown_policy: TolUnknownPolicy,
    global_sigma_L: float | None,
    global_sigma_C: float | None,
    global_sigma_map: Dict[str, float] | None,
    global_trunc_sigma: float,
    dist: TolDist,
    sigma_frac: float | None,
    trunc_sigma: float,
) -> tuple[int, Dict[str, int]]:
    pass_count = 0
    fail_counts: Dict[str, int] = {}
    for _ in range(int(n)):
        perturbed = apply_tolerance(
            components,
            rng=rng,
            tol_L=tol_L,
            tol_C=tol_C,
            tol_map=tol_map,
            unknown_policy=unknown_policy,
            global_sigma_L=global_sigma_L,
            global_sigma_C=global_sigma_C,
            global_sigma_map=global_sigma_map,
            global_trunc_sigma=global_trunc_sigma,
            dist=dist,
            sigma_frac=sigma_frac,
            trunc_sigma=trunc_sigma,
        )
        try:
            s21_db, s11_db = engine.simulate_sparams_db(
                perturbed,
                freq_hz,
                q_L=q_L,
                q_C=q_C,
                q_model=q_model,
                ref_freq_hz=ref_freq_hz,
            )
            if s21_db is None or np.any(np.isnan(s21_db)):
                raise RuntimeError("s21_db invalid")
            ok, reasons = _check_specs(s21_db, s11_db, spec)
        except Exception:
            ok = False
            reasons = ("sim_fail",)
        if ok:
            pass_count += 1
        else:
            for r in reasons:
                fail_counts[r] = fail_counts.get(r, 0) + 1
    return pass_count, fail_counts


def estimate_yield_mc(
    *,
    components: Sequence[ComponentSpec],
    freq_hz: Sequence[float] | np.ndarray,
    spec: YieldSpec,
    n: int,
    rng: np.random.Generator | None = None,
    engine: FastTrackEngine | None = None,
    z0: float = 50.0,
    device: str = "cpu",
    dtype=np.float64,
    q_L: float | None = None,
    q_C: float | None = None,
    q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
    ref_freq_hz: float | None = None,
    tol_L: float = 0.0,
    tol_C: float = 0.0,
    tol_map: Dict[str, float] | None = None,
    unknown_policy: TolUnknownPolicy = "zero",
    global_sigma_L: float | None = None,
    global_sigma_C: float | None = None,
    global_sigma_map: Dict[str, float] | None = None,
    global_trunc_sigma: float = 3.0,
    dist: TolDist = "uniform",
    sigma_frac: float | None = None,
    trunc_sigma: float = 3.0,
    ci_alpha: float = 0.05,
    ci_method: YieldCIMethod = "wilson",
) -> YieldEstimate:
    rng = rng or np.random.default_rng()
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    if engine is None:
        engine = FastTrackEngine(z0=float(z0), device=device, dtype=torch_dtype_from_np(dtype))
    if q_model == "fixed_ref" and (q_L is not None or q_C is not None) and ref_freq_hz is None:
        ref_freq_hz = float(np.sqrt(float(np.min(f)) * float(np.max(f))))

    pass_count, fail_counts = _simulate_yield_batch(
        components=components,
        freq_hz=f,
        spec=spec,
        n=int(n),
        rng=rng,
        engine=engine,
        q_L=q_L,
        q_C=q_C,
        q_model=q_model,
        ref_freq_hz=ref_freq_hz,
        tol_L=tol_L,
        tol_C=tol_C,
        tol_map=tol_map,
        unknown_policy=unknown_policy,
        global_sigma_L=global_sigma_L,
        global_sigma_C=global_sigma_C,
        global_sigma_map=global_sigma_map,
        global_trunc_sigma=global_trunc_sigma,
        dist=dist,
        sigma_frac=sigma_frac,
        trunc_sigma=trunc_sigma,
    )
    total = int(n)
    y_hat = float(pass_count) / float(max(total, 1))
    ci_low, ci_high = _binom_ci(pass_count, total, alpha=ci_alpha, method=ci_method)
    return YieldEstimate(
        yield_hat=y_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        num_pass=int(pass_count),
        num_total=int(total),
        fail_counts=fail_counts,
        ci_method=ci_method,
    )


def estimate_yield_sequential(
    *,
    components: Sequence[ComponentSpec],
    freq_hz: Sequence[float] | np.ndarray,
    spec: YieldSpec,
    n_min: int = 200,
    n_max: int = 2000,
    batch: int = 200,
    target_half_width: float = 0.02,
    rng: np.random.Generator | None = None,
    engine: FastTrackEngine | None = None,
    z0: float = 50.0,
    device: str = "cpu",
    dtype=np.float64,
    q_L: float | None = None,
    q_C: float | None = None,
    q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
    ref_freq_hz: float | None = None,
    tol_L: float = 0.0,
    tol_C: float = 0.0,
    tol_map: Dict[str, float] | None = None,
    unknown_policy: TolUnknownPolicy = "zero",
    global_sigma_L: float | None = None,
    global_sigma_C: float | None = None,
    global_sigma_map: Dict[str, float] | None = None,
    global_trunc_sigma: float = 3.0,
    dist: TolDist = "uniform",
    sigma_frac: float | None = None,
    trunc_sigma: float = 3.0,
    ci_alpha: float = 0.05,
    ci_method: YieldCIMethod = "wilson",
) -> YieldEstimate:
    rng = rng or np.random.default_rng()
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    if engine is None:
        engine = FastTrackEngine(z0=float(z0), device=device, dtype=torch_dtype_from_np(dtype))
    if q_model == "fixed_ref" and (q_L is not None or q_C is not None) and ref_freq_hz is None:
        ref_freq_hz = float(np.sqrt(float(np.min(f)) * float(np.max(f))))

    total = 0
    pass_count = 0
    fail_counts: Dict[str, int] = {}
    n_min = max(0, int(n_min))
    n_max = max(int(n_min), int(n_max))
    batch = max(1, int(batch))

    while total < n_min:
        n_take = min(batch, n_min - total)
        p_add, f_add = _simulate_yield_batch(
            components=components,
            freq_hz=f,
            spec=spec,
            n=n_take,
            rng=rng,
            engine=engine,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
            tol_L=tol_L,
            tol_C=tol_C,
            tol_map=tol_map,
            unknown_policy=unknown_policy,
            global_sigma_L=global_sigma_L,
            global_sigma_C=global_sigma_C,
            global_sigma_map=global_sigma_map,
            global_trunc_sigma=global_trunc_sigma,
            dist=dist,
            sigma_frac=sigma_frac,
            trunc_sigma=trunc_sigma,
        )
        total += int(n_take)
        pass_count += int(p_add)
        for k, v in f_add.items():
            fail_counts[k] = fail_counts.get(k, 0) + int(v)

    ci_low, ci_high = _binom_ci(pass_count, total, alpha=ci_alpha, method=ci_method)
    half_width = 0.5 * (ci_high - ci_low)
    while total < n_max and half_width > float(target_half_width):
        n_take = min(batch, n_max - total)
        p_add, f_add = _simulate_yield_batch(
            components=components,
            freq_hz=f,
            spec=spec,
            n=n_take,
            rng=rng,
            engine=engine,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
            tol_L=tol_L,
            tol_C=tol_C,
            tol_map=tol_map,
            unknown_policy=unknown_policy,
            global_sigma_L=global_sigma_L,
            global_sigma_C=global_sigma_C,
            global_sigma_map=global_sigma_map,
            global_trunc_sigma=global_trunc_sigma,
            dist=dist,
            sigma_frac=sigma_frac,
            trunc_sigma=trunc_sigma,
        )
        total += int(n_take)
        pass_count += int(p_add)
        for k, v in f_add.items():
            fail_counts[k] = fail_counts.get(k, 0) + int(v)
        ci_low, ci_high = _binom_ci(pass_count, total, alpha=ci_alpha, method=ci_method)
        half_width = 0.5 * (ci_high - ci_low)

    y_hat = float(pass_count) / float(max(total, 1))
    return YieldEstimate(
        yield_hat=y_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        num_pass=int(pass_count),
        num_total=int(total),
        fail_counts=fail_counts,
        ci_method=ci_method,
    )


def torch_dtype_from_np(np_dtype) -> "torch.dtype":
    import torch

    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.float64:
        return torch.float64
    return torch.float64
