"""
Scenario templates + shared frequency grid / mask builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .gen_prototype import insert_shunt_series_lc, sample_base_spec

ScenarioName = str

SCENARIO_NAMES: Sequence[ScenarioName] = (
    "general",
    "anti_jamming",
    "coexistence",
    "wideband_rejection",
    "random_basic",
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
)

SCENARIO_ID: Dict[ScenarioName, int] = {name: i for i, name in enumerate(SCENARIO_NAMES)}

DEFAULT_SCENARIO_WEIGHTS: Dict[ScenarioName, float] = {
    "general": 0.1167,
    "anti_jamming": 0.0667,
    "coexistence": 0.0667,
    "wideband_rejection": 0.05,
    "random_basic": 0.7,
}


@dataclass(frozen=True)
class ScenarioSpec:
    name: ScenarioName
    spec: Dict[str, object]


def _normalize_weights(weights: Mapping[ScenarioName, float]) -> Dict[ScenarioName, float]:
    total = float(sum(float(v) for v in weights.values()))
    if total <= 0:
        return {k: 1.0 / max(1, len(weights)) for k in weights}
    return {k: float(v) / total for k, v in weights.items()}


def _resolve_filter_type_override(
    *,
    override: Optional[str],
    allowed: Sequence[str] | str,
    scenario: str,
) -> Sequence[str] | str:
    if override is None:
        return allowed
    if isinstance(allowed, str):
        allowed_set = {allowed}
    else:
        allowed_set = set(allowed)
    if override not in allowed_set:
        raise ValueError(f"filter_type={override} is incompatible with scenario={scenario}.")
    return override


def _resolve_prototype_types_override(override: Optional[Sequence[str]]) -> Sequence[str] | None:
    if override is None:
        return None
    return list(override)


def _resolve_topology_types_override(override: Optional[str]) -> Sequence[str] | None:
    if override is None:
        return None
    return [str(override)]


def sample_general_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed=["lowpass", "highpass", "bandpass"],
            scenario="general",
        ),
        order_range=(3, 7),
        ripple_db_range=(0.05, 0.5),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "general",
            "scenario_id": SCENARIO_ID["general"],
            "return_loss_min_db": 15.0,
        }
    )
    return spec


def sample_anti_jamming_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="bandpass",
            scenario="anti_jamming",
        ),
        order_range=(3, 7),
        bw_frac_range=(0.05, 0.2),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    fc = float(spec["fc_hz"])
    bw = float(spec.get("bw_frac") or 0.2)
    side = -1.0 if rng.random() < 0.5 else 1.0
    offset = float(rng.uniform(0.6, 1.2)) * bw
    notch_freq = fc * (1.0 + side * offset)
    spec.update(
        {
            "scenario": "anti_jamming",
            "scenario_id": SCENARIO_ID["anti_jamming"],
            "notch_freq_hz": notch_freq,
            "notch_depth_db": 40.0,
            "notch_bw_frac": float(rng.uniform(0.01, 0.03)),
        }
    )
    return spec


def sample_coexistence_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="bandpass",
            scenario="coexistence",
        ),
        order_range=(6, 9),
        bw_frac_range=(0.05, 0.2),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "coexistence",
            "scenario_id": SCENARIO_ID["coexistence"],
            "asymmetry_factor": float(rng.uniform(0.1, 0.3)),
        }
    )
    return spec


def sample_wideband_rejection_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="bandstop",
            scenario="wideband_rejection",
        ),
        order_range=(3, 7),
        bw_frac_range=(0.1, 0.4),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "wideband_rejection",
            "scenario_id": SCENARIO_ID["wideband_rejection"],
        }
    )
    return spec


def sample_random_basic_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed=["lowpass", "highpass", "bandpass", "bandstop"],
            scenario="random_basic",
        ),
        order_range=(2, 8),
        bw_frac_range=(0.05, 0.4),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "random_basic",
            "scenario_id": SCENARIO_ID["random_basic"],
        }
    )
    return spec


def sample_lowpass_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="lowpass",
            scenario="lowpass",
        ),
        order_range=(2, 8),
        bw_frac_range=(0.05, 0.4),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "lowpass",
            "scenario_id": SCENARIO_ID["lowpass"],
        }
    )
    return spec


def sample_highpass_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="highpass",
            scenario="highpass",
        ),
        order_range=(2, 8),
        bw_frac_range=(0.05, 0.4),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "highpass",
            "scenario_id": SCENARIO_ID["highpass"],
        }
    )
    return spec


def sample_bandpass_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="bandpass",
            scenario="bandpass",
        ),
        order_range=(2, 8),
        bw_frac_range=(0.05, 0.4),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "bandpass",
            "scenario_id": SCENARIO_ID["bandpass"],
        }
    )
    return spec


def sample_bandstop_spec(
    rng: np.random.Generator,
    *,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    spec = sample_base_spec(
        rng=rng,
        filter_type=_resolve_filter_type_override(
            override=filter_type_override,
            allowed="bandstop",
            scenario="bandstop",
        ),
        order_range=(2, 8),
        bw_frac_range=(0.05, 0.4),
        prototype_types=_resolve_prototype_types_override(prototype_types_override) or ("cheby1", "butter"),
        topology_types=_resolve_topology_types_override(topology_type_override) or ("pi", "t"),
    )
    spec.update(
        {
            "scenario": "bandstop",
            "scenario_id": SCENARIO_ID["bandstop"],
        }
    )
    return spec


SCENARIO_SAMPLERS = {
    "general": sample_general_spec,
    "anti_jamming": sample_anti_jamming_spec,
    "coexistence": sample_coexistence_spec,
    "wideband_rejection": sample_wideband_rejection_spec,
    "random_basic": sample_random_basic_spec,
    "lowpass": sample_lowpass_spec,
    "highpass": sample_highpass_spec,
    "bandpass": sample_bandpass_spec,
    "bandstop": sample_bandstop_spec,
}


def sample_scenario_spec(
    *,
    rng: np.random.Generator | None = None,
    scenario: ScenarioName | None = None,
    scenario_weights: Mapping[ScenarioName, float] | None = None,
    filter_type_override: Optional[str] = None,
    prototype_types_override: Optional[Sequence[str]] = None,
    topology_type_override: Optional[str] = None,
) -> Dict[str, object]:
    rng = rng or np.random.default_rng()
    if scenario is None or str(scenario).lower() == "random":
        weights = _normalize_weights(scenario_weights or DEFAULT_SCENARIO_WEIGHTS)
        names = list(weights.keys())
        probs = [weights[n] for n in names]
        scenario = rng.choice(names, p=probs).item()
    scenario = str(scenario)
    if scenario not in SCENARIO_SAMPLERS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return SCENARIO_SAMPLERS[scenario](
        rng,
        filter_type_override=filter_type_override,
        prototype_types_override=prototype_types_override,
        topology_type_override=topology_type_override,
    )


def build_freq_grid(spec: Mapping[str, object], *, num_freqs: int = 256) -> np.ndarray:
    fc = float(spec.get("fc_hz", 1.0))
    f_min = fc / 10.0
    f_max = fc * 10.0
    ftype = str(spec.get("filter_type", "lowpass"))
    if ftype in ("bandpass", "bandstop"):
        bw = float(spec.get("bw_frac") or spec.get("stopband_bw_frac") or 0.2)
        span = max(0.2, 2.5 * bw)
        f_min = fc * (1.0 - span)
        f_max = fc * (1.0 + span)
        if f_min <= 0:
            f_min = fc / 10.0
    notch_freq = spec.get("notch_freq_hz")
    if notch_freq is not None:
        f0 = float(notch_freq)
        f_min = min(f_min, f0 * 0.8)
        f_max = max(f_max, f0 * 1.2)
    return np.logspace(np.log10(f_min), np.log10(f_max), int(num_freqs))


def build_spec_masks(
    spec: Mapping[str, object],
    freq_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    f = np.asarray(freq_hz, dtype=float)
    mask_min = np.full_like(f, np.nan, dtype=float)
    mask_max = np.full_like(f, np.nan, dtype=float)
    fc = float(spec.get("fc_hz", 0.0))
    ripple_db = float(spec.get("ripple_db", 0.5))
    passband_min_db = -abs(ripple_db)
    stopband_max_db = float(spec.get("stopband_max_db", -40.0))
    ftype = str(spec.get("filter_type", "lowpass"))

    if ftype == "lowpass":
        pass_mask = f <= fc
        stop_mask = f >= 2.0 * fc
    elif ftype == "highpass":
        pass_mask = f >= fc
        stop_mask = f <= 0.5 * fc
    elif ftype == "bandpass":
        bw = float(spec.get("bw_frac") or 0.2)
        pass_lo = fc * (1.0 - 0.5 * bw)
        pass_hi = fc * (1.0 + 0.5 * bw)
        asym = spec.get("asymmetry_factor")
        if asym is not None:
            left_guard = float(asym) * bw
            right_guard = max(bw, 2.0 * bw)
        else:
            left_guard = bw
            right_guard = bw
        stop_lo = fc * (1.0 - (0.5 * bw + left_guard))
        stop_hi = fc * (1.0 + (0.5 * bw + right_guard))
        pass_mask = (f >= pass_lo) & (f <= pass_hi)
        stop_mask = (f <= stop_lo) | (f >= stop_hi)
    elif ftype == "bandstop":
        bw = float(spec.get("bw_frac") or spec.get("stopband_bw_frac") or 0.2)
        stop_lo = fc * (1.0 - 0.5 * bw)
        stop_hi = fc * (1.0 + 0.5 * bw)
        pass_lo = fc * (1.0 - 1.5 * bw)
        pass_hi = fc * (1.0 + 1.5 * bw)
        pass_mask = (f <= pass_lo) | (f >= pass_hi)
        stop_mask = (f >= stop_lo) & (f <= stop_hi)
    else:
        pass_mask = f <= fc
        stop_mask = f >= 2.0 * fc

    mask_min[pass_mask] = passband_min_db
    mask_max[stop_mask] = stopband_max_db

    # Extra stopbands (e.g., notch).
    extra_stopbands: List[Mapping[str, float]] = []
    if spec.get("notch_freq_hz") is not None:
        f0 = float(spec["notch_freq_hz"])
        bw_frac = float(spec.get("notch_bw_frac") or 0.02)
        depth_db = float(spec.get("notch_depth_db") or 40.0)
        extra_stopbands.append(
            {
                "lo_hz": f0 * (1.0 - 0.5 * bw_frac),
                "hi_hz": f0 * (1.0 + 0.5 * bw_frac),
                "max_db": -abs(depth_db),
            }
        )

    for sb in spec.get("extra_stopbands") or []:
        extra_stopbands.append(sb)

    for sb in extra_stopbands:
        lo = float(sb.get("lo_hz", 0.0))
        hi = float(sb.get("hi_hz", 0.0))
        max_db = float(sb.get("max_db", stopband_max_db))
        sb_mask = (f >= lo) & (f <= hi)
        if not np.any(sb_mask):
            continue
        current = mask_max[sb_mask]
        mask_max[sb_mask] = np.where(np.isnan(current), max_db, np.minimum(current, max_db))
        mask_min[sb_mask] = np.nan

    return mask_min, mask_max, passband_min_db, stopband_max_db


def apply_scenario_postprocess(
    components: Iterable[object],
    spec: MutableMapping[str, object],
    *,
    rng: np.random.Generator | None = None,
) -> List[object]:
    """
    Apply scenario-specific topology edits (e.g., notch branch insertion).
    """
    rng = rng or np.random.default_rng()
    scenario = str(spec.get("scenario") or "")
    comps = list(components)
    if scenario == "anti_jamming":
        notch_freq = float(spec.get("notch_freq_hz") or spec.get("fc_hz"))
        comps, L_val, C_val, anchor = insert_shunt_series_lc(
            comps,
            anchor=None,
            notch_freq_hz=notch_freq,
            rng=rng,
        )
        spec["notch_L"] = L_val
        spec["notch_C"] = C_val
        spec["notch_anchor"] = anchor
        base_order = int(spec.get("order") or 0)
        spec["order_effective"] = base_order + 1
    return comps
