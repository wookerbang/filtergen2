"""
将各步骤串联生成样本并落盘。
"""

from __future__ import annotations

import json
import os
from typing import List, Mapping

import numpy as np
import torch

from .gen_prototype import synthesize_filter
from .quantization import quantize_components
from .schema import ComponentSpec, FilterSample
from .vact_codec import components_to_vact_tokens
from .sfci_net_codec import components_to_sfci_net_tokens, components_to_sfci_net_tokens_and_values
from .spice_runner import simulate_real_waveform
from .vact_struct import components_to_vact_struct_tokens
from .node_canonicalizer import canonicalize_nodes
from .action_codec import components_to_action_tokens
from .dsl import VAL_NONE, components_to_dsl_segments, components_to_dsl_tokens, components_to_macro_ir
from .scenarios import apply_scenario_postprocess, build_data_driven_masks, build_freq_grid, build_spec_masks, sample_scenario_spec
from src.physics import FastTrackEngine


def _serialize_components(comps: List[ComponentSpec]) -> List[dict]:
    return [c.to_dict() for c in comps]


def _serialize_sample(sample: FilterSample) -> dict:
    data = sample.to_metadata_dict()
    data.update(
        {
            "freq_hz": np.asarray(sample.freqs_hz, dtype=float).tolist() if sample.freqs_hz is not None else None,
            "ideal_s21_db": np.asarray(sample.w_ideal_S21_db, dtype=float).tolist() if sample.w_ideal_S21_db is not None else None,
            "ideal_s11_db": np.asarray(sample.w_ideal_S11_db, dtype=float).tolist() if sample.w_ideal_S11_db is not None else None,
            "real_s21_db": np.asarray(sample.w_real_S21_db, dtype=float).tolist() if sample.w_real_S21_db is not None else None,
            "real_s11_db": np.asarray(sample.w_real_S11_db, dtype=float).tolist() if sample.w_real_S11_db is not None else None,
            "mask_min_db": np.asarray(sample.mask_min_db, dtype=float).tolist() if sample.mask_min_db is not None else None,
            "mask_max_db": np.asarray(sample.mask_max_db, dtype=float).tolist() if sample.mask_max_db is not None else None,
            "ideal_components": _serialize_components(sample.ideal_components or []),
            "discrete_components": _serialize_components(sample.discrete_components or []),
        }
    )
    if sample.vact_tokens is not None:
        data["vact_tokens"] = sample.vact_tokens
    if sample.vact_struct_tokens is not None:
        data["vact_struct_tokens"] = sample.vact_struct_tokens
    if sample.dsl_tokens is not None:
        data["dsl_tokens"] = sample.dsl_tokens
    if sample.dsl_slot_values is not None:
        data["dsl_slot_values"] = sample.dsl_slot_values
    if sample.sfci_tokens is not None:
        data["sfci_tokens"] = sample.sfci_tokens
    if sample.sfci_slot_values is not None:
        data["sfci_slot_values"] = sample.sfci_slot_values
    if sample.action_tokens is not None:
        data["action_tokens"] = sample.action_tokens
    return data


def build_dataset(
    num_samples: int,
    output_dir: str,
    split: str = "train",
    use_ngspice: bool = False,
    seed: int = 42,
    scenario: str | None = None,
    scenario_weights: dict | None = None,
    emit_vact_tokens: bool = False,
    emit_vact_cells: bool = False,
    emit_vact_struct: bool = False,
    emit_actions: bool = False,
    emit_dsl: bool = True,
    emit_sfci: bool = False,
    sfci_value_mode: str = "discrete",
    dsl_include_order: bool = True,
    dsl_use_cell_indices: bool = False,
    dsl_strict: bool = False,
    max_nodes: int = 32,
    q_L: float | None = 50.0,
    q_C: float | None = 50.0,
    q_model: str = "freq_dependent",
    check_insertion_loss: bool = True,
    filter_type_override: str | None = None,
    prototype_type_override: str | None = None,
    topology_type_override: str | None = None,
    spec_fixed: Mapping[str, object] | None = None,
    spec_ranges: Mapping[str, object] | None = None,
    spec_profile: Mapping[str, Mapping[str, object]] | None = None,
    narrow_freq_grid: bool = False,
    narrow_freq_span: float = 0.5,
    quantize_series: str | None = "E24",
    mask_mode: str = "data",
    ensure_spec: bool = False,
    ensure_spec_wave: str = "real",
    ensure_max_tries: int = 0,
    ensure_spec_strategy: str = "mixed",
    ensure_struct_tries: int = 2,
    ensure_order_tries: int = 2,
    ensure_order_bias: float = 0.7,
) -> str:
    """
    串起采样 → 原型 → 离散化 → 仿真 → 序列化。
    返回写入的 jsonl 路径。
    When emit_vact_cells=True, insert <CELL> section markers into VACT tokens.
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split}.jsonl")

    rng = np.random.default_rng(seed)
    fast_engine: FastTrackEngine | None = None
    total_attempts = 0
    spec_rejects = 0
    written = 0

    def _parse_range(val: object) -> tuple[float, float] | None:
        if val is None:
            return None
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[0]), float(val[1])
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        return fval, fval

    def _get_range_from_profile(
        profile: Mapping[str, Mapping[str, object]] | None,
        scenario_name: str,
        key: str,
    ) -> tuple[float, float] | None:
        if not profile:
            return None
        entry = profile.get(scenario_name)
        if not isinstance(entry, Mapping):
            return None
        ranges = entry.get("ranges") or entry.get("spec_ranges")
        if not isinstance(ranges, Mapping):
            return None
        return _parse_range(ranges.get(key))

    def _resolve_order_range(
        scenario_name: str,
    ) -> tuple[int, int] | None:
        range_val = _get_range_from_profile(spec_profile, scenario_name, "order")
        if range_val is None:
            range_val = _parse_range(spec_ranges.get("order")) if spec_ranges else None
        if range_val is None:
            return None
        lo, hi = int(round(range_val[0])), int(round(range_val[1]))
        lo = max(1, lo)
        hi = max(lo, hi)
        return lo, hi

    def _sample_order(
        order_range: tuple[int, int],
        *,
        bias_high: float,
    ) -> int:
        lo, hi = order_range
        if lo == hi:
            return lo
        bias = float(np.clip(bias_high, 0.0, 1.0))
        if rng.random() < bias:
            mid = (lo + hi) // 2
            lo = max(lo, mid)
        return int(rng.integers(lo, hi + 1))

    def _spec_fixed_keep(spec: Mapping[str, object]) -> Dict[str, object]:
        keep_keys = (
            "filter_type",
            "order",
            "fc_hz",
            "bw_frac",
            "ripple_db",
            "stopband_max_db",
            "notch_freq_hz",
            "notch_depth_db",
            "notch_bw_frac",
            "asymmetry_factor",
            "return_loss_min_db",
            "z0",
            "bp_order_lp",
            "bp_order_hp",
            "bp_cascade_order",
        )
        fixed: Dict[str, object] = {}
        for key in keep_keys:
            if key in spec:
                fixed[key] = spec[key]
        return fixed

    def _resample_structure(spec: Dict[str, object]) -> Dict[str, object]:
        scenario_name = str(spec.get("scenario") or scenario)
        fixed = _spec_fixed_keep(spec)
        new_spec = sample_scenario_spec(
            rng=rng,
            scenario=scenario_name,
            scenario_weights=scenario_weights,
            filter_type_override=str(filter_type_override) if filter_type_override is not None else None,
            prototype_types_override=proto_override,
            topology_type_override=str(topology_type_override) if topology_type_override is not None else None,
            spec_fixed=fixed,
            spec_ranges=None,
            spec_profile=None,
        )
        new_spec.pop("order_effective", None)
        return new_spec

    def _resample_full_spec() -> Dict[str, object]:
        return sample_scenario_spec(
            rng=rng,
            scenario=scenario,
            scenario_weights=scenario_weights,
            filter_type_override=str(filter_type_override) if filter_type_override is not None else None,
            prototype_types_override=proto_override,
            topology_type_override=str(topology_type_override) if topology_type_override is not None else None,
            spec_fixed=spec_fixed,
            spec_ranges=spec_ranges,
            spec_profile=spec_profile,
        )

    def _resample_order_only(spec: Dict[str, object]) -> Dict[str, object] | None:
        scenario_name = str(spec.get("scenario") or scenario)
        order_range = _resolve_order_range(scenario_name)
        if order_range is None:
            return None
        new_order = _sample_order(order_range, bias_high=ensure_order_bias)
        spec["order"] = new_order
        spec.pop("order_effective", None)
        if str(spec.get("filter_type")) == "bandpass" and new_order < 4:
            spec["order"] = 4
        return spec

    def _meets_spec(
        s21_db: np.ndarray,
        mask_min_db: np.ndarray,
        mask_max_db: np.ndarray,
    ) -> bool:
        min_mask = np.isfinite(mask_min_db)
        if np.any(min_mask):
            if np.any(s21_db[min_mask] < mask_min_db[min_mask]):
                return False
        max_mask = np.isfinite(mask_max_db)
        if np.any(max_mask):
            if np.any(s21_db[max_mask] > mask_max_db[max_mask]):
                return False
        return True

    proto_override = None
    if prototype_type_override is not None:
        proto_override = (str(prototype_type_override),)

    with open(jsonl_path, "w") as f:
        while written < num_samples:
            i = written
            attempt_in_sample = 0
            spec = _resample_full_spec()
            while True:
                attempt_in_sample += 1
                total_attempts += 1
                if ensure_max_tries > 0 and total_attempts > ensure_max_tries:
                    raise RuntimeError(
                        f"Reached ensure_max_tries={ensure_max_tries} before collecting "
                        f"{num_samples} samples (written={written}, spec_rejects={spec_rejects})."
                    )
                z0 = spec["z0"]
                if fast_engine is None or float(z0) != float(fast_engine.z0):
                    fast_engine = FastTrackEngine(z0=float(z0), device="cpu", dtype=torch.float64)
                base_components = synthesize_filter(spec)
                base_components = apply_scenario_postprocess(base_components, spec, rng=rng)
    
                freq_hz = build_freq_grid(
                    spec,
                    num_freqs=256,
                    grid_mode="narrow" if narrow_freq_grid else "default",
                    narrow_span=float(narrow_freq_span),
                )
                spec_mask_min_db, spec_mask_max_db, passband_min_db, stopband_max_db = build_spec_masks(spec, freq_hz)
    
                # 非纯 ladder（notch/BP/BS）优先用仿真获取 ideal，以避免 ABCD 近似误差
                need_sim_for_ideal = spec.get("scenario") in ("anti_jamming",) or spec.get("filter_type") != "lowpass"
    
                # Canonicalize node names so tokenization stays in-vocab.
                base_components = canonicalize_nodes(base_components, max_nodes=max_nodes)
    
                # Output label: nominal standard parts (no tolerance, no loss).
                if quantize_series is None or str(quantize_series).lower() == "none":
                    discrete_components = list(base_components)
                    variant = "ideal"
                else:
                    discrete_components = quantize_components(base_components, series=str(quantize_series))
                    variant = "quantized"
                ideal_components = base_components
                ref_freq_hz = float(spec.get("fc_hz") or np.sqrt(float(np.min(freq_hz)) * float(np.max(freq_hz))))
                if need_sim_for_ideal and use_ngspice:
                    ideal_s21_db, ideal_s11_db = simulate_real_waveform(
                        ideal_components,
                        spec,
                        freq_hz,
                        use_ngspice=True,
                        q_L=None,
                        q_C=None,
                        ref_freq_hz=ref_freq_hz,
                        q_model=q_model,
                    )
                else:
                    ideal_s21_db, ideal_s11_db = fast_engine.simulate_sparams_db(
                        ideal_components,
                        freq_hz,
                        q_L=None,
                        q_C=None,
                        q_model="freq_dependent",
                    )
    
                # Input waveform: finite-Q loss model (no tolerance perturbation).
                real_components = list(discrete_components)
                use_spice_real = bool(use_ngspice) and ((q_L is None and q_C is None) or str(q_model) == "fixed_ref")
                if use_spice_real:
                    real_s21_db, real_s11_db = simulate_real_waveform(
                        real_components,
                        spec,
                        freq_hz,
                        use_ngspice=True,
                        q_L=q_L,
                        q_C=q_C,
                        ref_freq_hz=ref_freq_hz,
                        q_model=q_model,
                    )
                else:
                    real_s21_db, real_s11_db = fast_engine.simulate_sparams_db(
                        real_components,
                        freq_hz,
                        q_L=q_L,
                        q_C=q_C,
                        q_model=str(q_model),
                        ref_freq_hz=ref_freq_hz if str(q_model) == "fixed_ref" else None,
                    )
    
                # --- Sanity checks ---
                if real_s21_db is None or np.any(np.isnan(real_s21_db)):
                    print(f"Sample {i}: Simulation failed or NaN.")
                    spec = _resample_full_spec()
                    attempt_in_sample = 0
                    continue
                if check_insertion_loss:
                    is_broken = False
                    ftype = spec.get("filter_type", "lowpass")
                    if ftype == "lowpass":
                        if np.mean(real_s21_db[:10]) < -10.0:
                            is_broken = True
                    elif ftype == "bandpass":
                        mid = len(real_s21_db) // 2
                        window = real_s21_db[max(0, mid - 5) : mid + 5]
                        if np.mean(window) < -10.0:
                            is_broken = True
                    if is_broken:
                        print(f"Sample {i}: Circuit broken (High insertion loss).")
                        spec = _resample_full_spec()
                        attempt_in_sample = 0
                        continue
    
                if ensure_spec:
                    wave_key = str(ensure_spec_wave).lower()
                    if wave_key not in ("real", "ideal"):
                        raise ValueError(f"Unknown ensure_spec_wave={ensure_spec_wave} (expected 'real' or 'ideal').")
                    wave = real_s21_db if wave_key == "real" else ideal_s21_db
                    failed_spec = False
                    if wave is None or np.any(np.isnan(wave)):
                        print(f"Sample {i}: Spec check failed (missing {wave_key} wave).")
                        failed_spec = True
                    elif not _meets_spec(np.asarray(wave, dtype=float), spec_mask_min_db, spec_mask_max_db):
                        failed_spec = True
    
                    if failed_spec:
                        spec_rejects += 1
                        if spec_rejects % 200 == 0:
                            print(f"[spec] rejected {spec_rejects} samples so far.")
                        strategy = str(ensure_spec_strategy).lower()
                        if strategy == "mixed":
                            if attempt_in_sample <= ensure_struct_tries:
                                spec = _resample_structure(spec)
                            elif attempt_in_sample <= ensure_struct_tries + ensure_order_tries:
                                spec = _resample_order_only(spec)
                                if spec is None:
                                    spec = _resample_structure(spec)
                            else:
                                spec = _resample_full_spec()
                                attempt_in_sample = 0
                        elif strategy == "struct":
                            spec = _resample_structure(spec)
                        elif strategy == "order":
                            spec = _resample_order_only(spec)
                            if spec is None:
                                spec = _resample_structure(spec)
                        elif strategy == "resample":
                            spec = _resample_full_spec()
                            attempt_in_sample = 0
                        else:
                            raise ValueError(
                                f"Unknown ensure_spec_strategy={ensure_spec_strategy} "
                                "(expected resample/struct/order/mixed)."
                            )
                        continue
    
                if str(mask_mode).lower() == "spec":
                    mask_min_db, mask_max_db = spec_mask_min_db, spec_mask_max_db
                elif str(mask_mode).lower() == "data":
                    mask_min_db, mask_max_db = build_data_driven_masks(freq_hz, real_s21_db)
                else:
                    raise ValueError(f"Unknown mask_mode={mask_mode} (expected 'data' or 'spec').")
    
                order_token = spec.get("order_effective", spec.get("order"))
                vact_tokens = None
                if emit_vact_tokens:
                    if order_token is None:
                        raise ValueError("ORDER token requested but spec missing order.")
                    vact_tokens = [f"<ORDER_{int(order_token)}>", "<SEP>"] + components_to_vact_tokens(
                        discrete_components,
                        emit_cell_tokens=emit_vact_cells,
                        normalize_node_order=True,
                    )
                vact_struct_tokens = None
                if emit_vact_struct:
                    if order_token is None:
                        raise ValueError("ORDER token requested but spec missing order.")
                    vact_struct_tokens = [f"<ORDER_{int(order_token)}>", "<SEP>"] + components_to_vact_struct_tokens(
                        discrete_components,
                        z0=float(z0),
                        include_ports=True,
                        emit_cells=True,
                    )
                try:
                    macro_ir_macros = components_to_macro_ir(discrete_components)
                except ValueError as exc:
                    print(f"Sample {i}: Macro-IR parse failed ({exc}).")
                    spec = _resample_full_spec()
                    attempt_in_sample = 0
                    continue
                if not macro_ir_macros:
                    print(f"Sample {i}: Macro-IR is empty.")
                    spec = _resample_full_spec()
                    attempt_in_sample = 0
                    continue
                dsl_tokens = None
                dsl_slot_values = None
                if emit_dsl:
                    ftype = str(spec.get("filter_type") or "lowpass")
                    try:
                        segments = components_to_dsl_segments(
                            discrete_components,
                            filter_type=ftype,
                            topology_type=str(spec.get("topology_type") or "t"),
                        )
                        dsl_tokens, dsl_slot_values = components_to_dsl_tokens(
                            [],
                            segments=segments,
                            include_order=dsl_include_order,
                            order=int(order_token) if order_token is not None else None,
                            use_cell_indices=dsl_use_cell_indices,
                            allow_incomplete=not dsl_strict,
                        )
                    except ValueError as exc:
                        print(f"Sample {i}: DSL encode failed ({exc}).")
                        spec = _resample_full_spec()
                        attempt_in_sample = 0
                        continue
                    if dsl_strict and dsl_tokens and VAL_NONE in dsl_tokens:
                        print(f"Sample {i}: DSL contains <VAL_NONE>; skipping due to --dsl-strict.")
                        spec = _resample_full_spec()
                        attempt_in_sample = 0
                        continue
                sfci_tokens = None
                sfci_slot_values = None
                if emit_sfci:
                    mode = str(sfci_value_mode or "discrete").lower()
                    if mode == "continuous":
                        sfci_tokens, sfci_slot_values = components_to_sfci_net_tokens_and_values(
                            discrete_components,
                            value_mode="continuous",
                        )
                    elif mode in ("none", "discrete"):
                        sfci_tokens = components_to_sfci_net_tokens(
                            discrete_components,
                            value_mode="none" if mode == "none" else "discrete",
                        )
                    else:
                        raise ValueError(f"Unknown sfci_value_mode: {sfci_value_mode}")
                action_tokens = components_to_action_tokens(discrete_components) if emit_actions else None
                sample = FilterSample(
                    spec_id=i,
                    circuit_id=i,
                    sample_id=f"{split}_{i}",
                    filter_type=spec["filter_type"],
                    prototype_type=spec["prototype_type"],
                    order=spec["order"],
                    ripple_db=spec["ripple_db"],
                    fc_hz=spec["fc_hz"],
                    variant=variant,
                    z0=z0,
                    num_L=sum(1 for c in base_components if c.ctype == "L"),
                    num_C=sum(1 for c in base_components if c.ctype == "C"),
                    scenario=spec.get("scenario"),
                    scenario_id=spec.get("scenario_id"),
                    bw_frac=spec.get("bw_frac"),
                    freq_range=spec.get("freq_range"),
                    return_loss_min_db=spec.get("return_loss_min_db"),
                    notch_freq_hz=spec.get("notch_freq_hz"),
                    notch_depth_db=spec.get("notch_depth_db"),
                    notch_bw_frac=spec.get("notch_bw_frac"),
                    asymmetry_factor=spec.get("asymmetry_factor"),
                    ideal_components=base_components,
                    discrete_components=discrete_components,
                    json_components=None,
                    freqs_hz=freq_hz,
                    w_ideal_S21_db=ideal_s21_db,
                    w_real_S21_db=real_s21_db,
                    w_ideal_S11_db=ideal_s11_db,
                    w_real_S11_db=real_s11_db,
                    passband_min_db=passband_min_db,
                    stopband_max_db=stopband_max_db,
                    mask_min_db=mask_min_db,
                    mask_max_db=mask_max_db,
                    vact_tokens=vact_tokens,
                    vact_struct_tokens=vact_struct_tokens,
                    dsl_tokens=dsl_tokens,
                    dsl_slot_values=dsl_slot_values,
                    macro_ir_macros=macro_ir_macros,
                    sfci_tokens=sfci_tokens,
                    sfci_slot_values=sfci_slot_values,
                    action_tokens=action_tokens,
                )
    
                f.write(json.dumps(_serialize_sample(sample)) + "\n")
                written += 1
                break
    print(
        f"[done] wrote {written} samples to {jsonl_path} "
        f"(attempts={total_attempts}, spec_rejects={spec_rejects})."
    )
    return jsonl_path
