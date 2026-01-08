"""
将各步骤串联生成样本并落盘。
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch

from .gen_prototype import synthesize_filter
from .quantization import quantize_components
from .schema import ComponentSpec, FilterSample
from .vact_codec import components_to_vact_tokens
from .sfci_net_codec import components_to_sfci_net_tokens
from .spice_runner import simulate_real_waveform
from .vact_struct import components_to_vact_struct_tokens
from .node_canonicalizer import canonicalize_nodes
from .action_codec import components_to_action_tokens
from .dsl import VAL_NONE, components_to_dsl_segments, components_to_dsl_tokens
from .scenarios import apply_scenario_postprocess, build_freq_grid, build_spec_masks, sample_scenario_spec
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
    dsl_include_order: bool = True,
    dsl_use_cell_indices: bool = False,
    dsl_strict: bool = False,
    max_nodes: int = 32,
    q_L: float | None = 50.0,
    q_C: float | None = 50.0,
    tol_frac: float = 0.05,
    q_model: str = "freq_dependent",
    check_insertion_loss: bool = True,
    filter_type_override: str | None = None,
    prototype_type_override: str | None = None,
    topology_type_override: str | None = None,
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

    def _apply_tolerance(comps: List[ComponentSpec]) -> List[ComponentSpec]:
        t = float(tol_frac or 0.0)
        if t <= 0:
            return list(comps)
        out: List[ComponentSpec] = []
        for c in comps:
            scale = float(rng.uniform(1.0 - t, 1.0 + t))
            out.append(
                ComponentSpec(
                    ctype=c.ctype,
                    role=c.role,
                    value_si=float(c.value_si) * scale,
                    std_label=c.std_label,
                    node1=c.node1,
                    node2=c.node2,
                )
            )
        return out

    with open(jsonl_path, "w") as f:
        for i in range(num_samples):
            proto_override = None
            if prototype_type_override is not None:
                proto_override = (str(prototype_type_override),)
            spec = sample_scenario_spec(
                rng=rng,
                scenario=scenario,
                scenario_weights=scenario_weights,
                filter_type_override=str(filter_type_override) if filter_type_override is not None else None,
                prototype_types_override=proto_override,
                topology_type_override=str(topology_type_override) if topology_type_override is not None else None,
            )
            z0 = spec["z0"]
            if fast_engine is None or float(z0) != float(fast_engine.z0):
                fast_engine = FastTrackEngine(z0=float(z0), device="cpu", dtype=torch.float64)
            base_components = synthesize_filter(spec)
            base_components = apply_scenario_postprocess(base_components, spec, rng=rng)

            freq_hz = build_freq_grid(spec, num_freqs=256)
            mask_min_db, mask_max_db, passband_min_db, stopband_max_db = build_spec_masks(spec, freq_hz)

            # 非纯 ladder（notch/BP/BS）优先用仿真获取 ideal，以避免 ABCD 近似误差
            need_sim_for_ideal = spec.get("scenario") in ("anti_jamming",) or spec.get("filter_type") != "lowpass"

            # Canonicalize node names so tokenization stays in-vocab.
            base_components = canonicalize_nodes(base_components, max_nodes=max_nodes)

            # Output label: nominal standard parts (no tolerance, no loss).
            discrete_components = quantize_components(base_components, series="E24")
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

            # Input waveform: tolerance-perturbed + finite-Q loss model.
            real_components = _apply_tolerance(discrete_components)
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
                    continue

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
                    continue
                if dsl_strict and dsl_tokens and VAL_NONE in dsl_tokens:
                    print(f"Sample {i}: DSL contains <VAL_NONE>; skipping due to --dsl-strict.")
                    continue
            sfci_tokens = components_to_sfci_net_tokens(discrete_components) if emit_sfci else None
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
                variant="quantized",
                z0=z0,
                num_L=sum(1 for c in base_components if c.ctype == "L"),
                num_C=sum(1 for c in base_components if c.ctype == "C"),
                scenario=spec.get("scenario"),
                scenario_id=spec.get("scenario_id"),
                bw_frac=spec.get("bw_frac"),
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
                sfci_tokens=sfci_tokens,
                action_tokens=action_tokens,
            )

            f.write(json.dumps(_serialize_sample(sample)) + "\n")
    return jsonl_path
