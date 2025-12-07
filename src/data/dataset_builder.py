"""
将各步骤串联生成样本并落盘。
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .gen_prototype import (
    compute_ideal_waveform,
    denormalize_lowpass_to_LC,
    get_g_values,
    sample_filter_spec,
)
from .quantization import quantize_components
from .schema import ComponentSpec, FilterSample
from .sfci_codec import components_to_sfci
from .sfci_codec_netcentric import components_to_netcentric
from .spice_runner import simulate_real_waveform


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
            "ideal_components": _serialize_components(sample.ideal_components or []),
            "discrete_components": _serialize_components(sample.discrete_components or []),
            "sfci_tokens": sample.sfci_tokens or [],
            "sfci_tokens_net": sample.sfci_tokens_net or [],
        }
    )
    return data


def build_dataset(
    num_samples: int,
    output_dir: str,
    split: str = "train",
    use_ngspice: bool = False,
    seed: int = 42,
) -> str:
    """
    串起采样 → 原型 → 离散化 → 仿真 → 序列化。
    返回写入的 jsonl 路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split}.jsonl")

    rng = np.random.default_rng(seed)

    with open(jsonl_path, "w") as f:
        for i in range(num_samples):
            spec = sample_filter_spec(rng=rng)
            z0 = spec["z0"]
            g = get_g_values(spec["order"], spec["ripple_db"], prototype_type="cheby1")
            ideal_components = denormalize_lowpass_to_LC(g, spec["fc_hz"], z0, spec["topology_type"])

            freq_hz = np.logspace(np.log10(spec["fc_hz"] / 10.0), np.log10(spec["fc_hz"] * 10.0), 256)
            ideal_s21_db, ideal_s11_db = compute_ideal_waveform(ideal_components, spec, freq_hz)

            discrete_components = quantize_components(ideal_components, series="E24")
            real_s21_db, real_s11_db = simulate_real_waveform(discrete_components, spec, freq_hz, use_ngspice=use_ngspice)

            # --- Sanity checks ---
            if real_s21_db is None or np.any(np.isnan(real_s21_db)):
                print(f"Sample {i}: Simulation failed or NaN.")
                continue
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

            sfci_tokens = [f"<ORDER_{spec['order']}>", "<SEP>"] + components_to_sfci(discrete_components)
            sfci_tokens_net = components_to_netcentric(discrete_components)
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
                num_L=sum(1 for c in ideal_components if c.ctype == "L"),
                num_C=sum(1 for c in ideal_components if c.ctype == "C"),
                ideal_components=ideal_components,
                discrete_components=discrete_components,
                json_components=None,
                freqs_hz=freq_hz,
                w_ideal_S21_db=ideal_s21_db,
                w_real_S21_db=real_s21_db,
                w_ideal_S11_db=ideal_s11_db,
                w_real_S11_db=real_s11_db,
                sfci_sequence=" ".join(sfci_tokens),
                sfci_tokens=sfci_tokens,
                sfci_tokens_net=sfci_tokens_net,
            )

            f.write(json.dumps(_serialize_sample(sample)) + "\n")
    return jsonl_path
