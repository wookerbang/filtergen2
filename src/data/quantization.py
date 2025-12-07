"""
连续元件值映射到标准 E 系列。
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .schema import ComponentSpec

E12_MANT = np.array(
    [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
    dtype=float,
)
E24_MANT = np.array(
    [
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.7,
        3.0,
        3.3,
        3.6,
        3.9,
        4.3,
        4.7,
        5.1,
        5.6,
        6.2,
        6.8,
        7.5,
        8.2,
        9.1,
    ],
    dtype=float,
)


def _nearest_e_series_value(value_si: float, series: str, base_unit: float) -> float:
    """
    找到距离 value_si 最近的 E12/E24 值。
    base_unit 决定指数起点，例如 nH -> 1e-9。
    """
    if value_si <= 0:
        return value_si

    mant = E24_MANT if series.upper() == "E24" else E12_MANT
    v_norm = value_si / base_unit
    exp_guess = int(math.floor(math.log10(v_norm)))

    best_val = value_si
    best_err = float("inf")

    for exp in range(exp_guess - 1, exp_guess + 2):
        candidates = mant * (10.0**exp) * base_unit
        diffs = np.abs(candidates - value_si)
        idx = int(np.argmin(diffs))
        cand = float(candidates[idx])
        err = float(diffs[idx])
        if err < best_err:
            best_err = err
            best_val = cand
    return best_val


def _format_value(v: float, unit: str | None = None) -> str:
    """
    将 SI 值格式化为带前缀的短字符串，例如 3.3e-9 -> '3.3nH'。
    """
    if v == 0:
        return f"0{unit or ''}"

    prefixes = [
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "u"),
        (1e-3, "m"),
        (1.0, ""),
        (1e3, "k"),
        (1e6, "M"),
        (1e9, "G"),
    ]

    abs_v = abs(v)
    chosen_scale, chosen_prefix = prefixes[0]
    for scale, prefix in prefixes:
        if abs_v >= scale:
            chosen_scale, chosen_prefix = scale, prefix
    value = v / chosen_scale
    suffix = unit or ""
    return f"{value:.3g}{chosen_prefix}{suffix}"


_PREFIX_TO_SCALE = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}


def label_to_value(label: str) -> float:
    """
    解析形如 'L_3.3nH' 或 'C_4.7pF' 的 std_label -> SI 数值。
    """
    if not label or "_" not in label:
        raise ValueError(f"Unrecognized label: {label}")
    _, magnitude = label.split("_", 1)
    unit = magnitude[-1]  # H 或 F
    # 前缀字符（可能不存在）
    prefix_char = magnitude[-2] if magnitude[-2].isalpha() else ""
    if prefix_char.lower() == unit.lower():
        prefix_char = ""
    scale = _PREFIX_TO_SCALE.get(prefix_char, 1.0)
    number_str = magnitude.replace(prefix_char + unit, "")
    return float(number_str) * scale


def generate_value_labels(
    series: str = "E24",
    kind: str = "L",
    exp_min: int = -12,
    exp_max: int = 3,
) -> list[str]:
    """
    生成一套标准值 label 列表，用于 tokenizer 词表构建。
    kind: 'L' -> Henry, 'C' -> Farad
    exp_min/exp_max: 10^exp 量级范围
    """
    mant = E24_MANT if series.upper() == "E24" else E12_MANT
    unit = "H" if kind.upper() == "L" else "F"
    labels: list[str] = []
    for exp in range(exp_min, exp_max + 1):
        for m in mant:
            val = m * (10.0**exp)
            labels.append(f"{kind.upper()}_{_format_value(val, unit)}")
    return labels


def quantize_components(ideal_components: List[ComponentSpec], series: str = "E24") -> List[ComponentSpec]:
    """
    将连续值元件映射到标准值，并附加 std_label。
    """
    discrete: List[ComponentSpec] = []
    for comp in ideal_components:
        if comp.ctype == "L":
            std_value = _nearest_e_series_value(comp.value_si, series, base_unit=1e-9)
            label = f"L_{_format_value(std_value, 'H')}"
        elif comp.ctype == "C":
            std_value = _nearest_e_series_value(comp.value_si, series, base_unit=1e-12)
            label = f"C_{_format_value(std_value, 'F')}"
        else:
            std_value = comp.value_si
            label = comp.std_label

        discrete.append(
            ComponentSpec(
                ctype=comp.ctype,
                role=comp.role,
                value_si=std_value,
                std_label=label,
                node1=comp.node1,
                node2=comp.node2,
            )
        )
    return discrete
