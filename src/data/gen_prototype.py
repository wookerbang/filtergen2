"""
原型滤波器生成 + 理想波形计算。

当前仅支持 Chebyshev Type I 原型（cheby1），滤波器形状以低通为主。
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Tuple

import numpy as np

from .schema import ComponentSpec


def sample_filter_spec(rng: np.random.Generator | None = None) -> Dict[str, object]:
    """
    随机采样一条设计规格。
    仅使用 cheby1 原型，滤波器类型默认低通以简化流程。
    """
    rng = rng or np.random.default_rng()
    filter_type = "lowpass"
    order = int(rng.integers(2, 7))
    fc = float(10 ** rng.uniform(8.0, 9.7))  # 100MHz ~ 5GHz
    ripple_db = float(rng.uniform(0.1, 0.5))
    z0 = 50.0
    topology_type = rng.choice(["pi", "t"]).item()
    return {
        "filter_type": filter_type,
        "prototype_type": "cheby1",
        "order": order,
        "fc_hz": fc,
        "z0": z0,
        "ripple_db": ripple_db,
        "topology_type": topology_type,
    }


def get_g_values(
    order: int,
    ripple_db: float,
    prototype_type: Literal["cheby1"] = "cheby1",
) -> np.ndarray:
    """
    计算切比一型低通原型的 g 值序列。
    返回 shape [order + 2]，包含 g0 和 g_{n+1}。
    """
    if prototype_type != "cheby1":
        raise ValueError(f"Only cheby1 supported, got {prototype_type}")

    if order < 1:
        raise ValueError("order must be >= 1")

    epsilon = math.sqrt(max(1e-12, 10 ** (ripple_db / 10.0) - 1.0))
    beta = math.asinh(1.0 / epsilon) / order
    sinh_beta = math.sinh(beta)

    a_vals = [math.sin((2 * k - 1) * math.pi / (2 * order)) for k in range(1, order + 1)]
    b_vals = [sinh_beta**2 + math.sin(k * math.pi / order) ** 2 for k in range(1, order + 1)]

    g = np.zeros(order + 2, dtype=float)
    g[0] = 1.0
    g[1] = 2.0 * a_vals[0] / sinh_beta

    for k in range(2, order + 1):
        num = 4.0 * a_vals[k - 2] * a_vals[k - 1]
        den = b_vals[k - 2] * g[k - 1]
        g[k] = num / den

    if order % 2 == 0:
        g[order + 1] = (1.0 / math.tanh(beta / 4.0)) ** 2
    else:
        g[order + 1] = 1.0
    return g


def denormalize_lowpass_to_LC(
    g_values: np.ndarray,
    fc_hz: float,
    z0: float,
    topology_type: Literal["pi", "t"] = "pi",
) -> List[ComponentSpec]:
    """
    将归一化 g 序列映射到具体 L/C 元件值和节点连接。
    节点按 ladder 顺序命名：in -> n1 -> n2 -> ... -> out。
    约定：series 元件连接相邻主路节点；shunt 元件挂在当前主路节点到地。
    """
    w_c = 2.0 * math.pi * fc_hz
    comps: List[ComponentSpec] = []
    g_body = g_values[1:-1]  # 跳过 g0 与 g_{n+1}
    n = len(g_body)

    # 预先确定“主路”节点列表，确保最后一个节点总是 out
    if topology_type == "pi":
        num_series = n // 2  # 偶数索引是串联
    else:
        num_series = (n + 1) // 2  # 奇数索引是串联

    main_nodes = ["in"] + [f"n{k}" for k in range(1, num_series)] + ["out"]
    series_seen = 0

    for idx, g in enumerate(g_body, start=1):
        if g <= 0:
            continue

        is_shunt = (topology_type == "pi" and idx % 2 == 1) or (topology_type == "t" and idx % 2 == 0)

        if is_shunt:
            value = g / (z0 * w_c)
            node = main_nodes[series_seen]
            comps.append(ComponentSpec("C", "shunt", value, None, node, "gnd"))
        else:
            value = g * z0 / w_c
            start_node = main_nodes[series_seen]
            end_node = main_nodes[series_seen + 1]
            comps.append(ComponentSpec("L", "series", value, None, start_node, end_node))
            series_seen += 1

    return comps


def compute_ideal_waveform(
    components: List[ComponentSpec],
    spec: Dict[str, object],
    freq_hz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 ABCD 矩阵级联计算理想 S21, S11（无寄生/离散误差）。
    """
    from .circuits import abcd_to_sparams, components_to_abcd

    z0 = float(spec["z0"])
    A, B, C, D = components_to_abcd(components, freq_hz, z0)
    s21_db, s11_db = abcd_to_sparams(A, B, C, D, z0)
    return s21_db, s11_db
