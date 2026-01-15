"""
原型滤波器生成 + 理想波形计算（工具化接口）。

支持 Chebyshev Type I 与 Butterworth 原型，覆盖 LP/HP/BP/BS。
Scenario 层应负责具体任务簇采样与后处理（例如 notch）。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np

from .schema import ComponentSpec


def sample_base_spec(
    *,
    rng: np.random.Generator | None = None,
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] | Sequence[str] = "lowpass",
    order_range: Tuple[int, int] = (3, 7),
    fc_range_hz: Tuple[float, float] = (1e8, 5e9),
    ripple_db_range: Tuple[float, float] = (0.1, 0.5),
    prototype_types: Sequence[str] = ("cheby1", "butter"),
    topology_types: Sequence[str] = ("pi", "t"),
    bw_frac_range: Tuple[float, float] = (0.05, 0.3),
    z0: float = 50.0,
) -> Dict[str, object]:
    """
    采样一个“基础滤波器”规格（不包含任务场景/后处理）。
    Scenario 层应在此基础上补充场景字段（如 notch 频点、非对称约束等）。
    """
    rng = rng or np.random.default_rng()
    if isinstance(filter_type, str):
        ftype = filter_type
    else:
        ftype = rng.choice(list(filter_type)).item()
    order_lo, order_hi = (int(order_range[0]), int(order_range[1]))
    if ftype == "bandpass":
        order_lo = max(order_lo, 4)
        order_hi = max(order_hi, order_lo)
    order = int(rng.integers(order_lo, order_hi + 1))
    fc = float(10 ** rng.uniform(np.log10(float(fc_range_hz[0])), np.log10(float(fc_range_hz[1]))))
    ripple_db = float(rng.uniform(float(ripple_db_range[0]), float(ripple_db_range[1])))
    prototype_type = rng.choice(list(prototype_types)).item()
    topology_type = rng.choice(list(topology_types)).item()
    if ftype == "bandstop" and "t" in topology_types:
        topology_type = "t"
    bw_frac = None
    freq_range = None
    if ftype in ("bandpass", "bandstop"):
        bw_frac = float(rng.uniform(float(bw_frac_range[0]), float(bw_frac_range[1])))
        f_min = fc * (1.0 - 0.5 * bw_frac)
        f_max = fc * (1.0 + 0.5 * bw_frac)
        if f_min <= 0:
            f_min = fc / 10.0
        freq_range = [float(f_min), float(f_max)]
    return {
        "filter_type": ftype,
        "prototype_type": prototype_type,
        "order": order,
        "fc_hz": fc,
        "bw_frac": bw_frac,
        "freq_range": freq_range,
        "z0": float(z0),
        "ripple_db": ripple_db,
        "topology_type": topology_type,
    }


def get_g_values(
    order: int,
    ripple_db: float,
    prototype_type: Literal["cheby1", "butter"] = "cheby1",
) -> np.ndarray:
    """
    计算低通归一化原型的 g 值序列。
    返回 shape [order + 2]，包含 g0 和 g_{n+1}。
    """

    if order < 1:
        raise ValueError("order must be >= 1")

    if prototype_type == "cheby1":
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

    if prototype_type == "butter":
        # 参照经典 Butterworth 原型闭式解
        g = np.zeros(order + 2, dtype=float)
        g[0] = 1.0
        for k in range(1, order + 1):
            g[k] = 2.0 * math.sin((2 * k - 1) * math.pi / (2 * order))
        g[order + 1] = 1.0
        return g

    raise ValueError(f"Unsupported prototype_type: {prototype_type}")


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


def denormalize_highpass_to_LC(
    g_values: np.ndarray,
    fc_hz: float,
    z0: float,
    topology_type: Literal["pi", "t"] = "pi",
) -> List[ComponentSpec]:
    """
    低通原型 -> 高频变换：串联电感 -> 串联电容；并联电容 -> 并联电感。
    """
    w_c = 2.0 * math.pi * fc_hz
    comps: List[ComponentSpec] = []
    g_body = g_values[1:-1]
    n = len(g_body)
    if topology_type == "pi":
        num_series = n // 2
    else:
        num_series = (n + 1) // 2
    main_nodes = ["in"] + [f"n{k}" for k in range(1, num_series)] + ["out"]
    series_seen = 0

    for idx, g in enumerate(g_body, start=1):
        if g <= 0:
            continue
        is_shunt = (topology_type == "pi" and idx % 2 == 1) or (topology_type == "t" and idx % 2 == 0)
        if is_shunt:
            value = z0 * g / w_c  # shunt L
            node = main_nodes[series_seen]
            comps.append(ComponentSpec("L", "shunt", value, None, node, "gnd"))
        else:
            value = 1.0 / (w_c * z0 * g)  # series C
            start_node = main_nodes[series_seen]
            end_node = main_nodes[series_seen + 1]
            comps.append(ComponentSpec("C", "series", value, None, start_node, end_node))
            series_seen += 1
    return comps


def denormalize_bandpass_to_LC(
    g_values: np.ndarray,
    fc_hz: float,
    z0: float,
    fbw: float,
    topology_type: Literal["pi", "t"] = "pi",
) -> List[ComponentSpec]:
    """
    低通原型 -> 带通变换（窄带近似）：串联元件 -> 串联 LC；并联元件 -> 并联 LC。
    为简单起见，串联 LC 通过引入中间节点实现。
    """
    w0 = 2.0 * math.pi * fc_hz
    comps: List[ComponentSpec] = []
    g_body = g_values[1:-1]
    n = len(g_body)
    if topology_type == "pi":
        num_series = n // 2
    else:
        num_series = (n + 1) // 2
    main_nodes = ["in"] + [f"n{k}" for k in range(1, num_series)] + ["out"]
    series_seen = 0
    series_counter = 0

    for idx, g in enumerate(g_body, start=1):
        if g <= 0:
            continue
        is_shunt = (topology_type == "pi" and idx % 2 == 1) or (topology_type == "t" and idx % 2 == 0)
        if is_shunt:
            # 并联 LC to gnd
            Lp = z0 * fbw / (w0 * g)
            Cp = g / (w0 * z0 * fbw)
            node = main_nodes[series_seen]
            comps.append(ComponentSpec("L", "shunt", Lp, None, node, "gnd"))
            comps.append(ComponentSpec("C", "shunt", Cp, None, node, "gnd"))
        else:
            # 串联 LC，插入一个中间节点
            Ls = z0 * g / (w0 * fbw)
            Cs = fbw / (w0 * z0 * g)
            start_node = main_nodes[series_seen]
            end_node = main_nodes[series_seen + 1]
            mid_node = f"bp_mid{series_counter}"
            series_counter += 1
            comps.append(ComponentSpec("L", "series", Ls, None, start_node, mid_node))
            comps.append(ComponentSpec("C", "series", Cs, None, mid_node, end_node))
            series_seen += 1
    return comps


def denormalize_bandstop_to_LC(
    g_values: np.ndarray,
    fc_hz: float,
    z0: float,
    fbw: float,
    topology_type: Literal["pi", "t"] = "pi",
) -> List[ComponentSpec]:
    """
    低通原型 -> 带阻变换（窄带近似）：
      - series 元件 -> 并联 LC（串在主路）
      - shunt 元件 -> 串联 LC（接地支路）
    """
    w0 = 2.0 * math.pi * fc_hz
    comps: List[ComponentSpec] = []
    g_body = g_values[1:-1]
    n = len(g_body)
    if topology_type == "pi":
        num_series = n // 2
    else:
        num_series = (n + 1) // 2
    main_nodes = ["in"] + [f"n{k}" for k in range(1, num_series)] + ["out"]
    series_seen = 0
    shunt_counter = 0

    for idx, g in enumerate(g_body, start=1):
        if g <= 0:
            continue
        is_shunt = (topology_type == "pi" and idx % 2 == 1) or (topology_type == "t" and idx % 2 == 0)
        if is_shunt:
            # shunt: series LC to gnd
            Ls = z0 * fbw / (w0 * g)
            Cs = g / (w0 * z0 * fbw)
            node = main_nodes[series_seen]
            mid_node = f"bs_mid{shunt_counter}"
            shunt_counter += 1
            comps.append(ComponentSpec("L", "series", Ls, None, node, mid_node))
            comps.append(ComponentSpec("C", "series", Cs, None, mid_node, "gnd"))
        else:
            # series: parallel LC across main path
            Lp = z0 * g / (w0 * fbw)
            Cp = fbw / (w0 * z0 * g)
            start_node = main_nodes[series_seen]
            end_node = main_nodes[series_seen + 1]
            comps.append(ComponentSpec("L", "series", Lp, None, start_node, end_node))
            comps.append(ComponentSpec("C", "series", Cp, None, start_node, end_node))
            series_seen += 1
    return comps


def _map_node_name(node: str, *, node_map: Dict[str, str], prefix: str | None = None) -> str:
    if node in node_map:
        return node_map[node]
    if node in ("in", "out", "gnd"):
        return node
    if prefix:
        return f"{prefix}_{node}"
    return node


def _remap_components(
    components: Sequence[ComponentSpec],
    *,
    node_map: Dict[str, str],
    prefix: str | None = None,
) -> List[ComponentSpec]:
    remapped: List[ComponentSpec] = []
    for c in components:
        n1 = _map_node_name(c.node1, node_map=node_map, prefix=prefix)
        n2 = _map_node_name(c.node2, node_map=node_map, prefix=prefix)
        remapped.append(ComponentSpec(c.ctype, c.role, c.value_si, c.std_label, n1, n2))
    return remapped


def synthesize_cascade_bandpass(
    spec: Dict[str, object],
    order: int,
    z0: float,
    *,
    rng: np.random.Generator | None = None,
) -> List[ComponentSpec]:
    """
    Wideband BP via LP+HP cascade.

    Requires spec["freq_range"] = [f_min, f_max]. Falls back to fc/bw if missing.
    """
    rng = rng or np.random.default_rng()
    freq_range = spec.get("freq_range")
    if freq_range is not None:
        f_min, f_max = float(freq_range[0]), float(freq_range[1])
    else:
        fc = float(spec.get("fc_hz") or 1.0)
        bw = float(spec.get("bw_frac") or 0.2)
        f_min = fc * (1.0 - 0.5 * bw)
        f_max = fc * (1.0 + 0.5 * bw)
    if f_min <= 0 or f_max <= 0:
        raise ValueError(f"Invalid freq_range for cascade bandpass: [{f_min}, {f_max}]")
    if f_min >= f_max:
        f_min, f_max = f_max, f_min

    proto = str(spec.get("prototype_type") or "cheby1")
    topology = str(spec.get("topology_type") or "pi")

    base_order = int(order)
    if base_order < 4:
        # Cascade needs >=2 sections per side to keep a valid series path.
        base_order = 4
        spec["order_effective"] = base_order
    order_lp_override = spec.get("bp_order_lp")
    order_hp_override = spec.get("bp_order_hp")
    orders_overridden = order_lp_override is not None or order_hp_override is not None
    order_lp: int
    order_hp: int
    if order_lp_override is None and order_hp_override is None:
        order_lp = int(rng.integers(2, base_order - 1))  # [2, order-2]
        order_hp = int(base_order - order_lp)
    else:
        if order_lp_override is not None:
            order_lp = int(order_lp_override)
        else:
            order_lp = -1
        if order_hp_override is not None:
            order_hp = int(order_hp_override)
        else:
            order_hp = -1
        if order_lp < 0 and order_hp < 0:
            order_lp = int(rng.integers(2, base_order - 1))
            order_hp = int(base_order - order_lp)
        elif order_lp < 0:
            order_hp = max(0, order_hp)
            order_lp = int(base_order - order_hp)
        elif order_hp < 0:
            order_lp = max(0, order_lp)
            order_hp = int(base_order - order_lp)
        if order_lp + order_hp != base_order:
            base_order = int(order_lp + order_hp)
            spec["order_effective"] = base_order
        if orders_overridden:
            spec["order"] = base_order
        if order_lp < 2 or order_hp < 2:
            raise ValueError(
                f"Invalid BP cascade orders: order_lp={order_lp}, order_hp={order_hp}. Each must be >=2."
            )

    g_lp = get_g_values(order_lp, float(spec.get("ripple_db") or 0.5), prototype_type=proto)
    g_hp = get_g_values(order_hp, float(spec.get("ripple_db") or 0.5), prototype_type=proto)

    comps_lp = denormalize_lowpass_to_LC(g_lp, f_max, z0, topology)
    comps_hp = denormalize_highpass_to_LC(g_hp, f_min, z0, topology)

    order_mode = str(spec.get("bp_cascade_order") or "random").lower()
    if order_mode in ("lp_hp", "lp-hp", "lp->hp", "lphp"):
        first, second = comps_lp, comps_hp
    elif order_mode in ("hp_lp", "hp-lp", "hp->lp", "hplp"):
        first, second = comps_hp, comps_lp
    else:
        if rng.random() < 0.5:
            first, second = comps_lp, comps_hp
        else:
            first, second = comps_hp, comps_lp

    join_node = "bp_join"
    first_remap = _remap_components(first, node_map={"out": join_node}, prefix=None)
    second_remap = _remap_components(second, node_map={"in": join_node, "out": "out"}, prefix="bp2")
    return first_remap + second_remap


def synthesize_filter(spec: Dict[str, object]) -> List[ComponentSpec]:
    """根据 filter_type/prototype_type 将原型 g 值映射到具体 LC 电路。"""
    g = get_g_values(spec["order"], spec["ripple_db"], prototype_type=spec["prototype_type"])
    ftype = spec.get("filter_type", "lowpass")
    if ftype == "lowpass":
        return denormalize_lowpass_to_LC(g, spec["fc_hz"], spec["z0"], spec["topology_type"])
    if ftype == "highpass":
        return denormalize_highpass_to_LC(g, spec["fc_hz"], spec["z0"], spec["topology_type"])
    if ftype == "bandpass":
        return synthesize_cascade_bandpass(spec, int(spec["order"]), float(spec["z0"]))
    if ftype == "bandstop":
        fbw = float(spec.get("bw_frac") or spec.get("stopband_bw_frac") or 0.2)
        return denormalize_bandstop_to_LC(g, spec["fc_hz"], spec["z0"], fbw, spec["topology_type"])
    # 回退：按低通处理
    return denormalize_lowpass_to_LC(g, spec["fc_hz"], spec["z0"], spec["topology_type"])


def pick_anchor_node(components: Iterable[ComponentSpec]) -> str:
    candidates: List[str] = []
    seen: set[str] = set()
    for c in components:
        for node in (c.node1, c.node2):
            if node in ("gnd", "out"):
                continue
            if node not in seen:
                seen.add(node)
                candidates.append(node)
    return candidates[0] if candidates else "in"


def insert_shunt_series_lc(
    components: Iterable[ComponentSpec],
    *,
    anchor: str | None = None,
    notch_freq_hz: float,
    rng: np.random.Generator | None = None,
    c_range_pf: Tuple[float, float] = (0.5, 5.0),
    mid_prefix: str = "notch",
) -> Tuple[List[ComponentSpec], float, float, str]:
    """
    在给定 anchor 节点挂一个串联 LC 到地（用于 notch）。
    返回 (new_components, L_value, C_value, anchor_node)。
    """
    rng = rng or np.random.default_rng()
    comps = list(components)
    anchor_node = anchor or pick_anchor_node(comps)
    c_min, c_max = float(c_range_pf[0]), float(c_range_pf[1])
    Cn = 1e-12 * float(rng.uniform(c_min, c_max))
    Ln = 1.0 / ((2 * math.pi * float(notch_freq_hz)) ** 2 * Cn)
    mid_node = f"{anchor_node}_{mid_prefix}"
    comps.append(ComponentSpec("L", "series", Ln, None, anchor_node, mid_node))
    comps.append(ComponentSpec("C", "series", Cn, None, mid_node, "gnd"))
    return comps, float(Ln), float(Cn), anchor_node


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
