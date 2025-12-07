"""
电路中间表示 + ABCD 工具。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .schema import ComponentSpec


def _infer_output_node(components: List[ComponentSpec]) -> str:
    for comp in reversed(components):
        if comp.node2 != "gnd":
            return comp.node2
        if comp.node1 != "gnd":
            return comp.node1
    return "out"


class Circuit:
    def __init__(
        self,
        components: List[ComponentSpec],
        z0: float = 50.0,
        in_port: Tuple[str, str] = ("in", "gnd"),
        out_port: Tuple[str, str] | None = None,
    ):
        self.components = components
        self.z0 = z0
        self.in_port = in_port
        self.out_port = out_port or (_infer_output_node(components), "gnd")

    def to_spice_netlist(self, title: str = "LC_FILTER") -> str:
        """
        输出一个简单的 2-port AC 仿真 netlist。
        """
        lines = [f"* {title}"]
        src_node, gnd_node = self.in_port
        out_node, _ = self.out_port
        lines.append(f"V1 {src_node} {gnd_node} AC 1")
        lines.append(f"Rload {out_node} {gnd_node} {self.z0}")

        for idx, comp in enumerate(self.components, start=1):
            name = f"{comp.ctype}{idx}"
            value = f"{comp.value_si}"
            lines.append(f"{name} {comp.node1} {comp.node2} {value}")

        # 控制语句由调用者补充
        return "\n".join(lines)


def components_to_abcd(
    components: List[ComponentSpec],
    freq_hz: np.ndarray,
    z0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * freq_hz
    n = len(freq_hz)
    A = np.ones(n, dtype=complex)
    B = np.zeros(n, dtype=complex)
    C = np.zeros(n, dtype=complex)
    D = np.ones(n, dtype=complex)

    for comp in components:
        if comp.ctype == "L" and comp.role == "series":
            Z = 1j * omega * comp.value_si
            B = A * Z + B
            D = C * Z + D
        elif comp.ctype == "C" and comp.role == "shunt":
            Y = 1j * omega * comp.value_si
            A = A + B * Y
            C = C + D * Y
        else:
            continue
    return A, B, C, D


def abcd_to_sparams(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    z0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    denom = A + B / z0 + C * z0 + D
    S21 = 2.0 / (denom + 1e-18)
    S11 = (A + B / z0 - C * z0 - D) / (denom + 1e-18)
    s21_db = 20.0 * np.log10(np.abs(S21) + 1e-12)
    s11_db = 20.0 * np.log10(np.abs(S11) + 1e-12)
    return s21_db, s11_db
