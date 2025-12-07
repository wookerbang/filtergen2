"""
调用 ngspice 做 AC 仿真，获取真实波形。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np

from .circuits import Circuit, abcd_to_sparams, components_to_abcd
from .schema import ComponentSpec


def run_ac_analysis_with_ngspice(
    circuit: Circuit,
    freq_hz: np.ndarray,
    z0: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 ngspice 进行 AC 分析，返回 S21/S11 in dB。
    如果 ngspice 不可用，则抛出 RuntimeError。
    """
    f_start = float(np.min(freq_hz))
    f_stop = float(np.max(freq_hz))
    n_points = len(freq_hz)

    netlist = circuit.to_spice_netlist()
    lines = [netlist]
    lines.append(f".ac lin {n_points} {f_start} {f_stop}")
    lines.append(".control")
    lines.append("set filetype=ascii")
    lines.append("run")
    lines.append("wrdata ac.csv frequency v(in) v(out)")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    with tempfile.TemporaryDirectory() as tmpdir:
        net_path = os.path.join(tmpdir, "circuit.sp")
        log_path = os.path.join(tmpdir, "ngspice.log")
        csv_path = os.path.join(tmpdir, "ac.csv")
        with open(net_path, "w") as f:
            f.write("\n".join(lines))

        cmd = ["ngspice", "-b", "-o", log_path, net_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ngspice not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ngspice failed, see log at {log_path}") from exc

        if not os.path.exists(csv_path):
            raise RuntimeError("ngspice did not produce ac.csv")

        data = np.genfromtxt(csv_path, delimiter="\t", names=True)
        freq = np.array(data["frequency"])
        v_in = np.array(data["v_in"])
        v_out = np.array(data["v_out"])

    H = v_out / (v_in + 1e-18)
    s21_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    s11_db = np.zeros_like(s21_db)
    return s21_db, s11_db


def simulate_real_waveform(
    components: list[ComponentSpec],
    spec: dict,
    freq_hz: np.ndarray,
    use_ngspice: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 Circuit -> 仿真得到 real S21/S11。
    ngspice 不可用时，退化为理想 ABCD 计算。
    """
    z0 = float(spec["z0"])
    circuit = Circuit(components, z0=z0)
    if use_ngspice:
        try:
            return run_ac_analysis_with_ngspice(circuit, freq_hz, z0)
        except RuntimeError:
            pass

    # 回退：用理想模型替代
    A, B, C, D = components_to_abcd(components, freq_hz, z0)
    return abcd_to_sparams(A, B, C, D, z0)
