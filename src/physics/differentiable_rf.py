from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import ComponentSpec


def _as_tensor(
    x: torch.Tensor | Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"Unsupported real dtype for complex simulation: {real_dtype}")


class DifferentiablePhysicsKernel:
    """
    Atomic differentiable RF operators for 2-port cascaded networks.

    Treat each component as a 2-port "layer" with ABCD transfer matrix:
      - series impedance Z: [[1, Z], [0, 1]]
      - shunt admittance Y: [[1, 0], [Y, 1]]

    This kernel is fully differentiable end-to-end (including complex ops)
    and supports batch parallelism via broadcasting.
    """

    OP_SERIES_L = 0
    OP_SERIES_C = 1
    OP_SHUNT_L = 2
    OP_SHUNT_C = 3
    OP_SHUNT_SERIES_LC = 4
    OP_SERIES_PARALLEL_LC = 5

    @staticmethod
    def omega(freq_hz: torch.Tensor) -> torch.Tensor:
        return 2.0 * math.pi * freq_hz

    @staticmethod
    def _jw(omega: torch.Tensor) -> torch.Tensor:
        return 1j * omega

    @staticmethod
    def _loss_omega(
        omega: torch.Tensor,
        *,
        q_model: Literal["freq_dependent", "fixed_ref"],
        ref_omega: torch.Tensor | None,
    ) -> torch.Tensor:
        if q_model == "freq_dependent":
            return omega
        if q_model == "fixed_ref":
            if ref_omega is None:
                raise ValueError("ref_omega is required when q_model='fixed_ref' and Q is enabled.")
            return ref_omega
        raise ValueError(f"Unknown q_model: {q_model}")

    @staticmethod
    def series_impedance(
        L_or_C: torch.Tensor,
        omega: torch.Tensor,
        *,
        kind: Literal["L", "C"],
        q: torch.Tensor | None,
        eps: float,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_omega: torch.Tensor | None = None,
    ) -> torch.Tensor:
        jw = DifferentiablePhysicsKernel._jw(omega)
        if kind == "L":
            # Series RL: Z = R + jωL, where R = ωL/Q.
            Z = jw * L_or_C
            if q is not None:
                loss_omega = DifferentiablePhysicsKernel._loss_omega(omega, q_model=q_model, ref_omega=ref_omega)
                Z = Z + (loss_omega * L_or_C) / q
            return Z
        if kind == "C":
            # Parallel RC branch between the two nodes: Y = G + jωC, where G = ωC/Q.
            Y = jw * L_or_C
            if q is not None:
                loss_omega = DifferentiablePhysicsKernel._loss_omega(omega, q_model=q_model, ref_omega=ref_omega)
                Y = Y + (loss_omega * L_or_C) / q
            return 1.0 / (Y + eps)
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def shunt_admittance(
        L_or_C: torch.Tensor,
        omega: torch.Tensor,
        *,
        kind: Literal["L", "C"],
        q: torch.Tensor | None,
        eps: float,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_omega: torch.Tensor | None = None,
    ) -> torch.Tensor:
        jw = DifferentiablePhysicsKernel._jw(omega)
        if kind == "C":
            Y = jw * L_or_C
            if q is not None:
                loss_omega = DifferentiablePhysicsKernel._loss_omega(omega, q_model=q_model, ref_omega=ref_omega)
                Y = Y + (loss_omega * L_or_C) / q
            return Y
        if kind == "L":
            Z = jw * L_or_C
            if q is not None:
                loss_omega = DifferentiablePhysicsKernel._loss_omega(omega, q_model=q_model, ref_omega=ref_omega)
                Z = Z + (loss_omega * L_or_C) / q
            return 1.0 / (Z + eps)
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def cascade_abcd(
        op_codes: Sequence[int] | torch.Tensor,
        values: torch.Tensor,
        freq_hz: torch.Tensor,
        *,
        op_param_counts: Sequence[int] | None = None,
        q_L: float | torch.Tensor | None = None,
        q_C: float | torch.Tensor | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | torch.Tensor | None = None,
        eps: float = 1e-30,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute overall ABCD for a cascaded network.

        Args:
            op_codes: length-N sequence of op codes (series/shunt, L/C).
            values: (..., N) component values in SI units.
            freq_hz: (F,) frequency axis in Hz.
        Returns:
            (A, B, C, D): each shape (..., F) complex.
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if values.ndim < 1:
            raise ValueError(f"values must have at least 1 dimension, got shape={tuple(values.shape)}")

        if q_model not in ("freq_dependent", "fixed_ref"):
            raise ValueError(f"Unknown q_model: {q_model}")

        if isinstance(op_codes, torch.Tensor):
            codes = [int(x) for x in op_codes.detach().cpu().tolist()]
        else:
            codes = [int(x) for x in op_codes]

        n_params = values.shape[-1]
        if op_param_counts is None:
            if len(codes) != int(n_params):
                raise ValueError(f"op_codes length must match values.shape[-1]: {len(codes)} != {int(n_params)}")
            param_counts = [1] * len(codes)
        else:
            param_counts = [int(x) for x in op_param_counts]
            expected = int(sum(param_counts))
            if expected != int(n_params):
                raise ValueError(f"Sum(op_param_counts) must match values.shape[-1]: {expected} != {int(n_params)}")
            if len(param_counts) != len(codes):
                raise ValueError(f"op_param_counts length must match op_codes: {len(param_counts)} != {len(codes)}")

        if freq_hz.ndim != 1:
            raise ValueError(f"freq_hz must be 1D (F,), got shape={tuple(freq_hz.shape)}")

        real_dtype = values.dtype
        complex_dtype = _complex_dtype_for(real_dtype)
        device = values.device

        omega = DifferentiablePhysicsKernel.omega(freq_hz.to(device=device, dtype=real_dtype))
        batch_shape = values.shape[:-1]
        omega_1d = omega
        omega = omega_1d.reshape((1,) * len(batch_shape) + (-1,))  # broadcast over batch
        shape = batch_shape + omega_1d.shape  # (..., F)

        if q_L is not None:
            ql = _as_tensor(q_L, device=device, dtype=real_dtype)
            if torch.any(ql <= 0):
                raise ValueError("q_L must be > 0 when provided")
            ql = ql.to(device=device, dtype=real_dtype)
        else:
            ql = None

        if q_C is not None:
            qc = _as_tensor(q_C, device=device, dtype=real_dtype)
            if torch.any(qc <= 0):
                raise ValueError("q_C must be > 0 when provided")
            qc = qc.to(device=device, dtype=real_dtype)
        else:
            qc = None

        ref_omega = None
        if q_model == "fixed_ref" and (ql is not None or qc is not None):
            if ref_freq_hz is None:
                raise ValueError("ref_freq_hz is required when q_model='fixed_ref' and Q is enabled.")
            ref_freq_t = _as_tensor(ref_freq_hz, device=device, dtype=real_dtype).reshape(-1)
            if ref_freq_t.numel() != 1:
                raise ValueError("ref_freq_hz must be a scalar when q_model='fixed_ref'.")
            ref_omega = DifferentiablePhysicsKernel.omega(ref_freq_t.reshape(()))

        A = torch.ones(shape, device=device, dtype=complex_dtype)
        B = torch.zeros(shape, device=device, dtype=complex_dtype)
        C = torch.zeros(shape, device=device, dtype=complex_dtype)
        D = torch.ones(shape, device=device, dtype=complex_dtype)

        param_offset = 0
        for idx, code in enumerate(codes):
            count = param_counts[idx]
            if count <= 0:
                raise ValueError(f"Invalid param count at op {idx}: {count}")
            v = values[..., param_offset : param_offset + count]
            param_offset += count
            if code == DifferentiablePhysicsKernel.OP_SERIES_L:
                Z = DifferentiablePhysicsKernel.series_impedance(
                    v,
                    omega,
                    kind="L",
                    eps=eps,
                    q=ql,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                B = A * Z + B
                D = C * Z + D
            elif code == DifferentiablePhysicsKernel.OP_SERIES_C:
                Z = DifferentiablePhysicsKernel.series_impedance(
                    v,
                    omega,
                    kind="C",
                    eps=eps,
                    q=qc,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                B = A * Z + B
                D = C * Z + D
            elif code == DifferentiablePhysicsKernel.OP_SHUNT_L:
                Y = DifferentiablePhysicsKernel.shunt_admittance(
                    v,
                    omega,
                    kind="L",
                    eps=eps,
                    q=ql,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                A = A + B * Y
                C = C + D * Y
            elif code == DifferentiablePhysicsKernel.OP_SHUNT_C:
                Y = DifferentiablePhysicsKernel.shunt_admittance(
                    v,
                    omega,
                    kind="C",
                    eps=eps,
                    q=qc,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                A = A + B * Y
                C = C + D * Y
            elif code == DifferentiablePhysicsKernel.OP_SHUNT_SERIES_LC:
                if v.shape[-1] != 2:
                    raise ValueError("OP_SHUNT_SERIES_LC expects 2 parameters (L, C).")
                L_val = v[..., 0].unsqueeze(-1)
                C_val = v[..., 1].unsqueeze(-1)
                Z_L = DifferentiablePhysicsKernel.series_impedance(
                    L_val,
                    omega,
                    kind="L",
                    eps=eps,
                    q=ql,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                Z_C = DifferentiablePhysicsKernel.series_impedance(
                    C_val,
                    omega,
                    kind="C",
                    eps=eps,
                    q=qc,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                Y = 1.0 / (Z_L + Z_C + eps)
                A = A + B * Y
                C = C + D * Y
            elif code == DifferentiablePhysicsKernel.OP_SERIES_PARALLEL_LC:
                if v.shape[-1] != 2:
                    raise ValueError("OP_SERIES_PARALLEL_LC expects 2 parameters (L, C).")
                L_val = v[..., 0].unsqueeze(-1)
                C_val = v[..., 1].unsqueeze(-1)
                Z_L = DifferentiablePhysicsKernel.series_impedance(
                    L_val,
                    omega,
                    kind="L",
                    eps=eps,
                    q=ql,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                Z_C = DifferentiablePhysicsKernel.series_impedance(
                    C_val,
                    omega,
                    kind="C",
                    eps=eps,
                    q=qc,
                    q_model=q_model,
                    ref_omega=ref_omega,
                )
                Y = 1.0 / (Z_L + eps) + 1.0 / (Z_C + eps)
                Z_eq = 1.0 / (Y + eps)
                B = A * Z_eq + B
                D = C * Z_eq + D
            else:
                raise ValueError(f"Unknown op code: {code}")

        return A, B, C, D

    @staticmethod
    def abcd_to_sparams(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        *,
        z0: float,
        eps: float = 1e-30,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert ABCD to complex S-parameters (equal reference z0).

        Returns (S11, S21, S12, S22), each shape (...,F) complex.
        """
        denom = A + B / z0 + C * z0 + D
        denom = denom + eps
        S11 = (A + B / z0 - C * z0 - D) / denom
        S21 = 2.0 / denom
        S12 = 2.0 * (A * D - B * C) / denom
        S22 = (-A + B / z0 - C * z0 + D) / denom
        return S11, S21, S12, S22

    @staticmethod
    def s21_db(S21: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
        mag = torch.abs(S21)
        return 20.0 * torch.log10(mag + eps)


class _BoundedPositiveReparam(nn.Module):
    def __init__(
        self,
        init_values: torch.Tensor,
        *,
        max_ratio: Optional[float],
        min_value: float,
    ) -> None:
        super().__init__()
        if init_values.ndim != 1:
            raise ValueError(f"init_values must be 1D, got shape={tuple(init_values.shape)}")
        self.register_buffer("init_values", init_values.detach().clone(), persistent=False)
        self.max_ratio = float(max_ratio) if max_ratio is not None else None
        self.min_value = float(min_value)
        self.raw = nn.Parameter(torch.zeros_like(init_values))

    def forward(self) -> torch.Tensor:
        base = self.init_values.clamp_min(self.min_value)
        if self.max_ratio is None:
            return base * torch.exp(self.raw)
        if self.max_ratio <= 1.0:
            return base
        scale = math.log(self.max_ratio)
        return base * torch.exp(scale * torch.tanh(self.raw))


class CascadedABCDCircuit(nn.Module):
    """
    A compiled cascaded circuit as a differentiable PyTorch computation graph.

    - Topology is fixed by `op_codes`.
    - Component values can be provided at forward-time (batched),
      or be made trainable via a positive reparameterization.
    """

    def __init__(
        self,
        op_codes: Sequence[int],
        init_values: torch.Tensor,
        *,
        z0: float = 50.0,
        q_L: float | None = 50.0,
        q_C: float | None = 50.0,
        trainable: bool = False,
        max_ratio: Optional[float] = 2.0,
        min_value: float = 1e-30,
        eps: float = 1e-30,
        op_param_counts: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if init_values.ndim != 1:
            raise ValueError(f"init_values must be 1D, got shape={tuple(init_values.shape)}")
        self._op_codes_list = [int(x) for x in op_codes]
        self._op_param_counts = [int(x) for x in op_param_counts] if op_param_counts is not None else None
        self.register_buffer("op_codes", torch.tensor(self._op_codes_list, dtype=torch.long), persistent=False)
        self.register_buffer("init_values", init_values.detach().clone(), persistent=False)
        self.z0 = float(z0)
        self.eps = float(eps)
        self.q_L = None if q_L is None else float(q_L)
        self.q_C = None if q_C is None else float(q_C)

        self._reparam: Optional[_BoundedPositiveReparam]
        if trainable:
            self._reparam = _BoundedPositiveReparam(self.init_values, max_ratio=max_ratio, min_value=min_value)
        else:
            self._reparam = None

    @property
    def n_components(self) -> int:
        return int(self.init_values.shape[0])

    def values(self) -> torch.Tensor:
        if self._reparam is None:
            return self.init_values
        return self._reparam()

    def abcd(
        self,
        freq_hz: torch.Tensor,
        *,
        values: Optional[torch.Tensor] = None,
        q_L: float | torch.Tensor | None = None,
        q_C: float | torch.Tensor | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v = values if values is not None else self.values()
        ql = self.q_L if q_L is None else q_L
        qc = self.q_C if q_C is None else q_C
        if v.ndim == 1:
            v_in = v.unsqueeze(0)  # (1,N) to reuse batch logic
            A, B, C, D = DifferentiablePhysicsKernel.cascade_abcd(
                self._op_codes_list,
                v_in,
                freq_hz,
                op_param_counts=self._op_param_counts,
                q_L=ql,
                q_C=qc,
                q_model=q_model,
                ref_freq_hz=ref_freq_hz,
                eps=self.eps,
            )
            return A[0], B[0], C[0], D[0]
        return DifferentiablePhysicsKernel.cascade_abcd(
            self._op_codes_list,
            v,
            freq_hz,
            op_param_counts=self._op_param_counts,
            q_L=ql,
            q_C=qc,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
            eps=self.eps,
        )

    def sparams(
        self,
        freq_hz: torch.Tensor,
        *,
        values: Optional[torch.Tensor] = None,
        q_L: float | torch.Tensor | None = None,
        q_C: float | torch.Tensor | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A, B, C, D = self.abcd(
            freq_hz,
            values=values,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
        )
        return DifferentiablePhysicsKernel.abcd_to_sparams(A, B, C, D, z0=self.z0, eps=self.eps)

    def forward(
        self,
        freq_hz: torch.Tensor,
        *,
        values: Optional[torch.Tensor] = None,
        q_L: float | torch.Tensor | None = None,
        q_C: float | torch.Tensor | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | torch.Tensor | None = None,
        output: Literal["s21_db", "s21_mag", "sparams"] = "s21_db",
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        S11, S21, S12, S22 = self.sparams(
            freq_hz,
            values=values,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
        )
        if output == "sparams":
            return S11, S21, S12, S22
        if output == "s21_mag":
            return torch.abs(S21)
        if output == "s21_db":
            return DifferentiablePhysicsKernel.s21_db(S21)
        raise ValueError(f"Unknown output: {output}")


class DynamicCircuitAssembler:
    """
    Compile a DSL-parsed ComponentSpec sequence into a differentiable circuit module.

    The returned module follows the "nn.Sequential" spirit: fixed topology,
    forward pass is a chain of differentiable physics operators.
    """

    def __init__(self, *, z0: float = 50.0) -> None:
        self.z0 = float(z0)

    @staticmethod
    def _get_attr(obj, name: str):
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    @staticmethod
    def _to_component_spec(obj) -> ComponentSpec:
        if isinstance(obj, ComponentSpec):
            return obj
        if isinstance(obj, dict):
            return ComponentSpec(
                ctype=str(obj["ctype"]),
                role=str(obj["role"]),
                value_si=float(obj["value_si"]),
                std_label=obj.get("std_label"),
                node1=str(obj["node1"]),
                node2=str(obj["node2"]),
            )
        return ComponentSpec(
            ctype=str(DynamicCircuitAssembler._get_attr(obj, "ctype")),
            role=str(DynamicCircuitAssembler._get_attr(obj, "role")),
            value_si=float(DynamicCircuitAssembler._get_attr(obj, "value_si")),
            std_label=getattr(obj, "std_label", None),
            node1=str(DynamicCircuitAssembler._get_attr(obj, "node1")),
            node2=str(DynamicCircuitAssembler._get_attr(obj, "node2")),
        )

    @staticmethod
    def _op_code_for(comp: ComponentSpec) -> int:
        if comp.ctype == "L" and comp.role == "series":
            return DifferentiablePhysicsKernel.OP_SERIES_L
        if comp.ctype == "C" and comp.role == "series":
            return DifferentiablePhysicsKernel.OP_SERIES_C
        if comp.ctype == "L" and comp.role == "shunt":
            return DifferentiablePhysicsKernel.OP_SHUNT_L
        if comp.ctype == "C" and comp.role == "shunt":
            return DifferentiablePhysicsKernel.OP_SHUNT_C
        raise ValueError(f"Unsupported component: ctype={comp.ctype} role={comp.role}")

    @staticmethod
    def _other_node(comp: ComponentSpec, node: str) -> str:
        return comp.node2 if comp.node1 == node else comp.node1

    def _compile_ladder_ops(
        self,
        comps: List[ComponentSpec],
        *,
        gnd: str = "gnd",
        port_in: str = "in",
        port_out: str = "out",
    ) -> tuple[list[int], list[int], list[int]] | None:
        node_to_comps: dict[str, list[int]] = {}
        for idx, c in enumerate(comps):
            for node in (c.node1, c.node2):
                if node == gnd:
                    continue
                node_to_comps.setdefault(node, []).append(idx)

        branch_comp_ids: set[int] = set()
        branches_by_anchor: dict[str, list[tuple[int, int]]] = {}
        for node, idxs in node_to_comps.items():
            if node in (port_in, port_out):
                continue
            if len(idxs) != 2:
                continue
            i1, i2 = idxs
            c1, c2 = comps[i1], comps[i2]
            if c1.role != "series" or c2.role != "series":
                continue
            o1 = self._other_node(c1, node)
            o2 = self._other_node(c2, node)
            c1_to_gnd = o1 == gnd
            c2_to_gnd = o2 == gnd
            if c1_to_gnd == c2_to_gnd:
                continue
            comp_to_gnd = c1 if c1_to_gnd else c2
            comp_to_anchor = c2 if c1_to_gnd else c1
            anchor = self._other_node(comp_to_anchor, node)
            if anchor == gnd:
                continue
            types = {comp_to_gnd.ctype, comp_to_anchor.ctype}
            if types != {"L", "C"}:
                continue
            L_idx = i1 if comps[i1].ctype == "L" else i2
            C_idx = i1 if comps[i1].ctype == "C" else i2
            branches_by_anchor.setdefault(anchor, []).append((L_idx, C_idx))
            branch_comp_ids.update([i1, i2])

        # Detect series-parallel LC (two series components between same nodes).
        series_pairs: dict[tuple[str, str], list[int]] = {}
        for idx, c in enumerate(comps):
            if idx in branch_comp_ids:
                continue
            if c.role != "series":
                continue
            if c.node1 == gnd or c.node2 == gnd:
                continue
            key = tuple(sorted((c.node1, c.node2)))
            series_pairs.setdefault(key, []).append(idx)

        parallel_series: dict[tuple[str, str], tuple[int, int]] = {}
        parallel_ids: set[int] = set()
        for key, idxs in series_pairs.items():
            if len(idxs) != 2:
                continue
            types = {comps[i].ctype for i in idxs}
            if types != {"L", "C"}:
                continue
            L_idx = idxs[0] if comps[idxs[0]].ctype == "L" else idxs[1]
            C_idx = idxs[0] if comps[idxs[0]].ctype == "C" else idxs[1]
            parallel_series[key] = (L_idx, C_idx)
            parallel_ids.update(idxs)

        # Build series adjacency (excluding branch + folded parallel components).
        series_edges: list[tuple[str, str, str, list[int]]] = []
        series_adj: dict[str, list[tuple[str, int]]] = {}

        def _add_edge(n1: str, n2: str, kind: str, comp_indices: list[int]) -> None:
            edge_id = len(series_edges)
            series_edges.append((n1, n2, kind, comp_indices))
            series_adj.setdefault(n1, []).append((n2, edge_id))
            series_adj.setdefault(n2, []).append((n1, edge_id))

        for (n1, n2), (l_idx, c_idx) in parallel_series.items():
            _add_edge(n1, n2, "parallel_lc", [l_idx, c_idx])

        for idx, c in enumerate(comps):
            if idx in branch_comp_ids or idx in parallel_ids:
                continue
            if c.role != "series":
                continue
            if c.node1 == gnd or c.node2 == gnd:
                continue
            _add_edge(c.node1, c.node2, "single", [idx])

        if port_in not in series_adj or port_out not in series_adj:
            return None

        path_nodes: list[str] = [port_in]
        path_series: list[int] = []
        visited_edges: set[int] = set()
        prev = None
        current = port_in
        while current != port_out:
            neighbors = [item for item in series_adj.get(current, []) if item[0] != prev]
            if len(neighbors) != 1:
                return None
            next_node, edge_id = neighbors[0]
            if edge_id in visited_edges:
                return None
            visited_edges.add(edge_id)
            path_series.append(edge_id)
            path_nodes.append(next_node)
            prev, current = current, next_node

        # Collect shunt components per node (excluding branch components).
        shunts_by_node: dict[str, list[int]] = {}
        for idx, c in enumerate(comps):
            if idx in branch_comp_ids:
                continue
            if c.role != "shunt":
                continue
            if c.node1 == gnd and c.node2 != gnd:
                anchor = c.node2
            elif c.node2 == gnd and c.node1 != gnd:
                anchor = c.node1
            else:
                anchor = c.node1
            shunts_by_node.setdefault(anchor, []).append(idx)

        op_codes: list[int] = []
        op_param_counts: list[int] = []
        value_comp_indices: list[int] = []

        for i, node in enumerate(path_nodes):
            for idx in shunts_by_node.get(node, []):
                op_codes.append(self._op_code_for(comps[idx]))
                op_param_counts.append(1)
                value_comp_indices.append(idx)
            for l_idx, c_idx in branches_by_anchor.get(node, []):
                op_codes.append(DifferentiablePhysicsKernel.OP_SHUNT_SERIES_LC)
                op_param_counts.append(2)
                value_comp_indices.extend([l_idx, c_idx])
            if i < len(path_series):
                edge_id = path_series[i]
                _, _, kind, comp_indices = series_edges[edge_id]
                if kind == "parallel_lc":
                    op_codes.append(DifferentiablePhysicsKernel.OP_SERIES_PARALLEL_LC)
                    op_param_counts.append(2)
                    value_comp_indices.extend(comp_indices)
                else:
                    s_idx = comp_indices[0]
                    op_codes.append(self._op_code_for(comps[s_idx]))
                    op_param_counts.append(1)
                    value_comp_indices.append(s_idx)

        if not op_codes:
            return None
        return op_codes, op_param_counts, value_comp_indices

    def assemble(
        self,
        components: Sequence[ComponentSpec] | Sequence[object],
        *,
        trainable: bool = False,
        max_ratio: Optional[float] = 2.0,
        min_value: float = 1e-30,
        q_L: float | None = 50.0,
        q_C: float | None = 50.0,
        eps: float = 1e-30,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[CascadedABCDCircuit, List[ComponentSpec]]:
        comps = [self._to_component_spec(c) for c in components]
        compiled = self._compile_ladder_ops(comps)
        if compiled is None:
            op_codes = [self._op_code_for(c) for c in comps]
            op_param_counts = None
            value_comp_indices = list(range(len(comps)))
        else:
            op_codes, op_param_counts, value_comp_indices = compiled
        init_values = _as_tensor(
            [float(comps[i].value_si) for i in value_comp_indices],
            device=torch.device(device),
            dtype=dtype,
        ).clamp_min(min_value)
        mod = CascadedABCDCircuit(
            op_codes,
            init_values,
            z0=self.z0,
            q_L=q_L,
            q_C=q_C,
            trainable=trainable,
            max_ratio=max_ratio,
            min_value=min_value,
            eps=eps,
            op_param_counts=op_param_counts,
        ).to(device=torch.device(device), dtype=dtype)
        mod.value_comp_indices = value_comp_indices
        return mod, comps


@dataclass(frozen=True)
class RefinementResult:
    refined_components: List[ComponentSpec]
    loss_history: List[float]
    initial_loss: float
    final_loss: float
    continuous_components: Optional[List[ComponentSpec]] = None
    snapped_components: Optional[List[ComponentSpec]] = None
    snapped_loss: Optional[float] = None
    final_s21_db: Optional[torch.Tensor] = None
    snapped_s21_db: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MacroBankEntry:
    op_codes: List[int]
    op_param_counts: Optional[List[int]]
    slot_idx: torch.Tensor


def _cascade_abcd_cells(
    A_cells: torch.Tensor,
    B_cells: torch.Tensor,
    C_cells: torch.Tensor,
    D_cells: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if A_cells.ndim != 2:
        raise ValueError(f"Expected A_cells shape (K, F), got {tuple(A_cells.shape)}")
    A = torch.ones_like(A_cells[0])
    B = torch.zeros_like(B_cells[0])
    C = torch.zeros_like(C_cells[0])
    D = torch.ones_like(D_cells[0])
    for k in range(A_cells.shape[0]):
        A_k = A_cells[k]
        B_k = B_cells[k]
        C_k = C_cells[k]
        D_k = D_cells[k]
        A_new = A * A_k + B * C_k
        B_new = A * B_k + B * D_k
        C_new = C * A_k + D * C_k
        D_new = C * B_k + D * D_k
        A, B, C, D = A_new, B_new, C_new, D_new
    return A, B, C, D


def mixed_s21_db(
    slot_raw: torch.Tensor,
    g_soft: torch.Tensor,
    macro_bank: Sequence[MacroBankEntry],
    freq_hz: torch.Tensor,
    *,
    z0: float = 50.0,
    q_L: float | torch.Tensor | None = 50.0,
    q_C: float | torch.Tensor | None = 50.0,
    q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
    ref_freq_hz: float | torch.Tensor | None = None,
    eps: float = 1e-30,
    raw_min: float | None = None,
    raw_max: float | None = None,
) -> torch.Tensor:
    if slot_raw.ndim != 2:
        raise ValueError(f"slot_raw must have shape (K, S), got {tuple(slot_raw.shape)}")
    if g_soft.ndim != 2:
        raise ValueError(f"g_soft must have shape (K, M+1), got {tuple(g_soft.shape)}")
    if freq_hz.ndim != 1:
        raise ValueError(f"freq_hz must be 1D, got {tuple(freq_hz.shape)}")
    macro_count = len(macro_bank)
    if g_soft.shape[1] != macro_count + 1:
        raise ValueError(f"g_soft last dim must be {macro_count + 1}, got {g_soft.shape[1]}")

    raw = slot_raw
    if raw_min is not None or raw_max is not None:
        min_val = float(raw_min) if raw_min is not None else None
        max_val = float(raw_max) if raw_max is not None else None
        raw = raw.clamp(min=min_val, max=max_val)
    values = torch.exp(raw) + float(eps)
    device = values.device
    dtype = values.dtype
    complex_dtype = _complex_dtype_for(dtype)

    K = values.shape[0]
    F = freq_hz.shape[0]
    A_mix = torch.zeros((K, F), device=device, dtype=complex_dtype)
    B_mix = torch.zeros((K, F), device=device, dtype=complex_dtype)
    C_mix = torch.zeros((K, F), device=device, dtype=complex_dtype)
    D_mix = torch.zeros((K, F), device=device, dtype=complex_dtype)

    for m, entry in enumerate(macro_bank):
        idx = entry.slot_idx.to(device=device)
        vals_m = values.index_select(dim=-1, index=idx)
        A_m, B_m, C_m, D_m = DifferentiablePhysicsKernel.cascade_abcd(
            entry.op_codes,
            vals_m,
            freq_hz,
            op_param_counts=entry.op_param_counts,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
            eps=eps,
        )
        weight = g_soft[:, m].to(device=device, dtype=dtype).unsqueeze(-1).to(complex_dtype)
        A_mix = A_mix + weight * A_m
        B_mix = B_mix + weight * B_m
        C_mix = C_mix + weight * C_m
        D_mix = D_mix + weight * D_m

    p_skip = g_soft[:, macro_count].to(device=device, dtype=dtype).unsqueeze(-1).to(complex_dtype)
    A_mix = A_mix + p_skip
    D_mix = D_mix + p_skip

    A_tot, B_tot, C_tot, D_tot = _cascade_abcd_cells(A_mix, B_mix, C_mix, D_mix)
    _, S21, _, _ = DifferentiablePhysicsKernel.abcd_to_sparams(A_tot, B_tot, C_tot, D_tot, z0=z0, eps=eps)
    return DifferentiablePhysicsKernel.s21_db(S21)


class InferenceTimeOptimizer:
    """
    Inference-time gradient-based refinement (topology frozen, values optimized).
    """

    def __init__(self, *, z0: float = 50.0) -> None:
        self.assembler = DynamicCircuitAssembler(z0=z0)

    def refine(
        self,
        components: Sequence[ComponentSpec] | Sequence[object],
        *,
        freq_hz: torch.Tensor | Sequence[float],
        target_s21_db: torch.Tensor | Sequence[float] | None = None,
        steps: int = 50,
        lr: float = 5e-2,
        optimizer: Literal["adam", "sgd"] = "adam",
        q_L: float | None = 50.0,
        q_C: float | None = 50.0,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | None = None,
        max_ratio: Optional[float] = 2.0,
        loss_kind: Literal["spec_hinge", "mse_db", "mae_db"] = "spec_hinge",
        mask_min_db: torch.Tensor | Sequence[float] | None = None,
        mask_max_db: torch.Tensor | Sequence[float] | None = None,
        passband_min_db: float = -3.0,
        stopband_max_db: float = -40.0,
        auto_passband_margin_db: float = 1.0,
        auto_stopband_margin_db: float = 5.0,
        warn_on_empty_mask: bool = True,
        guide_weight: float = 1e-2,
        hinge_power: float = 2.0,
        passband_ripple_max_db: float | None = None,
        ripple_weight: float = 0.0,
        ripple_smooth_tau: float = 0.5,
        snap_series: str | None = "E24",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> RefinementResult:
        device_t = torch.device(device)
        freq = _as_tensor(freq_hz, device=device_t, dtype=dtype)
        target = _as_tensor(target_s21_db, device=device_t, dtype=dtype) if target_s21_db is not None else None
        if target is not None:
            target = target.reshape(-1) if target.ndim != 1 else target
        freq = freq.reshape(-1) if freq.ndim != 1 else freq
        if freq.ndim != 1 or (target is not None and target.ndim != 1):
            raise ValueError(
                f"freq_hz and target_s21_db must be 1D, got shapes {tuple(freq.shape)} and {tuple(target.shape) if target is not None else None}"
            )
        if target is not None and int(freq.shape[0]) != int(target.shape[0]):
            raise ValueError(f"freq_hz and target_s21_db length mismatch: {int(freq.shape[0])} != {int(target.shape[0])}")
        if q_model not in ("freq_dependent", "fixed_ref"):
            raise ValueError(f"Unknown q_model: {q_model}")
        if q_model == "fixed_ref" and (q_L is not None or q_C is not None) and ref_freq_hz is None:
            f_min = float(torch.min(freq).item())
            f_max = float(torch.max(freq).item())
            ref_freq_hz = float(math.sqrt(f_min * f_max))

        circuit, comps = self.assembler.assemble(
            components,
            trainable=True,
            q_L=q_L,
            q_C=q_C,
            max_ratio=max_ratio,
            device=device_t,
            dtype=dtype,
        )
        value_comp_indices = getattr(circuit, "value_comp_indices", None)

        params = [p for p in circuit.parameters() if p.requires_grad]
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr=float(lr))
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=float(lr), momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        loss_history: List[float] = []

        if mask_min_db is not None or mask_max_db is not None:
            if mask_min_db is None or mask_max_db is None:
                raise ValueError("mask_min_db and mask_max_db must be provided together.")
            mask_min = _as_tensor(mask_min_db, device=device_t, dtype=dtype).reshape(-1)
            mask_max = _as_tensor(mask_max_db, device=device_t, dtype=dtype).reshape(-1)
            if mask_min.shape != freq.shape or mask_max.shape != freq.shape:
                raise ValueError(
                    f"mask_min_db/mask_max_db must match freq_hz shape {tuple(freq.shape)}, got {tuple(mask_min.shape)} and {tuple(mask_max.shape)}"
                )
            min_mask = torch.isfinite(mask_min)
            max_mask = torch.isfinite(mask_max)
            passband_mask = min_mask  # used for ripple by default
            stopband_mask = max_mask
        else:
            if target is None:
                raise ValueError("target_s21_db is required unless explicit mask_min_db/mask_max_db are provided.")
            finite = torch.isfinite(target)
            if not bool(finite.any()):
                raise ValueError("target_s21_db contains no finite values.")
            passband_mask = target >= float(passband_min_db)
            stopband_mask = target <= float(stopband_max_db)
            mask_min = None
            mask_max = None
            min_mask = None
            max_mask = None

            if not bool(passband_mask.any()):
                max_db = torch.max(target[finite])
                fallback_min = max_db - float(auto_passband_margin_db)
                passband_mask = target >= fallback_min
                if warn_on_empty_mask:
                    warnings.warn(
                        f"Passband mask empty for passband_min_db={passband_min_db:.3g} dB. "
                        f"Falling back to relative threshold {fallback_min:.3g} dB.",
                        RuntimeWarning,
                    )

            if not bool(stopband_mask.any()):
                min_db = torch.min(target[finite])
                fallback_max = min_db + float(auto_stopband_margin_db)
                stopband_mask = target <= fallback_max

        def _hinge_mask_loss(pred_db: torch.Tensor) -> torch.Tensor:
            loss = pred_db.new_zeros(())
            if mask_min is not None and min_mask is not None:
                if bool(min_mask.any()):
                    viol = torch.relu(mask_min - pred_db)
                    loss = loss + torch.mean(viol[min_mask] ** float(hinge_power))
            else:
                if bool(passband_mask.any()):
                    viol = torch.relu(float(passband_min_db) - pred_db)
                    loss = loss + torch.mean(viol[passband_mask] ** float(hinge_power))

            if mask_max is not None and max_mask is not None:
                if bool(max_mask.any()):
                    viol = torch.relu(pred_db - mask_max)
                    loss = loss + torch.mean(viol[max_mask] ** float(hinge_power))
            else:
                if bool(stopband_mask.any()):
                    viol = torch.relu(pred_db - float(stopband_max_db))
                    loss = loss + torch.mean(viol[stopband_mask] ** float(hinge_power))

            if passband_ripple_max_db is not None and float(ripple_weight) > 0 and bool(passband_mask.any()):
                pb = pred_db[passband_mask]
                tau = float(ripple_smooth_tau)
                smooth_max = tau * torch.logsumexp(pb / tau, dim=0)
                smooth_min = -tau * torch.logsumexp(-pb / tau, dim=0)
                ripple = smooth_max - smooth_min
                ripple_viol = torch.relu(ripple - float(passband_ripple_max_db))
                loss = loss + float(ripple_weight) * (ripple_viol**2)
            return loss

        def _loss(pred_db: torch.Tensor) -> torch.Tensor:
            if loss_kind == "spec_hinge":
                hinge = _hinge_mask_loss(pred_db)
                if target is not None and float(guide_weight) > 0:
                    guide = torch.mean((pred_db - target) ** 2)
                    return hinge + float(guide_weight) * guide
                return hinge
            if loss_kind == "mse_db":
                if target is None:
                    raise ValueError("target_s21_db is required for mse_db loss_kind.")
                return torch.mean((pred_db - target) ** 2)
            if loss_kind == "mae_db":
                if target is None:
                    raise ValueError("target_s21_db is required for mae_db loss_kind.")
                return torch.mean(torch.abs(pred_db - target))
            raise ValueError(f"Unknown loss_kind: {loss_kind}")

        with torch.no_grad():
            init_pred = circuit(freq, output="s21_db", q_model=q_model, ref_freq_hz=ref_freq_hz)
            init_loss = float(_loss(init_pred).item())

        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            pred = circuit(freq, output="s21_db", q_model=q_model, ref_freq_hz=ref_freq_hz)
            loss = _loss(pred)
            loss.backward()
            opt.step()
            loss_history.append(float(loss.detach().cpu().item()))

        with torch.no_grad():
            final_pred = circuit(freq, output="s21_db", q_model=q_model, ref_freq_hz=ref_freq_hz)
            final_loss = float(_loss(final_pred).item())
            refined_values = circuit.values().detach().cpu().tolist()

        values_by_comp = [None] * len(comps)
        if value_comp_indices is None:
            for idx, v in enumerate(refined_values):
                if idx < len(values_by_comp):
                    values_by_comp[idx] = float(v)
        else:
            for v, comp_idx in zip(refined_values, value_comp_indices):
                values_by_comp[int(comp_idx)] = float(v)
        continuous_components: List[ComponentSpec] = []
        for idx, c in enumerate(comps):
            v = values_by_comp[idx] if values_by_comp[idx] is not None else float(c.value_si)
            continuous_components.append(
                ComponentSpec(
                    ctype=c.ctype,
                    role=c.role,
                    value_si=float(v),
                    std_label=c.std_label,
                    node1=c.node1,
                    node2=c.node2,
                )
            )

        refined_components: List[ComponentSpec] = list(continuous_components)
        snapped_components: Optional[List[ComponentSpec]] = None
        snapped_loss: Optional[float] = None
        snapped_s21_db: Optional[torch.Tensor] = None
        if snap_series is not None:
            from src.data.quantization import quantize_components

            snapped_components = quantize_components(continuous_components, series=str(snap_series))
            if value_comp_indices is None:
                snapped_values = _as_tensor([c.value_si for c in snapped_components], device=device_t, dtype=dtype)
            else:
                snapped_values = _as_tensor(
                    [snapped_components[int(i)].value_si for i in value_comp_indices],
                    device=device_t,
                    dtype=dtype,
                )
            with torch.no_grad():
                snapped_pred = circuit(freq, values=snapped_values, output="s21_db", q_model=q_model, ref_freq_hz=ref_freq_hz)
                snapped_s21_db = snapped_pred.detach().cpu()
                snapped_loss = float(_loss(snapped_pred).item())
            refined_components = snapped_components

        return RefinementResult(
            refined_components=refined_components,
            loss_history=loss_history,
            initial_loss=init_loss,
            final_loss=final_loss,
            continuous_components=continuous_components,
            snapped_components=snapped_components,
            snapped_loss=snapped_loss,
            final_s21_db=final_pred.detach().cpu(),
            snapped_s21_db=snapped_s21_db,
        )


def unroll_refine_slots(
    slot_raw: torch.Tensor,
    slot_mask: torch.Tensor,
    slot_indices: torch.Tensor,
    circuit: CascadedABCDCircuit,
    freq_hz: torch.Tensor,
    target_s21_db: torch.Tensor,
    *,
    steps: int = 5,
    lr: float = 5e-2,
    eps: float = 1e-30,
    max_step: float | None = 0.5,
    raw_min: float = -32.0,
    raw_max: float = -12.0,
    nan_backoff: float = 0.5,
    max_backoff: int = 3,
    create_graph: bool = True,
    return_raw: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable unrolled refinement over per-cell slot values with stability guards.
    """
    raw = slot_raw
    mask = slot_mask
    raw_flat = raw.reshape(-1)
    mask_flat = mask.reshape(-1)
    idx = slot_indices.to(device=raw.device, dtype=torch.long)

    loss = None
    base_lr = float(lr)
    backoff_tries = max(0, int(max_backoff))
    max_step_val = float(max_step) if max_step is not None else None
    for _ in range(int(steps)):
        raw_flat = raw_flat.clamp(min=float(raw_min), max=float(raw_max))
        step_lr = base_lr
        updated = False
        for _ in range(backoff_tries + 1):
            values_flat = torch.exp(raw_flat) * mask_flat + float(eps)
            values_vec = values_flat.index_select(0, idx)
            pred = circuit(freq_hz, values=values_vec, output="s21_db")
            loss = F.mse_loss(pred, target_s21_db)
            if not torch.isfinite(loss):
                step_lr *= float(nan_backoff)
                continue
            grad = torch.autograd.grad(loss, raw_flat, create_graph=bool(create_graph))[0]
            if not torch.isfinite(grad).all():
                step_lr *= float(nan_backoff)
                continue
            step_delta = step_lr * grad
            if max_step_val is not None and max_step_val > 0.0:
                step_delta = step_delta.clamp(-max_step_val, max_step_val)
            raw_next = raw_flat - step_delta * mask_flat
            if not torch.isfinite(raw_next).all():
                step_lr *= float(nan_backoff)
                continue
            raw_flat = raw_next
            updated = True
            break
        if not updated:
            if loss is not None and torch.isfinite(loss):
                if return_raw:
                    return loss, raw_flat.view_as(raw)
                return loss
            if return_raw:
                return torch.zeros((), device=raw.device, dtype=raw.dtype), raw_flat.view_as(raw)
            return torch.zeros((), device=raw.device, dtype=raw.dtype)

    if loss is None:
        raise ValueError("unroll_refine_slots requires steps >= 1.")
    if return_raw:
        return loss, raw_flat.view_as(raw)
    return loss


def unroll_refine_slots_mixed(
    slot_raw: torch.Tensor,
    g_soft: torch.Tensor,
    macro_bank: Sequence[MacroBankEntry],
    freq_hz: torch.Tensor,
    target_s21_db: torch.Tensor,
    *,
    steps: int = 5,
    lr: float = 5e-2,
    eps: float = 1e-30,
    max_step: float | None = 0.5,
    raw_min: float = -32.0,
    raw_max: float = -12.0,
    nan_backoff: float = 0.5,
    max_backoff: int = 3,
    create_graph: bool = True,
    return_raw: bool = False,
    z0: float = 50.0,
    q_L: float | torch.Tensor | None = 50.0,
    q_C: float | torch.Tensor | None = 50.0,
    q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
    ref_freq_hz: float | torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Unrolled refinement where the forward pass uses mixed macro ABCD matrices.
    """
    raw = slot_raw
    loss = None
    base_lr = float(lr)
    backoff_tries = max(0, int(max_backoff))
    max_step_val = float(max_step) if max_step is not None else None
    for _ in range(int(steps)):
        raw = raw.clamp(min=float(raw_min), max=float(raw_max))
        step_lr = base_lr
        updated = False
        for _ in range(backoff_tries + 1):
            pred = mixed_s21_db(
                raw,
                g_soft,
                macro_bank,
                freq_hz,
                z0=z0,
                q_L=q_L,
                q_C=q_C,
                q_model=q_model,
                ref_freq_hz=ref_freq_hz,
                eps=eps,
                raw_min=None,
                raw_max=None,
            )
            loss = F.mse_loss(pred, target_s21_db)
            if not torch.isfinite(loss):
                step_lr *= float(nan_backoff)
                continue
            grad = torch.autograd.grad(loss, raw, create_graph=bool(create_graph))[0]
            if not torch.isfinite(grad).all():
                step_lr *= float(nan_backoff)
                continue
            step_delta = step_lr * grad
            if max_step_val is not None and max_step_val > 0.0:
                step_delta = step_delta.clamp(-max_step_val, max_step_val)
            raw_next = raw - step_delta
            if not torch.isfinite(raw_next).all():
                step_lr *= float(nan_backoff)
                continue
            raw = raw_next
            updated = True
            break
        if not updated:
            if loss is not None and torch.isfinite(loss):
                if return_raw:
                    return loss, raw
                return loss
            if return_raw:
                return torch.zeros((), device=slot_raw.device, dtype=slot_raw.dtype), raw
            return torch.zeros((), device=slot_raw.device, dtype=slot_raw.dtype)

    if loss is None:
        raise ValueError("unroll_refine_slots_mixed requires steps >= 1.")
    if return_raw:
        return loss, raw
    return loss
