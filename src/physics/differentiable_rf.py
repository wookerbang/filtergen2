from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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

    @staticmethod
    def omega(freq_hz: torch.Tensor) -> torch.Tensor:
        return 2.0 * math.pi * freq_hz

    @staticmethod
    def _jw(omega: torch.Tensor) -> torch.Tensor:
        return 1j * omega

    @staticmethod
    def series_impedance(L_or_C: torch.Tensor, omega: torch.Tensor, *, kind: Literal["L", "C"], eps: float) -> torch.Tensor:
        jw = DifferentiablePhysicsKernel._jw(omega)
        if kind == "L":
            return jw * L_or_C
        if kind == "C":
            return 1.0 / (jw * L_or_C + eps)
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def shunt_admittance(L_or_C: torch.Tensor, omega: torch.Tensor, *, kind: Literal["L", "C"], eps: float) -> torch.Tensor:
        jw = DifferentiablePhysicsKernel._jw(omega)
        if kind == "C":
            return jw * L_or_C
        if kind == "L":
            return 1.0 / (jw * L_or_C + eps)
        raise ValueError(f"Unknown kind: {kind}")

    @staticmethod
    def cascade_abcd(
        op_codes: Sequence[int] | torch.Tensor,
        values: torch.Tensor,
        freq_hz: torch.Tensor,
        *,
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

        if isinstance(op_codes, torch.Tensor):
            codes = [int(x) for x in op_codes.detach().cpu().tolist()]
        else:
            codes = [int(x) for x in op_codes]

        n_components = values.shape[-1]
        if len(codes) != int(n_components):
            raise ValueError(f"op_codes length must match values.shape[-1]: {len(codes)} != {int(n_components)}")

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

        A = torch.ones(shape, device=device, dtype=complex_dtype)
        B = torch.zeros(shape, device=device, dtype=complex_dtype)
        C = torch.zeros(shape, device=device, dtype=complex_dtype)
        D = torch.ones(shape, device=device, dtype=complex_dtype)

        for idx, code in enumerate(codes):
            v = values[..., idx].unsqueeze(-1)
            if code == DifferentiablePhysicsKernel.OP_SERIES_L:
                Z = DifferentiablePhysicsKernel.series_impedance(v, omega, kind="L", eps=eps)
                B = A * Z + B
                D = C * Z + D
            elif code == DifferentiablePhysicsKernel.OP_SERIES_C:
                Z = DifferentiablePhysicsKernel.series_impedance(v, omega, kind="C", eps=eps)
                B = A * Z + B
                D = C * Z + D
            elif code == DifferentiablePhysicsKernel.OP_SHUNT_L:
                Y = DifferentiablePhysicsKernel.shunt_admittance(v, omega, kind="L", eps=eps)
                A = A + B * Y
                C = C + D * Y
            elif code == DifferentiablePhysicsKernel.OP_SHUNT_C:
                Y = DifferentiablePhysicsKernel.shunt_admittance(v, omega, kind="C", eps=eps)
                A = A + B * Y
                C = C + D * Y
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
        trainable: bool = False,
        max_ratio: Optional[float] = 2.0,
        min_value: float = 1e-30,
        eps: float = 1e-30,
    ) -> None:
        super().__init__()
        if init_values.ndim != 1:
            raise ValueError(f"init_values must be 1D, got shape={tuple(init_values.shape)}")
        self._op_codes_list = [int(x) for x in op_codes]
        self.register_buffer("op_codes", torch.tensor(self._op_codes_list, dtype=torch.long), persistent=False)
        self.register_buffer("init_values", init_values.detach().clone(), persistent=False)
        self.z0 = float(z0)
        self.eps = float(eps)

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

    def abcd(self, freq_hz: torch.Tensor, *, values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v = values if values is not None else self.values()
        if v.ndim == 1:
            v_in = v.unsqueeze(0)  # (1,N) to reuse batch logic
            A, B, C, D = DifferentiablePhysicsKernel.cascade_abcd(self._op_codes_list, v_in, freq_hz, eps=self.eps)
            return A[0], B[0], C[0], D[0]
        return DifferentiablePhysicsKernel.cascade_abcd(self._op_codes_list, v, freq_hz, eps=self.eps)

    def sparams(self, freq_hz: torch.Tensor, *, values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A, B, C, D = self.abcd(freq_hz, values=values)
        return DifferentiablePhysicsKernel.abcd_to_sparams(A, B, C, D, z0=self.z0, eps=self.eps)

    def forward(
        self,
        freq_hz: torch.Tensor,
        *,
        values: Optional[torch.Tensor] = None,
        output: Literal["s21_db", "s21_mag", "sparams"] = "s21_db",
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        S11, S21, S12, S22 = self.sparams(freq_hz, values=values)
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

    def assemble(
        self,
        components: Sequence[ComponentSpec] | Sequence[object],
        *,
        trainable: bool = False,
        max_ratio: Optional[float] = 2.0,
        min_value: float = 1e-30,
        eps: float = 1e-30,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[CascadedABCDCircuit, List[ComponentSpec]]:
        comps = [self._to_component_spec(c) for c in components]
        op_codes = [self._op_code_for(c) for c in comps]
        init_values = _as_tensor([float(c.value_si) for c in comps], device=torch.device(device), dtype=dtype).clamp_min(min_value)
        mod = CascadedABCDCircuit(
            op_codes,
            init_values,
            z0=self.z0,
            trainable=trainable,
            max_ratio=max_ratio,
            min_value=min_value,
            eps=eps,
        ).to(device=torch.device(device), dtype=dtype)
        return mod, comps


@dataclass(frozen=True)
class RefinementResult:
    refined_components: List[ComponentSpec]
    loss_history: List[float]
    initial_loss: float
    final_loss: float


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
        target_s21_db: torch.Tensor | Sequence[float],
        steps: int = 50,
        lr: float = 5e-2,
        optimizer: Literal["adam", "sgd"] = "adam",
        max_ratio: Optional[float] = 2.0,
        loss_kind: Literal["mse_db", "mae_db"] = "mse_db",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> RefinementResult:
        device_t = torch.device(device)
        freq = _as_tensor(freq_hz, device=device_t, dtype=dtype)
        target = _as_tensor(target_s21_db, device=device_t, dtype=dtype)
        if freq.ndim != 1 or target.ndim != 1:
            raise ValueError(f"freq_hz and target_s21_db must be 1D, got shapes {tuple(freq.shape)} and {tuple(target.shape)}")
        if int(freq.shape[0]) != int(target.shape[0]):
            raise ValueError(f"freq_hz and target_s21_db length mismatch: {int(freq.shape[0])} != {int(target.shape[0])}")

        circuit, comps = self.assembler.assemble(
            components,
            trainable=True,
            max_ratio=max_ratio,
            device=device_t,
            dtype=dtype,
        )

        params = [p for p in circuit.parameters() if p.requires_grad]
        if optimizer == "adam":
            opt = torch.optim.Adam(params, lr=float(lr))
        elif optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=float(lr), momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        loss_history: List[float] = []

        def _loss(pred_db: torch.Tensor) -> torch.Tensor:
            if loss_kind == "mse_db":
                return torch.mean((pred_db - target) ** 2)
            if loss_kind == "mae_db":
                return torch.mean(torch.abs(pred_db - target))
            raise ValueError(f"Unknown loss_kind: {loss_kind}")

        with torch.no_grad():
            init_pred = circuit(freq, output="s21_db")
            init_loss = float(_loss(init_pred).item())

        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            pred = circuit(freq, output="s21_db")
            loss = _loss(pred)
            loss.backward()
            opt.step()
            loss_history.append(float(loss.detach().cpu().item()))

        with torch.no_grad():
            final_pred = circuit(freq, output="s21_db")
            final_loss = float(_loss(final_pred).item())
            refined_values = circuit.values().detach().cpu().tolist()

        refined_components: List[ComponentSpec] = []
        for c, v in zip(comps, refined_values):
            refined_components.append(
                ComponentSpec(
                    ctype=c.ctype,
                    role=c.role,
                    value_si=float(v),
                    std_label=c.std_label,
                    node1=c.node1,
                    node2=c.node2,
                )
            )

        return RefinementResult(
            refined_components=refined_components,
            loss_history=loss_history,
            initial_loss=init_loss,
            final_loss=final_loss,
        )
