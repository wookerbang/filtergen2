from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np
import torch

from .differentiable_rf import DynamicCircuitAssembler, InferenceTimeOptimizer, RefinementResult


@dataclass(frozen=True)
class FastTrackSParams:
    s11: np.ndarray
    s21: np.ndarray
    s12: np.ndarray
    s22: np.ndarray


class FastTrackEngine:
    """
    Fast Track physics engine (ABCD + smart fusion + Q).

    Intended usage:
      - training data generation
      - best-of-K screening
      - gradient-based refinement
    """

    def __init__(
        self,
        *,
        z0: float = 50.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.z0 = float(z0)
        self.device = torch.device(device)
        self.dtype = dtype
        self._assembler = DynamicCircuitAssembler(z0=self.z0)
        self._refiner = InferenceTimeOptimizer(z0=self.z0)

    def compile(
        self,
        components: Sequence[object],
        *,
        trainable: bool = False,
        max_ratio: float | None = 2.0,
        min_value: float = 1e-30,
        q_L: float | None = None,
        q_C: float | None = None,
        eps: float = 1e-30,
    ):
        return self._assembler.assemble(
            components,
            trainable=trainable,
            max_ratio=max_ratio,
            min_value=min_value,
            q_L=q_L,
            q_C=q_C,
            eps=eps,
            device=self.device,
            dtype=self.dtype,
        )

    def simulate_sparams(
        self,
        components: Sequence[object],
        freq_hz: Sequence[float] | np.ndarray,
        *,
        q_L: float | None = None,
        q_C: float | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | None = None,
    ) -> FastTrackSParams:
        if q_model == "fixed_ref" and (q_L is not None or q_C is not None) and ref_freq_hz is None:
            f = np.asarray(freq_hz, dtype=float).reshape(-1)
            ref_freq_hz = float(np.sqrt(float(np.min(f)) * float(np.max(f))))
        circuit, _ = self._assembler.assemble(
            components,
            trainable=False,
            q_L=q_L,
            q_C=q_C,
            device=self.device,
            dtype=self.dtype,
        )
        freq = torch.as_tensor(freq_hz, device=self.device, dtype=self.dtype).reshape(-1)
        S11, S21, S12, S22 = circuit(
            freq,
            output="sparams",
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
        )
        return FastTrackSParams(
            s11=S11.detach().cpu().numpy(),
            s21=S21.detach().cpu().numpy(),
            s12=S12.detach().cpu().numpy(),
            s22=S22.detach().cpu().numpy(),
        )

    def simulate_sparams_db(
        self,
        components: Sequence[object],
        freq_hz: Sequence[float] | np.ndarray,
        *,
        q_L: float | None = None,
        q_C: float | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | None = None,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sparams = self.simulate_sparams(
            components,
            freq_hz,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
        )
        s21_db = 20.0 * np.log10(np.abs(sparams.s21) + float(eps))
        s11_db = 20.0 * np.log10(np.abs(sparams.s11) + float(eps))
        return s21_db, s11_db

    def simulate_s21_db(
        self,
        components: Sequence[object],
        freq_hz: Sequence[float] | np.ndarray,
        *,
        q_L: float | None = None,
        q_C: float | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | None = None,
        eps: float = 1e-12,
    ) -> np.ndarray:
        s21_db, _ = self.simulate_sparams_db(
            components,
            freq_hz,
            q_L=q_L,
            q_C=q_C,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
            eps=eps,
        )
        return s21_db

    def refine(
        self,
        components: Sequence[object],
        *,
        freq_hz: Sequence[float] | np.ndarray,
        target_s21_db: Sequence[float] | np.ndarray | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        q_model: Literal["freq_dependent", "fixed_ref"] = "freq_dependent",
        ref_freq_hz: float | None = None,
        **kwargs,
    ) -> RefinementResult:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        return self._refiner.refine(
            components,
            freq_hz=freq_hz,
            target_s21_db=target_s21_db,
            device=dev,
            dtype=dt,
            q_model=q_model,
            ref_freq_hz=ref_freq_hz,
            **kwargs,
        )
