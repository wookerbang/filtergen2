from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


def db_to_mag(db: np.ndarray) -> np.ndarray:
    """dB magnitude -> linear magnitude."""
    return np.power(10.0, np.asarray(db, dtype=float) / 20.0)


def mag_to_db(mag: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Linear magnitude -> dB magnitude."""
    mag = np.asarray(mag, dtype=float)
    return 20.0 * np.log10(np.abs(mag) + float(eps))


ErrorKind = Literal["mae_lin", "rmse_lin", "mae_db", "rmse_db", "maxe_lin", "maxe_db"]


@dataclass(frozen=True)
class WaveformError:
    kind: ErrorKind
    value: float


def waveform_error(
    pred_s21_db: np.ndarray,
    target_s21_db: np.ndarray,
    *,
    kind: ErrorKind = "mae_lin",
) -> WaveformError:
    """
    Compute a scalar error between two S21 magnitude responses.

    """
    pred_db = np.asarray(pred_s21_db, dtype=float)
    tgt_db = np.asarray(target_s21_db, dtype=float)
    if pred_db.shape != tgt_db.shape:
        raise ValueError(f"Shape mismatch: pred {pred_db.shape} vs target {tgt_db.shape}")

    if kind.endswith("_lin"):
        pred = db_to_mag(pred_db)
        tgt = db_to_mag(tgt_db)
        diff = pred - tgt
    else:
        diff = pred_db - tgt_db

    if kind.startswith("mae"):
        v = float(np.mean(np.abs(diff)))
    elif kind.startswith("rmse"):
        v = float(np.sqrt(np.mean(np.square(diff))))
    elif kind.startswith("maxe"):
        v = float(np.max(np.abs(diff)))
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return WaveformError(kind=kind, value=v)


def success_at_tau(err: WaveformError, tau: float) -> bool:
    return float(err.value) <= float(tau)


def max_singular_value_2port(S: np.ndarray) -> np.ndarray:
    """
    Compute σ_max(S) per frequency for 2-port S-parameters.
    S: (F,2,2) complex.
    Returns: (F,) float.
    """
    S = np.asarray(S)
    if S.ndim != 3 or S.shape[1:] != (2, 2):
        raise ValueError(f"Expected S shape (F,2,2), got {S.shape}")
    out = np.empty(S.shape[0], dtype=float)
    for k in range(S.shape[0]):
        sv = np.linalg.svd(S[k], compute_uv=False)
        out[k] = float(np.max(sv))
    return out


@dataclass(frozen=True)
class PassivityMetrics:
    sigma_max_max: float
    violation_max: float
    violation_mean: float


def passivity_metrics(S: np.ndarray, *, tol: float = 1e-6) -> PassivityMetrics:
    """
    Passivity diagnostic for 2-port networks: σ_max(S(jw)) <= 1.
    """
    sigma = max_singular_value_2port(S)
    viol = np.maximum(sigma - 1.0, 0.0)
    return PassivityMetrics(
        sigma_max_max=float(np.max(sigma)) if sigma.size else 0.0,
        violation_max=float(np.max(viol)) if viol.size else 0.0,
        violation_mean=float(np.mean(viol)) if viol.size else 0.0,
    )
