"""
DSL: ML-friendly IR with Macro / Repeat / Typed numeric slots.

Key ideas:
- Canonical main program: <MAIN> Ports Body </MAIN>
- Body supports structured cascade repeat blocks with frozen macro library.
- Numeric slots are typed (<VAL_L>/<VAL_C>/<VAL_R>/...), paired with continuous heads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

from .schema import ComponentSpec
from .action_codec import components_to_action_tokens

# ---- Tokens ----

BOS = "<BOS>"
# Align EOS with tokenizer eos (T5 uses </s>) to avoid double-eos drift.
EOS = "</s>"

MAIN_START = "<MAIN>"
MAIN_END = "</MAIN>"

REPEAT_START = "<REPEAT>"
REPEAT_END = "</REPEAT>"
CASCADE = "<CASCADE>"
CALL = "<CALL>"

CELL = "<CELL>"
# Optional cell index markers to stabilize long-K decoding.
CELL_INDEX_TOKENS = [f"<CELL_IDX_{i}>" for i in range(32)]

PORT_IN = "<P_IN>"
PORT_OUT = "<P_OUT>"
PORT_GND = "<P_GND>"

VAL_L = "<VAL_L>"
VAL_C = "<VAL_C>"
VAL_R = "<VAL_R>"
VAL_TL_Z0 = "<VAL_TL_Z0>"
VAL_TL_LEN = "<VAL_TL_LEN>"
VAL_L_PAR = "<VAL_L_PAR>"
VAL_C_PAR = "<VAL_C_PAR>"
VAL_L_SER = "<VAL_L_SER>"
VAL_C_SER = "<VAL_C_SER>"
VAL_L_NOTCH = "<VAL_L_NOTCH>"
VAL_C_NOTCH = "<VAL_C_NOTCH>"
VAL_NONE = "<VAL_NONE>"

VALUE_SLOTS = [
    VAL_L,
    VAL_C,
    VAL_R,
    VAL_TL_Z0,
    VAL_TL_LEN,
    VAL_L_PAR,
    VAL_C_PAR,
    VAL_L_SER,
    VAL_C_SER,
    VAL_L_NOTCH,
    VAL_C_NOTCH,
]

ORDER_TOKENS = [f"<ORDER_{i}>" for i in range(1, 17)]

# Repeat factors (frozen finite set + extensible varint).
K_TOKENS = [f"<K_{k}>" for k in range(1, 13)]
K_VAR_START = "<K>"
K_VAR_END = "</K>"
DIGIT_TOKENS = [f"<D_{d}>" for d in range(10)]

# Macro tokens (frozen library)
MACRO_CELL_LS_CS = "<MAC_CELL_LS_CS>"
MACRO_CELL_LS_CS_A = "<MAC_CELL_LS_CS_A>"
MACRO_CELL_LS_CS_AB = "<MAC_CELL_LS_CS_AB>"
MACRO_CELL_CS_LS = "<MAC_CELL_CS_LS>"
MACRO_CELL_CS_LS_A = "<MAC_CELL_CS_LS_A>"
MACRO_CELL_CS_LS_AB = "<MAC_CELL_CS_LS_AB>"
MACRO_NOTCH_SHUNT_LC_SER = "<MAC_NOTCH_SHUNT_LC_SER>"
MACRO_CELL_LS = "<MAC_CELL_LS>"
MACRO_CELL_CS = "<MAC_CELL_CS>"
MACRO_PI_CLC = "<MAC_PI_CLC>"
MACRO_T_LCL = "<MAC_T_LCL>"
MACRO_DOUBLE_SERIES_LC = "<MAC_DOUBLE_SERIES_LC>"
MACRO_DOUBLE_SHUNT_LC = "<MAC_DOUBLE_SHUNT_LC>"
MACRO_BRIDGE_C = "<MAC_BRIDGE_C>"
MACRO_CELL_BS_PAR_SER_LC = "<MAC_CELL_BS_PAR_SER_LC>"
MACRO_CELL_BS_PAR_SER_LC_A = "<MAC_CELL_BS_PAR_SER_LC_A>"
MACRO_CELL_BS_PAR_SER_LC_AB = "<MAC_CELL_BS_PAR_SER_LC_AB>"
MACRO_CELL_BS_PAR_LC = "<MAC_CELL_BS_PAR_LC>"
MACRO_CELL_BP_SER_PAR_LC = "<MAC_CELL_BP_SER_PAR_LC>"
MACRO_CELL_BP_SER_PAR_LC_A = "<MAC_CELL_BP_SER_PAR_LC_A>"
MACRO_CELL_BP_SER_PAR_LC_AB = "<MAC_CELL_BP_SER_PAR_LC_AB>"
MACRO_CELL_BP_SER_LC = "<MAC_CELL_BP_SER_LC>"

MACRO_IDS = [
    MACRO_CELL_LS_CS,
    MACRO_CELL_LS_CS_A,
    MACRO_CELL_LS_CS_AB,
    MACRO_CELL_CS_LS,
    MACRO_CELL_CS_LS_A,
    MACRO_CELL_CS_LS_AB,
    MACRO_NOTCH_SHUNT_LC_SER,
    MACRO_CELL_LS,
    MACRO_CELL_CS,
    MACRO_PI_CLC,
    MACRO_T_LCL,
    MACRO_DOUBLE_SERIES_LC,
    MACRO_DOUBLE_SHUNT_LC,
    MACRO_BRIDGE_C,
    MACRO_CELL_BS_PAR_SER_LC,
    MACRO_CELL_BS_PAR_SER_LC_A,
    MACRO_CELL_BS_PAR_SER_LC_AB,
    MACRO_CELL_BS_PAR_LC,
    MACRO_CELL_BP_SER_PAR_LC,
    MACRO_CELL_BP_SER_PAR_LC_A,
    MACRO_CELL_BP_SER_PAR_LC_AB,
    MACRO_CELL_BP_SER_LC,
]


# ---- Macro definitions ----


@dataclass(frozen=True)
class MacroDef:
    name: str
    slot_types: Tuple[str, ...]  # e.g., ("L", "C")
    # inst_idx is a deterministic macro instance index used for internal node allocation.
    expand_fn: Callable[[str, str, str, List[float], int], List[ComponentSpec]]


def _valid_value(v: float) -> float | None:
    try:
        fv = float(v)
    except Exception:
        return None
    if not math.isfinite(fv) or fv <= 0.0:
        return None
    return fv


def _is_valid_value(v: float) -> bool:
    try:
        fv = float(v)
    except Exception:
        return False
    return math.isfinite(fv) and fv > 0.0


def _maybe_add_component(
    out: List[ComponentSpec],
    ctype: str,
    role: str,
    value: float,
    node1: str,
    node2: str,
) -> None:
    fv = _valid_value(value)
    if fv is None:
        return
    out.append(ComponentSpec(ctype, role, fv, None, node1, node2))


def _expand_cell_ls_cs(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, b)
    _maybe_add_component(comps, "C", "shunt", C_val, b, gnd)
    return comps


def _expand_cell_ls_cs_a(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, b)
    _maybe_add_component(comps, "C", "shunt", C_val, a, gnd)
    return comps


def _expand_cell_ls_cs_ab(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_a, C_b = (vals + [0.0, 0.0, 0.0])[:3]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, b)
    _maybe_add_component(comps, "C", "shunt", C_a, a, gnd)
    _maybe_add_component(comps, "C", "shunt", C_b, b, gnd)
    return comps


def _expand_cell_cs_ls(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val, L_val = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    _maybe_add_component(comps, "L", "shunt", L_val, b, gnd)
    return comps


def _expand_cell_cs_ls_a(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val, L_val = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    _maybe_add_component(comps, "L", "shunt", L_val, a, gnd)
    return comps


def _expand_cell_cs_ls_ab(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val, L_a, L_b = (vals + [0.0, 0.0, 0.0])[:3]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    _maybe_add_component(comps, "L", "shunt", L_a, a, gnd)
    _maybe_add_component(comps, "L", "shunt", L_b, b, gnd)
    return comps


def _expand_notch_shunt(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # series LC from anchor (b) to ground via internal node x
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, b, x)
    _maybe_add_component(comps, "C", "series", C_val, x, gnd)
    return comps


def _expand_cell_ls(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, b)
    return comps


def _expand_cell_cs(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    return comps


def _expand_pi_clc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C1, L, C2 = (vals + [0.0, 0.0, 0.0])[:3]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "shunt", C1, a, gnd)
    _maybe_add_component(comps, "L", "series", L, a, b)
    _maybe_add_component(comps, "C", "shunt", C2, b, gnd)
    return comps


def _expand_t_lcl(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L1, C, L2 = (vals + [0.0, 0.0, 0.0])[:3]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L1, a, x)
    _maybe_add_component(comps, "C", "shunt", C, x, gnd)
    _maybe_add_component(comps, "L", "series", L2, x, b)
    return comps


def _expand_double_series_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L, C = (vals + [0.0, 0.0])[:2]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L, a, x)
    _maybe_add_component(comps, "C", "series", C, x, b)
    return comps


def _expand_double_shunt_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L, C = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "shunt", L, b, gnd)
    _maybe_add_component(comps, "C", "shunt", C, b, gnd)
    return comps


def _expand_bridge_c(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Simple bridge capacitor across anchor nodes.
    C_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    return comps


def _expand_cell_bs_par_ser_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series path: parallel LC between a and b.
    # Shunt path: series LC from b to gnd via internal node.
    Lp, Cp, Ls, Cs = (vals + [0.0, 0.0, 0.0, 0.0])[:4]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", Lp, a, b)
    _maybe_add_component(comps, "C", "series", Cp, a, b)
    _maybe_add_component(comps, "L", "series", Ls, b, x)
    _maybe_add_component(comps, "C", "series", Cs, x, gnd)
    return comps


def _expand_cell_bs_par_ser_lc_a(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series path: parallel LC between a and b.
    # Shunt path: series LC from a to gnd via internal node.
    Lp, Cp, Ls, Cs = (vals + [0.0, 0.0, 0.0, 0.0])[:4]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", Lp, a, b)
    _maybe_add_component(comps, "C", "series", Cp, a, b)
    _maybe_add_component(comps, "L", "series", Ls, a, x)
    _maybe_add_component(comps, "C", "series", Cs, x, gnd)
    return comps


def _expand_cell_bs_par_ser_lc_ab(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series path: parallel LC between a and b.
    # Shunt path: series LC from a and b to gnd (two branches).
    Lp, Cp, Ls_a, Cs_a, Ls_b, Cs_b = (vals + [0.0] * 6)[:6]
    x_a = f"x{inst_idx}_0"
    x_b = f"x{inst_idx}_1"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", Lp, a, b)
    _maybe_add_component(comps, "C", "series", Cp, a, b)
    _maybe_add_component(comps, "L", "series", Ls_a, a, x_a)
    _maybe_add_component(comps, "C", "series", Cs_a, x_a, gnd)
    _maybe_add_component(comps, "L", "series", Ls_b, b, x_b)
    _maybe_add_component(comps, "C", "series", Cs_b, x_b, gnd)
    return comps


def _expand_cell_bs_par_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series path: parallel LC between a and b.
    Lp, Cp = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", Lp, a, b)
    _maybe_add_component(comps, "C", "series", Cp, a, b)
    return comps


def _expand_cell_bp_ser_par_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series LC between a and b; parallel LC to gnd at b; optional notch series LC to gnd at b.
    Ls, Cs, Lp, Cp, Ln, Cn = (vals + [0.0] * 6)[:6]
    comps: List[ComponentSpec] = []
    mid = f"x{inst_idx}_0"
    _maybe_add_component(comps, "L", "series", Ls, a, mid)
    _maybe_add_component(comps, "C", "series", Cs, mid, b)
    _maybe_add_component(comps, "L", "shunt", Lp, b, gnd)
    _maybe_add_component(comps, "C", "shunt", Cp, b, gnd)
    if _valid_value(Ln) is not None and _valid_value(Cn) is not None:
        x = f"x{inst_idx}_1"
        _maybe_add_component(comps, "L", "series", Ln, b, x)
        _maybe_add_component(comps, "C", "series", Cn, x, gnd)
    return comps


def _expand_cell_bp_ser_par_lc_a(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series LC between a and b; parallel LC to gnd at a; optional notch series LC to gnd at a.
    Ls, Cs, Lp, Cp, Ln, Cn = (vals + [0.0] * 6)[:6]
    comps: List[ComponentSpec] = []
    mid = f"x{inst_idx}_0"
    _maybe_add_component(comps, "L", "series", Ls, a, mid)
    _maybe_add_component(comps, "C", "series", Cs, mid, b)
    _maybe_add_component(comps, "L", "shunt", Lp, a, gnd)
    _maybe_add_component(comps, "C", "shunt", Cp, a, gnd)
    if _valid_value(Ln) is not None and _valid_value(Cn) is not None:
        x = f"x{inst_idx}_1"
        _maybe_add_component(comps, "L", "series", Ln, a, x)
        _maybe_add_component(comps, "C", "series", Cn, x, gnd)
    return comps


def _expand_cell_bp_ser_lc(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series LC between a and b without shunt.
    Ls, Cs = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    mid = f"x{inst_idx}_0"
    _maybe_add_component(comps, "L", "series", Ls, a, mid)
    _maybe_add_component(comps, "C", "series", Cs, mid, b)
    return comps


def _expand_cell_bp_ser_par_lc_ab(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    # Series LC between a and b; parallel LC to gnd at a and b; optional notch series LC to gnd at a.
    Ls, Cs, Lp_a, Cp_a, Lp_b, Cp_b, Ln, Cn = (vals + [0.0] * 8)[:8]
    comps: List[ComponentSpec] = []
    mid = f"x{inst_idx}_0"
    _maybe_add_component(comps, "L", "series", Ls, a, mid)
    _maybe_add_component(comps, "C", "series", Cs, mid, b)
    _maybe_add_component(comps, "L", "shunt", Lp_a, a, gnd)
    _maybe_add_component(comps, "C", "shunt", Cp_a, a, gnd)
    _maybe_add_component(comps, "L", "shunt", Lp_b, b, gnd)
    _maybe_add_component(comps, "C", "shunt", Cp_b, b, gnd)
    if _valid_value(Ln) is not None and _valid_value(Cn) is not None:
        x = f"x{inst_idx}_1"
        _maybe_add_component(comps, "L", "series", Ln, a, x)
        _maybe_add_component(comps, "C", "series", Cn, x, gnd)
    return comps


MACRO_LIBRARY: Dict[str, MacroDef] = {
    MACRO_CELL_LS_CS: MacroDef(MACRO_CELL_LS_CS, ("L", "C"), _expand_cell_ls_cs),
    MACRO_CELL_LS_CS_A: MacroDef(MACRO_CELL_LS_CS_A, ("L", "C"), _expand_cell_ls_cs_a),
    MACRO_CELL_LS_CS_AB: MacroDef(MACRO_CELL_LS_CS_AB, ("L", "C", "C"), _expand_cell_ls_cs_ab),
    MACRO_CELL_CS_LS: MacroDef(MACRO_CELL_CS_LS, ("C", "L"), _expand_cell_cs_ls),
    MACRO_CELL_CS_LS_A: MacroDef(MACRO_CELL_CS_LS_A, ("C", "L"), _expand_cell_cs_ls_a),
    MACRO_CELL_CS_LS_AB: MacroDef(MACRO_CELL_CS_LS_AB, ("C", "L", "L"), _expand_cell_cs_ls_ab),
    MACRO_NOTCH_SHUNT_LC_SER: MacroDef(MACRO_NOTCH_SHUNT_LC_SER, ("L", "C"), _expand_notch_shunt),
    MACRO_CELL_LS: MacroDef(MACRO_CELL_LS, ("L",), _expand_cell_ls),
    MACRO_CELL_CS: MacroDef(MACRO_CELL_CS, ("C",), _expand_cell_cs),
    MACRO_PI_CLC: MacroDef(MACRO_PI_CLC, ("C", "L", "C"), _expand_pi_clc),
    MACRO_T_LCL: MacroDef(MACRO_T_LCL, ("L", "C", "L"), _expand_t_lcl),
    MACRO_DOUBLE_SERIES_LC: MacroDef(MACRO_DOUBLE_SERIES_LC, ("L", "C"), _expand_double_series_lc),
    MACRO_DOUBLE_SHUNT_LC: MacroDef(MACRO_DOUBLE_SHUNT_LC, ("L", "C"), _expand_double_shunt_lc),
    MACRO_BRIDGE_C: MacroDef(MACRO_BRIDGE_C, ("C",), _expand_bridge_c),
    MACRO_CELL_BS_PAR_SER_LC: MacroDef(MACRO_CELL_BS_PAR_SER_LC, ("L_PAR", "C_PAR", "L_SER", "C_SER"), _expand_cell_bs_par_ser_lc),
    MACRO_CELL_BS_PAR_SER_LC_A: MacroDef(
        MACRO_CELL_BS_PAR_SER_LC_A,
        ("L_PAR", "C_PAR", "L_SER", "C_SER"),
        _expand_cell_bs_par_ser_lc_a,
    ),
    MACRO_CELL_BS_PAR_SER_LC_AB: MacroDef(
        MACRO_CELL_BS_PAR_SER_LC_AB,
        ("L_PAR", "C_PAR", "L_SER", "C_SER", "L_SER", "C_SER"),
        _expand_cell_bs_par_ser_lc_ab,
    ),
    MACRO_CELL_BS_PAR_LC: MacroDef(MACRO_CELL_BS_PAR_LC, ("L_PAR", "C_PAR"), _expand_cell_bs_par_lc),
    MACRO_CELL_BP_SER_PAR_LC: MacroDef(
        MACRO_CELL_BP_SER_PAR_LC,
        ("L_SER", "C_SER", "L_PAR", "C_PAR", "L_NOTCH", "C_NOTCH"),
        _expand_cell_bp_ser_par_lc,
    ),
    MACRO_CELL_BP_SER_PAR_LC_A: MacroDef(
        MACRO_CELL_BP_SER_PAR_LC_A,
        ("L_SER", "C_SER", "L_PAR", "C_PAR", "L_NOTCH", "C_NOTCH"),
        _expand_cell_bp_ser_par_lc_a,
    ),
    MACRO_CELL_BP_SER_PAR_LC_AB: MacroDef(
        MACRO_CELL_BP_SER_PAR_LC_AB,
        ("L_SER", "C_SER", "L_PAR", "C_PAR", "L_PAR", "C_PAR", "L_NOTCH", "C_NOTCH"),
        _expand_cell_bp_ser_par_lc_ab,
    ),
    MACRO_CELL_BP_SER_LC: MacroDef(MACRO_CELL_BP_SER_LC, ("L_SER", "C_SER"), _expand_cell_bp_ser_lc),
}

# Slot type -> token mapping (kept centralized to avoid drift across encoder/decoder/mask).
SLOT_TYPE_TO_TOKEN = {
    "L": VAL_L,
    "C": VAL_C,
    "R": VAL_R,
    "TL_Z0": VAL_TL_Z0,
    "TL_LEN": VAL_TL_LEN,
    "L_PAR": VAL_L_PAR,
    "C_PAR": VAL_C_PAR,
    "L_SER": VAL_L_SER,
    "C_SER": VAL_C_SER,
    "L_NOTCH": VAL_L_NOTCH,
    "C_NOTCH": VAL_C_NOTCH,
}


# ---- Vocab builder ----


def build_dsl_vocab(
    *,
    macro_ids: Sequence[str] | None = None,
    include_bos: bool = True,
    order_range: Tuple[int, int] | None = None,
) -> List[str]:
    vocab: Set[str] = set()
    vocab.update([MAIN_START, MAIN_END, REPEAT_START, REPEAT_END, CASCADE, CALL])
    vocab.update([CELL])
    vocab.update(CELL_INDEX_TOKENS)
    vocab.update([PORT_IN, PORT_OUT, PORT_GND])
    vocab.update(K_TOKENS)
    vocab.update([K_VAR_START, K_VAR_END])
    vocab.update(DIGIT_TOKENS)
    vocab.update(VALUE_SLOTS)
    vocab.update([VAL_NONE])
    if order_range is None:
        vocab.update(ORDER_TOKENS)
    else:
        lo, hi = int(order_range[0]), int(order_range[1])
        if lo <= hi:
            vocab.update([f"<ORDER_{i}>" for i in range(lo, hi + 1)])
    vocab.update(macro_ids or MACRO_IDS)
    # EOS is always included because encoder unconditionally appends it.
    vocab.update([EOS])
    if include_bos:
        vocab.update([BOS])
    return sorted(vocab)


# ---- Encoding helpers (components -> segments) ----


def _detect_shunt_series_lc(comps: Sequence[ComponentSpec]) -> Tuple[Dict[str, Tuple[float, float]], Set[int]]:
    node_to_series: Dict[str, List[Tuple[int, ComponentSpec]]] = {}
    for idx, c in enumerate(comps):
        if c.role != "series":
            continue
        node_to_series.setdefault(c.node1, []).append((idx, c))
        node_to_series.setdefault(c.node2, []).append((idx, c))
    out: Dict[str, Tuple[float, float]] = {}
    branch_ids: Set[int] = set()
    for node, items in node_to_series.items():
        if node == "gnd":
            continue
        if len(items) != 2:
            continue
        (i1, c1), (i2, c2) = items
        o1 = c1.node2 if c1.node1 == node else c1.node1
        o2 = c2.node2 if c2.node1 == node else c2.node1
        c1_to_gnd = o1 == "gnd"
        c2_to_gnd = o2 == "gnd"
        if c1_to_gnd == c2_to_gnd:
            continue
        comp_to_gnd = c1 if c1_to_gnd else c2
        comp_to_anchor = c2 if c1_to_gnd else c1
        anchor = comp_to_anchor.node2 if comp_to_anchor.node1 == node else comp_to_anchor.node1
        if anchor == "gnd":
            continue
        types = {comp_to_gnd.ctype, comp_to_anchor.ctype}
        if types != {"L", "C"}:
            continue
        L_val = comp_to_gnd.value_si if comp_to_gnd.ctype == "L" else comp_to_anchor.value_si
        C_val = comp_to_gnd.value_si if comp_to_gnd.ctype == "C" else comp_to_anchor.value_si
        out[anchor] = (float(L_val), float(C_val))
        branch_ids.update([i1, i2])
    return out, branch_ids


def _build_series_path(series_pairs: Dict[Tuple[str, str], List[ComponentSpec]]) -> List[str] | None:
    series_adj: Dict[str, List[str]] = {}
    for (n1, n2) in series_pairs.keys():
        series_adj.setdefault(n1, []).append(n2)
        series_adj.setdefault(n2, []).append(n1)
    if "in" not in series_adj or "out" not in series_adj:
        return None
    path_nodes = ["in"]
    prev = None
    current = "in"
    while current != "out":
        neighbors = [n for n in series_adj.get(current, []) if n != prev]
        if len(neighbors) != 1:
            return None
        nxt = neighbors[0]
        path_nodes.append(nxt)
        prev, current = current, nxt
    return path_nodes


def _assign_shunts_to_edges(
    nodes: Sequence[str],
    shunt_map: Dict[str, Dict[str, float]],
    topology_type: str | None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    num_edges = max(0, len(nodes) - 1)
    shunt_a = [dict() for _ in range(num_edges)]
    shunt_b = [dict() for _ in range(num_edges)]
    topo = str(topology_type or "t").lower()
    use_pi = topo == "pi"
    for idx, node in enumerate(nodes):
        vals = shunt_map.get(node)
        if not vals:
            continue
        if use_pi:
            if node == "out":
                edge_idx = num_edges - 1
                if edge_idx >= 0:
                    shunt_b[edge_idx] = dict(vals)
            else:
                edge_idx = idx
                if edge_idx < num_edges:
                    shunt_a[edge_idx] = dict(vals)
        else:
            if node == "in":
                edge_idx = 0
                if edge_idx < num_edges:
                    shunt_a[edge_idx] = dict(vals)
            else:
                edge_idx = idx - 1
                if edge_idx >= 0:
                    shunt_b[edge_idx] = dict(vals)
    return shunt_a, shunt_b


def _group_cells_by_macro(cells: Sequence[Tuple[str, List[float]]]) -> List[Tuple[str, List[List[float]]]]:
    segments: List[Tuple[str, List[List[float]]]] = []
    current_macro: str | None = None
    current_vals: List[List[float]] = []
    for macro, vals in cells:
        if macro != current_macro:
            if current_macro is not None:
                segments.append((current_macro, current_vals))
            current_macro = macro
            current_vals = [list(vals)]
        else:
            current_vals.append(list(vals))
    if current_macro is not None:
        segments.append((current_macro, current_vals))
    return segments


def _extract_lc_pair(vals: Dict[str, float], *, allow_empty: bool = True) -> Tuple[float, float, bool]:
    if not vals:
        return float("nan"), float("nan"), False
    L_val = vals.get("L")
    C_val = vals.get("C")
    has_L = _is_valid_value(L_val)
    has_C = _is_valid_value(C_val)
    if has_L != has_C:
        raise ValueError("Incomplete LC pair.")
    if not has_L:
        if allow_empty:
            return float("nan"), float("nan"), False
        raise ValueError("Missing LC pair.")
    return float(L_val), float(C_val), True


def _extract_single(vals: Dict[str, float], key: str) -> Tuple[float, bool]:
    val = vals.get(key)
    if _is_valid_value(val):
        return float(val), True
    return float("nan"), False


def _build_ladder_cells(
    components: Sequence[ComponentSpec],
    *,
    topology_type: str | None,
    series_type: str,
    shunt_type: str,
    macro_a: str,
    macro_b: str,
    macro_ab: str,
    macro_series: str,
) -> List[Tuple[str, List[float]]]:
    series_pairs: Dict[Tuple[str, str], List[ComponentSpec]] = {}
    for c in components:
        if c.role != "series":
            continue
        if c.node1 == "gnd" or c.node2 == "gnd":
            continue
        if c.ctype != series_type:
            continue
        key = tuple(sorted((c.node1, c.node2)))
        series_pairs.setdefault(key, []).append(c)
    path_nodes = _build_series_path(series_pairs)
    if not path_nodes:
        raise ValueError("Main path parse failed.")

    shunt_map: Dict[str, Dict[str, float]] = {}
    for c in components:
        if c.role != "shunt":
            continue
        if c.ctype != shunt_type:
            continue
        node = c.node1 if c.node2 == "gnd" else c.node2
        entry = shunt_map.setdefault(node, {})
        if shunt_type in entry:
            raise ValueError(f"Duplicate shunt {shunt_type} at node {node}.")
        entry[shunt_type] = float(c.value_si)

    shunt_a, shunt_b = _assign_shunts_to_edges(path_nodes, shunt_map, topology_type)

    cells: List[Tuple[str, List[float]]] = []
    for idx in range(len(path_nodes) - 1):
        a = path_nodes[idx]
        b = path_nodes[idx + 1]
        key = tuple(sorted((a, b)))
        series = series_pairs.get(key, [])
        if len(series) != 1:
            raise ValueError("Series path parse failed.")
        series_val = float(series[0].value_si)

        a_val, has_a = _extract_single(shunt_a[idx], shunt_type)
        b_val, has_b = _extract_single(shunt_b[idx], shunt_type)

        if has_a and has_b:
            macro = macro_ab
            vals = [series_val, a_val, b_val]
        elif has_a:
            macro = macro_a
            vals = [series_val, a_val]
        elif has_b:
            macro = macro_b
            vals = [series_val, b_val]
        else:
            macro = macro_series
            vals = [series_val]
        cells.append((macro, vals))
    return cells


def _extract_series_lc_pair(comps: Sequence[ComponentSpec]) -> Tuple[float, float]:
    L_val = None
    C_val = None
    for c in comps:
        if c.ctype == "L":
            L_val = float(c.value_si)
        elif c.ctype == "C":
            C_val = float(c.value_si)
    if not _is_valid_value(L_val) or not _is_valid_value(C_val):
        raise ValueError("Series LC pair incomplete.")
    return float(L_val), float(C_val)


def _build_bandpass_cells(
    components: Sequence[ComponentSpec],
    *,
    topology_type: str | None,
) -> List[Tuple[str, List[float]]]:
    notch_map, branch_comp_ids = _detect_shunt_series_lc(components)

    series_pairs: Dict[Tuple[str, str], List[ComponentSpec]] = {}
    for idx, c in enumerate(components):
        if idx in branch_comp_ids:
            continue
        if c.role != "series":
            continue
        if c.node1 == "gnd" or c.node2 == "gnd":
            continue
        key = tuple(sorted((c.node1, c.node2)))
        series_pairs.setdefault(key, []).append(c)

    path_nodes = _build_series_path(series_pairs)
    if not path_nodes:
        raise ValueError("Bandpass path parse failed.")
    if (len(path_nodes) - 1) % 2 != 0:
        raise ValueError("Bandpass path length is not even.")

    main_nodes = path_nodes[::2]
    shunt_map: Dict[str, Dict[str, float]] = {}
    for c in components:
        if c.role != "shunt":
            continue
        node = c.node1 if c.node2 == "gnd" else c.node2
        entry = shunt_map.setdefault(node, {})
        if c.ctype in entry:
            raise ValueError(f"Duplicate shunt {c.ctype} at node {node}.")
        entry[c.ctype] = float(c.value_si)

    notch_node_map: Dict[str, Dict[str, float]] = {
        node: {"L": float(vals[0]), "C": float(vals[1])} for node, vals in notch_map.items()
    }

    shunt_a, shunt_b = _assign_shunts_to_edges(main_nodes, shunt_map, topology_type)
    notch_a, notch_b = _assign_shunts_to_edges(main_nodes, notch_node_map, topology_type)

    cells: List[Tuple[str, List[float]]] = []
    for idx in range(len(main_nodes) - 1):
        a = main_nodes[idx]
        mid = path_nodes[2 * idx + 1]
        b = main_nodes[idx + 1]
        key1 = tuple(sorted((a, mid)))
        key2 = tuple(sorted((mid, b)))
        series = series_pairs.get(key1, []) + series_pairs.get(key2, [])
        Ls, Cs = _extract_series_lc_pair(series)

        Lp_a, Cp_a, has_a = _extract_lc_pair(shunt_a[idx])
        Lp_b, Cp_b, has_b = _extract_lc_pair(shunt_b[idx])
        Ln_a, Cn_a, has_notch_a = _extract_lc_pair(notch_a[idx])
        Ln_b, Cn_b, has_notch_b = _extract_lc_pair(notch_b[idx])

        if has_a and has_b:
            macro = MACRO_CELL_BP_SER_PAR_LC_AB
            vals = [Ls, Cs, Lp_a, Cp_a, Lp_b, Cp_b, Ln_a, Cn_a]
        elif has_a:
            macro = MACRO_CELL_BP_SER_PAR_LC_A
            vals = [Ls, Cs, Lp_a, Cp_a, Ln_a, Cn_a]
        elif has_b:
            macro = MACRO_CELL_BP_SER_PAR_LC
            vals = [Ls, Cs, Lp_b, Cp_b, Ln_b, Cn_b]
        else:
            if has_notch_a:
                macro = MACRO_CELL_BP_SER_PAR_LC_A
                vals = [Ls, Cs, float("nan"), float("nan"), Ln_a, Cn_a]
            elif has_notch_b:
                macro = MACRO_CELL_BP_SER_PAR_LC
                vals = [Ls, Cs, float("nan"), float("nan"), Ln_b, Cn_b]
            else:
                macro = MACRO_CELL_BP_SER_LC
                vals = [Ls, Cs]
        cells.append((macro, vals))
    return cells


def _build_bandstop_cells(
    components: Sequence[ComponentSpec],
    *,
    topology_type: str | None,
) -> List[Tuple[str, List[float]]]:
    shunt_map_raw, branch_comp_ids = _detect_shunt_series_lc(components)

    series_pairs: Dict[Tuple[str, str], List[ComponentSpec]] = {}
    for idx, c in enumerate(components):
        if idx in branch_comp_ids:
            continue
        if c.role != "series":
            continue
        if c.node1 == "gnd" or c.node2 == "gnd":
            continue
        key = tuple(sorted((c.node1, c.node2)))
        series_pairs.setdefault(key, []).append(c)

    path_nodes = _build_series_path(series_pairs)
    if not path_nodes:
        raise ValueError("Bandstop path parse failed.")

    shunt_node_map: Dict[str, Dict[str, float]] = {
        node: {"L": float(vals[0]), "C": float(vals[1])} for node, vals in shunt_map_raw.items()
    }
    shunt_a, shunt_b = _assign_shunts_to_edges(path_nodes, shunt_node_map, topology_type)

    cells: List[Tuple[str, List[float]]] = []
    for idx in range(len(path_nodes) - 1):
        a = path_nodes[idx]
        b = path_nodes[idx + 1]
        key = tuple(sorted((a, b)))
        series = series_pairs.get(key, [])
        Lp, Cp = _extract_series_lc_pair(series)

        Ls_a, Cs_a, has_a = _extract_lc_pair(shunt_a[idx])
        Ls_b, Cs_b, has_b = _extract_lc_pair(shunt_b[idx])

        if has_a and has_b:
            macro = MACRO_CELL_BS_PAR_SER_LC_AB
            vals = [Lp, Cp, Ls_a, Cs_a, Ls_b, Cs_b]
        elif has_a:
            macro = MACRO_CELL_BS_PAR_SER_LC_A
            vals = [Lp, Cp, Ls_a, Cs_a]
        elif has_b:
            macro = MACRO_CELL_BS_PAR_SER_LC
            vals = [Lp, Cp, Ls_b, Cs_b]
        else:
            macro = MACRO_CELL_BS_PAR_LC
            vals = [Lp, Cp]
        cells.append((macro, vals))
    return cells


def components_to_dsl_segments(
    components: Sequence[ComponentSpec],
    *,
    filter_type: str,
    topology_type: str | None = None,
) -> List[Tuple[str, List[List[float]]]]:
    ftype = str(filter_type or "lowpass")
    if ftype == "lowpass":
        cells = _build_ladder_cells(
            components,
            topology_type=topology_type,
            series_type="L",
            shunt_type="C",
            macro_a=MACRO_CELL_LS_CS_A,
            macro_b=MACRO_CELL_LS_CS,
            macro_ab=MACRO_CELL_LS_CS_AB,
            macro_series=MACRO_CELL_LS,
        )
    elif ftype == "highpass":
        cells = _build_ladder_cells(
            components,
            topology_type=topology_type,
            series_type="C",
            shunt_type="L",
            macro_a=MACRO_CELL_CS_LS_A,
            macro_b=MACRO_CELL_CS_LS,
            macro_ab=MACRO_CELL_CS_LS_AB,
            macro_series=MACRO_CELL_CS,
        )
    elif ftype == "bandpass":
        cells = _build_bandpass_cells(components, topology_type=topology_type)
    elif ftype == "bandstop":
        cells = _build_bandstop_cells(components, topology_type=topology_type)
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}")
    return _group_cells_by_macro(cells)

# ---- Encoding (components -> tokens) ----


def components_to_dsl_tokens(
    components: Sequence[ComponentSpec],
    *,
    macro_name: str = MACRO_CELL_LS_CS,
    segments: Sequence[Tuple[str, Sequence[Sequence[float]]]] | None = None,
    use_varint_k: bool = False,
    use_cell_indices: bool = False,
    include_bos: bool = True,
    include_order: bool = False,
    order: int | None = None,
    allow_incomplete: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Convert components (or explicit macro segments) into DSL tokens.
    Returns (tokens, slot_values) where slot_values aligns to tokens (nan when not a slot).
    - segments: optional explicit [(macro_name, [cell_vals...]), ...]; if provided, components are ignored.
    - use_varint_k: encode K with <K> D_* </K> to allow large/unbounded repeat counts.
    - use_cell_indices: emit <CELL_IDX_i> after <CELL> to stabilize long-K decoding.
    - include_order: prepend <ORDER_k> after <BOS> when True (requires order).
    - allow_incomplete: if False, missing slot values raise instead of emitting <VAL_NONE>.
    """
    def _encode_k(k: int, tokens_out: List[str], slot_out: List[float]) -> None:
        # Prefer varint when requested or k exceeds frozen set.
        if use_varint_k or k > len(K_TOKENS):
            tokens_out.append(K_VAR_START)
            slot_out.append(float("nan"))
            for d in str(max(0, int(k))):
                tok = f"<D_{d}>"
                tokens_out.append(tok)
                slot_out.append(float("nan"))
            tokens_out.append(K_VAR_END)
            slot_out.append(float("nan"))
        else:
            tokens_out.append(f"<K_{k}>")
            slot_out.append(float("nan"))

    def _build_bs_cells(comps: Sequence[ComponentSpec]) -> List[List[float]]:
        # Identify series connections (non-gnd) and derive the main path.
        shunt_series, branch_comp_ids = _detect_shunt_series_lc(comps)
        series_pairs: Dict[Tuple[str, str], List[ComponentSpec]] = {}
        for idx, c in enumerate(comps):
            if idx in branch_comp_ids:
                continue
            if c.role != "series":
                continue
            if c.node1 == "gnd" or c.node2 == "gnd":
                continue
            key = tuple(sorted((c.node1, c.node2)))
            series_pairs.setdefault(key, []).append(c)
        series_adj: Dict[str, List[str]] = {}
        for (n1, n2) in series_pairs.keys():
            series_adj.setdefault(n1, []).append(n2)
            series_adj.setdefault(n2, []).append(n1)
        if "in" not in series_adj or "out" not in series_adj:
            return []
        path_nodes = ["in"]
        prev = None
        current = "in"
        while current != "out":
            neighbors = [n for n in series_adj.get(current, []) if n != prev]
            if len(neighbors) != 1:
                return []
            nxt = neighbors[0]
            path_nodes.append(nxt)
            prev, current = current, nxt

        vals_per_cell: List[List[float]] = []
        for i in range(len(path_nodes) - 1):
            a = path_nodes[i]
            b = path_nodes[i + 1]
            key = tuple(sorted((a, b)))
            Lp = float("nan")
            Cp = float("nan")
            for c in series_pairs.get(key, []):
                if c.ctype == "L":
                    Lp = float(c.value_si)
                elif c.ctype == "C":
                    Cp = float(c.value_si)
            Ls, Cs = shunt_series.get(b, (float("nan"), float("nan")))
            vals_per_cell.append([Lp, Cp, Ls, Cs])
        return vals_per_cell

    def _build_bp_cells(comps: Sequence[ComponentSpec]) -> List[List[float]] | None:
        # Build bandpass cells: series LC between main nodes + parallel LC to gnd at each main node.
        # Optional notch: series LC to gnd detected via _detect_shunt_series_lc.
        shunt_series, branch_comp_ids = _detect_shunt_series_lc(comps)

        series_pairs: Dict[Tuple[str, str], List[ComponentSpec]] = {}
        for idx, c in enumerate(comps):
            if idx in branch_comp_ids:
                # exclude shunt-series branches (notch) from main path
                continue
            if c.role != "series":
                continue
            if c.node1 == "gnd" or c.node2 == "gnd":
                continue
            key = tuple(sorted((c.node1, c.node2)))
            series_pairs.setdefault(key, []).append(c)

        series_adj: Dict[str, List[str]] = {}
        for (n1, n2) in series_pairs.keys():
            series_adj.setdefault(n1, []).append(n2)
            series_adj.setdefault(n2, []).append(n1)
        if "in" not in series_adj or "out" not in series_adj:
            return None
        path_nodes = ["in"]
        prev = None
        current = "in"
        while current != "out":
            neighbors = [n for n in series_adj.get(current, []) if n != prev]
            if len(neighbors) != 1:
                return None
            nxt = neighbors[0]
            path_nodes.append(nxt)
            prev, current = current, nxt

        # Expect two series components per cell (L + C), so edges should be even count.
        if (len(path_nodes) - 1) % 2 != 0:
            return None

        # Collect shunt LC pairs to gnd at each node.
        shunt_map: Dict[str, Dict[str, float]] = {}
        for c in comps:
            if c.role != "shunt":
                continue
            node = c.node1 if c.node2 == "gnd" else c.node2
            d = shunt_map.setdefault(node, {})
            d[c.ctype] = float(c.value_si)

        vals_per_cell: List[List[float]] = []
        for i in range(0, len(path_nodes) - 2, 2):
            a = path_nodes[i]
            mid = path_nodes[i + 1]
            b = path_nodes[i + 2]
            key1 = tuple(sorted((a, mid)))
            key2 = tuple(sorted((mid, b)))
            comps_pair = series_pairs.get(key1, []) + series_pairs.get(key2, [])
            Ls = next((float(c.value_si) for c in comps_pair if c.ctype == "L"), float("nan"))
            Cs = next((float(c.value_si) for c in comps_pair if c.ctype == "C"), float("nan"))
            if not (_is_valid_value(Ls) and _is_valid_value(Cs)):
                return None

            sh = shunt_map.get(b, {})
            Lp = sh.get("L", float("nan"))
            Cp = sh.get("C", float("nan"))

            Ln, Cn = shunt_series.get(b, (float("nan"), float("nan")))
            vals_per_cell.append([Ls, Cs, Lp, Cp, Ln, Cn])
        return vals_per_cell

    def _emit_slot(slot_type: str, val: float):
        if slot_type not in SLOT_TYPE_TO_TOKEN:
            raise ValueError(f"Unknown slot type '{slot_type}'")
        if _is_valid_value(val):
            tokens.append(SLOT_TYPE_TO_TOKEN[slot_type])
            slot_values.append(float(val))
        else:
            if not allow_incomplete:
                raise ValueError(f"Missing value for slot '{slot_type}'")
            tokens.append(VAL_NONE)
            slot_values.append(float("nan"))

    tokens: List[str] = []
    if include_bos:
        tokens.append(BOS)
    if include_order:
        if order is None:
            raise ValueError("include_order=True requires order to be provided.")
        tokens.append(f"<ORDER_{int(order)}>")
    tokens.extend([MAIN_START, PORT_IN, PORT_OUT, PORT_GND])
    slot_values: List[float] = [float("nan")] * len(tokens)

    def _emit_repeat(macro_id: str, cell_vals: Sequence[Sequence[float]]):
        macro = MACRO_LIBRARY[macro_id]
        k = max(1, len(cell_vals))
        tokens.extend([REPEAT_START])
        slot_values.append(float("nan"))
        _encode_k(k, tokens, slot_values)
        tokens.extend([CASCADE, CALL, macro_id])
        slot_values.extend([float("nan")] * 3)
        for idx_cell in range(k):
            vals = list(cell_vals[idx_cell]) if idx_cell < len(cell_vals) else [float("nan")] * len(macro.slot_types)
            vals = (vals + [float("nan")] * len(macro.slot_types))[: len(macro.slot_types)]
            tokens.append(CELL)
            slot_values.append(float("nan"))
            if use_cell_indices and idx_cell < len(CELL_INDEX_TOKENS):
                tokens.append(CELL_INDEX_TOKENS[idx_cell])
                slot_values.append(float("nan"))
            for slot_idx, slot_type in enumerate(macro.slot_types):
                _emit_slot(slot_type, vals[slot_idx])
        tokens.extend([REPEAT_END])
        slot_values.append(float("nan"))

    def _emit_call(macro_id: str, vals: Sequence[float]):
        macro = MACRO_LIBRARY[macro_id]
        tokens.extend([CALL, macro_id])
        slot_values.extend([float("nan")] * 2)
        vs = list(vals) + [float("nan")] * len(macro.slot_types)
        vs = vs[: len(macro.slot_types)]
        for slot_idx, slot_type in enumerate(macro.slot_types):
            _emit_slot(slot_type, vs[slot_idx])

    if segments is not None:
        for macro_id, cell_vals in segments:
            cell_vals = list(cell_vals)
            if len(cell_vals) <= 1:
                _emit_call(macro_id, cell_vals[0] if cell_vals else [])
            else:
                _emit_repeat(macro_id, cell_vals)
    else:
        if macro_name == MACRO_CELL_BS_PAR_SER_LC:
            vals_per_cell = _build_bs_cells(components)
            if not vals_per_cell:
                raise ValueError("Bandstop structure parse failed.")
            _emit_repeat(macro_name, vals_per_cell)
        elif macro_name == MACRO_CELL_BP_SER_PAR_LC:
            vals_per_cell = _build_bp_cells(components)
            if not vals_per_cell:
                raise ValueError("Bandpass structure parse failed.")
            _emit_repeat(macro_name, vals_per_cell)
        else:
            macro = MACRO_LIBRARY[macro_name]
            # Heuristic: derive K from number of series components along main path (non-gnd).
            series = [c for c in components if c.role == "series" and c.node1 != "gnd" and c.node2 != "gnd"]
            K = max(1, len(series))
            # Collect slot values per cell from components: greedy by series/shunt ordering.
            vals_per_cell = []
            shunt_map: Dict[str, Dict[str, float]] = {}
            for c in components:
                if c.role != "shunt":
                    continue
                node = c.node1 if c.node2 == "gnd" else c.node2
                d = shunt_map.setdefault(node, {})
                d[c.ctype] = float(c.value_si)
            for c in series:
                vals = []
                for t in macro.slot_types:
                    if t == "L":
                        if c.ctype == "L":
                            vals.append(float(c.value_si))
                        else:
                            sh = shunt_map.get(c.node2, {})
                            vals.append(float(sh.get("L", float("nan"))))
                    elif t == "C":
                        if c.ctype == "C":
                            vals.append(float(c.value_si))
                        else:
                            sh = shunt_map.get(c.node2, {})
                            vals.append(float(sh.get("C", float("nan"))))
                    else:
                        vals.append(float("nan"))
                vals_per_cell.append(vals)

            while len(vals_per_cell) < K:
                vals_per_cell.append([float("nan")] * len(macro.slot_types))
            vals_per_cell = vals_per_cell[:K]

            _emit_repeat(macro_name, vals_per_cell)

    tokens.append(MAIN_END)
    slot_values.append(float("nan"))
    tokens.append(EOS)
    slot_values.append(float("nan"))
    return tokens, slot_values


# ---- Decoding (tokens -> components) ----


def dsl_tokens_to_components(
    tokens: Sequence[str],
    *,
    slot_values: Sequence[float] | None = None,
) -> List[ComponentSpec]:
    """
    Parse DSL tokens into ComponentSpec list. slot_values should align with tokens;
    if absent, slots are filled with 0.0.
    """
    toks = list(tokens)
    vals = list(slot_values) if slot_values is not None else [float("nan")] * len(toks)
    idx = 0

    def _next():
        nonlocal idx
        tok = toks[idx] if idx < len(toks) else None
        val = vals[idx] if idx < len(vals) else float("nan")
        idx += 1
        return tok, val

    tok, _ = _next()
    if tok == BOS:
        tok, _ = _next()
    while tok is not None and tok.startswith("<ORDER_"):
        tok, _ = _next()
    if tok != MAIN_START:
        return []
    for expected in (PORT_IN, PORT_OUT, PORT_GND):
        tok, _ = _next()
        if tok != expected:
            return []

    segments: List[Tuple[str, List[float]]] = []
    while True:
        tok, _ = _next()
        if tok is None or tok == MAIN_END:
            break
        if tok == REPEAT_START:
            tok_k, _ = _next()
            if tok_k == K_VAR_START:
                digits: List[str] = []
                while True:
                    tok_digit, _ = _next()
                    if tok_digit is None:
                        return []
                    if tok_digit == K_VAR_END:
                        break
                    if tok_digit not in DIGIT_TOKENS:
                        return []
                    digits.append(tok_digit.removeprefix("<D_").removesuffix(">"))
                try:
                    k_val = max(1, int("".join(digits))) if digits else 1
                except Exception:
                    return []
            elif tok_k in K_TOKENS:
                try:
                    k_val = int(tok_k.removeprefix("<K_").removesuffix(">"))
                except Exception:
                    return []
            else:
                return []
            tok_c, _ = _next()
            if tok_c != CASCADE:
                return []
            tok_call, _ = _next()
            if tok_call != CALL:
                return []
            tok_macro, _ = _next()
            if tok_macro not in MACRO_LIBRARY:
                return []
            macro = MACRO_LIBRARY[tok_macro]
            # New canonical form: each cell starts with <CELL>, then the macro's slot tokens.
            # Backward compatible: allow the older form without <CELL> boundaries.
            use_cell_tokens = (idx < len(toks) and toks[idx] == CELL)
            for _ in range(k_val):
                if use_cell_tokens:
                    tok_cell, _ = _next()
                    if tok_cell != CELL:
                        return []
                    # optional cell index markers
                    while idx < len(toks) and toks[idx] in CELL_INDEX_TOKENS:
                        _next()
                vals_for_macro: List[float] = []
                for slot_type in macro.slot_types:
                    tok_slot, v_slot = _next()
                    while tok_slot in CELL_INDEX_TOKENS:
                        tok_slot, v_slot = _next()
                    expected_tok = SLOT_TYPE_TO_TOKEN.get(slot_type)
                    if tok_slot == VAL_NONE:
                        vals_for_macro.append(float("nan"))
                    elif expected_tok is None or tok_slot != expected_tok:
                        return []
                    else:
                        vals_for_macro.append(float(v_slot) if v_slot == v_slot else float("nan"))
                segments.append((tok_macro, vals_for_macro))
            tok_end, _ = _next()
            if tok_end != REPEAT_END:
                return []
        elif tok == CALL:
            tok_macro, _ = _next()
            if tok_macro not in MACRO_LIBRARY:
                return []
            macro = MACRO_LIBRARY[tok_macro]
            vals_for_macro: List[float] = []
            for slot_type in macro.slot_types:
                tok_slot, v_slot = _next()
                while tok_slot in CELL_INDEX_TOKENS:
                    tok_slot, v_slot = _next()
                expected_tok = SLOT_TYPE_TO_TOKEN.get(slot_type)
                if tok_slot == VAL_NONE:
                    vals_for_macro.append(float("nan"))
                elif expected_tok is None or tok_slot != expected_tok:
                    return []
                else:
                    vals_for_macro.append(float(v_slot) if v_slot == v_slot else float("nan"))
            segments.append((tok_macro, vals_for_macro))
        else:
            return []

    if not segments:
        return []

    comps: List[ComponentSpec] = []
    nodes: List[str] = ["in"]
    last_end = "in"
    for i, (macro_name, vals_for_macro) in enumerate(segments):
        a = nodes[-1]
        b = "out" if i == len(segments) - 1 else f"n{i+1}"
        if b != "out":
            nodes.append(b)
        macro = MACRO_LIBRARY[macro_name]
        comps.extend(macro.expand_fn(a, b, "gnd", vals_for_macro, i))
        last_end = b
    # ensure last node renamed to out if not already
    if last_end != "out":
        for idx_c, c in enumerate(comps):
            n1 = "out" if c.node1 == last_end else c.node1
            n2 = "out" if c.node2 == last_end else c.node2
            comps[idx_c] = ComponentSpec(c.ctype, c.role, c.value_si, c.std_label, n1, n2)
    return comps


# ---- Grammar mask ----


def make_dsl_prefix_allowed_tokens_fn(tokenizer) -> Callable[[int, List[int]], List[int]]:
    vocab = tokenizer.get_vocab()
    all_ids = list(vocab.values())

    def _id(tok: str) -> int | None:
        return vocab.get(tok)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # precompute sets
    macro_ids = {_id(tok) for tok in MACRO_IDS if _id(tok) is not None}
    k_ids = {_id(tok) for tok in K_TOKENS if _id(tok) is not None}
    k_var_start_id = _id(K_VAR_START)
    k_var_end_id = _id(K_VAR_END)
    digit_ids = {_id(tok) for tok in DIGIT_TOKENS if _id(tok) is not None}
    cell_idx_ids = {_id(tok) for tok in CELL_INDEX_TOKENS if _id(tok) is not None}
    val_slot_ids = {_id(tok) for tok in VALUE_SLOTS if _id(tok) is not None}
    val_none_id = _id(VAL_NONE)
    order_ids = {tid for tok, tid in vocab.items() if tok.startswith("<ORDER_")}
    k_id_to_val = {_id(tok): int(tok.removeprefix("<K_").removesuffix(">")) for tok in K_TOKENS if _id(tok) is not None}
    macro_slots_len: Dict[int, int] = {}
    macro_slots_order: Dict[int, List[int]] = {}
    slot_type_to_id: Dict[str, int] = {}
    for slot_type, slot_tok in SLOT_TYPE_TO_TOKEN.items():
        tid = _id(slot_tok)
        if tid is not None:
            slot_type_to_id[slot_type] = tid
    for tok in MACRO_IDS:
        tid = _id(tok)
        if tid is None:
            continue
        macro_slots_len[tid] = len(MACRO_LIBRARY[tok].slot_types)
        ordered: List[int] = []
        for st in MACRO_LIBRARY[tok].slot_types:
            tok_id = slot_type_to_id.get(st)
            if tok_id is None:
                # Tokenizer vocab drift: fail open rather than silently making everything invalid.
                return lambda batch_id, input_ids: all_ids
            ordered.append(tok_id)
        macro_slots_order[tid] = ordered

    main_start = _id(MAIN_START)
    main_end = _id(MAIN_END)
    repeat_start = _id(REPEAT_START)
    repeat_end = _id(REPEAT_END)
    cascade_id = _id(CASCADE)
    call_id = _id(CALL)
    cell_id = _id(CELL)
    port_order = [_id(PORT_IN), _id(PORT_OUT), _id(PORT_GND)]
    bos_id = _id(BOS)
    explicit_eos_id = _id(EOS)

    required = [main_start, main_end, repeat_start, repeat_end, cascade_id, call_id, cell_id]
    if any(r is None for r in required) or not macro_ids or not k_ids:
        return lambda batch_id, input_ids: all_ids

    S_START = 0
    S_AFTER_BOS = 1
    S_PORTS = 2
    S_BODY = 3
    S_IN_REPEAT = 4
    S_AFTER_K = 5
    S_AFTER_CASCADE = 6
    S_AFTER_CALL = 7
    S_IN_SLOTS = 8
    S_EXPECT_REPEAT_END = 9
    S_AFTER_MAIN_END = 10
    S_AFTER_EXPLICIT_EOS = 11
    S_IN_K_VAR = 13
    S_DONE = 14

    def _state_from_ids(input_ids: List[int]) -> Tuple[int, int, int, int | None, bool, bool, int, int, bool, bool, bool]:
        state = S_START
        slot_needed = 0
        slot_pos = 0
        current_macro_tid: int | None = None
        valid = True
        port_idx = 0
        in_repeat = False
        current_k = 1
        has_segment = False
        in_k_var = False
        k_digits_seen = False
        for tid in input_ids:
            if tid in special_ids:
                continue
            if tid in cell_idx_ids:
                continue
            if state == S_START:
                if tid == bos_id:
                    state = S_AFTER_BOS
                    continue
                if tid in order_ids:
                    continue
                if tid == main_start:
                    state = S_PORTS
                    continue
                valid = False
                break
            if state == S_AFTER_BOS:
                if tid in order_ids:
                    continue
                if tid == main_start:
                    state = S_PORTS
                    continue
                valid = False
                break
            if state == S_PORTS:
                expected = port_order[port_idx] if 0 <= port_idx < len(port_order) else None
                if expected is not None and tid == expected:
                    port_idx += 1
                    if port_idx >= len(port_order):
                        state = S_BODY
                    continue
                valid = False
                break
            if state == S_BODY:
                if tid == repeat_start:
                    state = S_IN_REPEAT
                    in_repeat = True
                    has_segment = True
                    continue
                if tid == call_id:
                    state = S_AFTER_CALL
                    has_segment = True
                    continue
                if tid == main_end:
                    state = S_AFTER_MAIN_END
                    continue
                valid = False
                break
            if state == S_IN_REPEAT:
                if tid in k_ids:
                    current_k = k_id_to_val.get(tid, 1)
                    state = S_AFTER_K
                    continue
                if tid == k_var_start_id:
                    in_k_var = True
                    k_digits_seen = False
                    current_k = 0
                    state = S_IN_K_VAR
                    continue
                valid = False
                break
            if state == S_IN_K_VAR:
                if tid in digit_ids:
                    k_digits_seen = True
                    digit = 0
                    for k, v in vocab.items():
                        if v == tid and k.startswith("<D_"):
                            try:
                                digit = int(k.removeprefix("<D_").removesuffix(">"))
                            except Exception:
                                digit = 0
                            break
                    current_k = current_k * 10 + digit
                    continue
                if tid == k_var_end_id and k_digits_seen:
                    state = S_AFTER_K
                    in_k_var = False
                    continue
                valid = False
                break
            if state == S_AFTER_K:
                if tid == cascade_id:
                    state = S_AFTER_CASCADE
                    continue
                valid = False
                break
            if state == S_AFTER_CASCADE:
                if tid == call_id:
                    state = S_AFTER_CALL
                    continue
                valid = False
                break
            if state == S_AFTER_CALL:
                if tid in macro_ids:
                    macro_len = macro_slots_len.get(tid, 0)
                    rep = current_k if in_repeat else 1
                    # Repeat slots are grouped as: (<CELL> + slot_types) * K
                    slot_needed = (1 + macro_len) * rep if in_repeat else macro_len
                    slot_pos = 0
                    current_macro_tid = tid
                    state = S_IN_SLOTS if slot_needed > 0 else (S_EXPECT_REPEAT_END if in_repeat else S_BODY)
                    continue
                valid = False
                break
            if state == S_IN_SLOTS:
                expected_slots = macro_slots_order.get(current_macro_tid, [])
                macro_len = len(expected_slots)
                if in_repeat:
                    group_len = 1 + macro_len
                    pos_in_group = slot_pos % group_len
                    expected_tok = cell_id if pos_in_group == 0 else (expected_slots[pos_in_group - 1] if expected_slots else None)
                else:
                    expected_tok = expected_slots[slot_pos] if 0 <= slot_pos < macro_len else None
                if expected_tok is not None and (tid == expected_tok or (expected_tok != cell_id and tid == val_none_id)):
                    slot_needed -= 1
                    slot_pos += 1
                    if slot_needed <= 0:
                        state = S_EXPECT_REPEAT_END if in_repeat else S_BODY
                    continue
                valid = False
                break
            if state == S_EXPECT_REPEAT_END:
                if tid == repeat_end:
                    if slot_needed <= 0:
                        state = S_BODY
                        in_repeat = False
                        current_k = 1
                    continue
                valid = False
                break
            if state == S_AFTER_MAIN_END:
                if explicit_eos_id is not None and tid == explicit_eos_id:
                    state = S_AFTER_EXPLICIT_EOS
                    continue
                if eos_id is not None and tid == eos_id:
                    state = S_DONE
                    continue
                valid = False
                break
            if state == S_AFTER_EXPLICIT_EOS:
                if eos_id is not None and tid == eos_id:
                    state = S_DONE
                    continue
                valid = False
                break
            if state == S_DONE:
                continue
        return state, slot_needed, port_idx, current_macro_tid, valid, in_repeat, current_k, slot_pos, has_segment, in_k_var, k_digits_seen

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        (
            state,
            slot_needed,
            port_idx,
            current_macro_tid,
            valid,
            in_repeat,
            current_k,
            slot_pos,
            has_segment,
            in_k_var,
            k_digits_seen,
        ) = _state_from_ids(ids)
        if not valid:
            return all_ids
        allowed: Set[int] = set()
        if state == S_START:
            if main_start is not None:
                allowed.add(main_start)
            if bos_id is not None:
                allowed.add(bos_id)
            allowed.update(order_ids)
        elif state == S_AFTER_BOS:
            if main_start is not None:
                allowed.add(main_start)
            allowed.update(order_ids)
        elif state == S_PORTS:
            expected = port_order[port_idx] if 0 <= port_idx < len(port_order) else None
            if expected is not None:
                allowed.add(expected)
        elif state == S_BODY:
            if repeat_start is not None:
                allowed.add(repeat_start)
            if call_id is not None:
                allowed.add(call_id)
            if has_segment and main_end is not None:
                allowed.add(main_end)
        elif state == S_IN_REPEAT:
            allowed.update(k_ids)
            if k_var_start_id is not None:
                allowed.add(k_var_start_id)
        elif state == S_IN_K_VAR:
            if digit_ids:
                allowed.update(digit_ids)
            if k_digits_seen and k_var_end_id is not None:
                allowed.add(k_var_end_id)
        elif state == S_AFTER_K:
            if cascade_id is not None:
                allowed.add(cascade_id)
        elif state == S_AFTER_CASCADE:
            if call_id is not None:
                allowed.add(call_id)
        elif state == S_AFTER_CALL:
            allowed.update(macro_ids)
        elif state == S_IN_SLOTS:
            expected_slots = macro_slots_order.get(current_macro_tid, [])
            macro_len = len(expected_slots)
            if slot_needed <= 0:
                expected_tok = None
            elif in_repeat:
                group_len = 1 + macro_len
                pos_in_group = slot_pos % group_len
                expected_tok = cell_id if pos_in_group == 0 else (expected_slots[pos_in_group - 1] if expected_slots else None)
            else:
                expected_tok = expected_slots[slot_pos] if 0 <= slot_pos < macro_len else None
            if expected_tok is not None:
                allowed.add(expected_tok)
                if expected_tok != cell_id and val_none_id is not None:
                    allowed.add(val_none_id)
            if in_repeat and cell_idx_ids:
                allowed.update(cell_idx_ids)
        elif state == S_EXPECT_REPEAT_END:
            if repeat_end is not None:
                allowed.add(repeat_end)
        elif state == S_AFTER_MAIN_END:
            if explicit_eos_id is not None:
                allowed.add(explicit_eos_id)
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == S_AFTER_EXPLICIT_EOS:
            if eos_id is not None:
                allowed.add(eos_id)
        elif state == S_DONE:
            if eos_id is not None:
                allowed.add(eos_id)
        if pad_id is not None:
            allowed.add(pad_id)
        if not allowed:
            return all_ids
        return sorted(allowed)

    return _prefix_allowed_tokens_fn


def dsl_tokens_to_action_tokens(
    tokens: Sequence[str],
    *,
    slot_values: Sequence[float] | None = None,
) -> List[str]:
    """
    Convenience: DSL tokens (+slots) -> ComponentSpec -> Action tokens.
    Useful for dual-IR supervision or post-hoc compilation to ActionVACT.
    """
    comps = dsl_tokens_to_components(tokens, slot_values=slot_values)
    return components_to_action_tokens(
        [
            ComponentSpec(c.ctype, c.role, c.value_si, c.std_label, c.node1, c.node2)
            for c in comps
        ]
    )
