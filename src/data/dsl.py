"""
DSL: ML-friendly IR with Macro / Repeat / Typed numeric slots.

Key ideas:
- Canonical main program: <MAIN> Ports Body </MAIN>
- Body supports structured cascade repeat blocks with frozen macro library.
- Numeric slots are typed (<VAL_L>/<VAL_C>/...), paired with continuous heads.
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

# Motif macros (atomic / branch-level)
MACRO_SER_L = "<MAC_SER_L>"
MACRO_SER_C = "<MAC_SER_C>"
MACRO_SHUNT_L = "<MAC_SHUNT_L>"
MACRO_SHUNT_C = "<MAC_SHUNT_C>"
MACRO_SER_RESO = "<MAC_SER_RESO>"
MACRO_SER_TANK = "<MAC_SER_TANK>"
MACRO_SHUNT_RESO = "<MAC_SHUNT_RESO>"
MACRO_SHUNT_NOTCH = "<MAC_SHUNT_NOTCH>"

MACRO_IDS = [
    MACRO_SER_L,
    MACRO_SER_C,
    MACRO_SHUNT_L,
    MACRO_SHUNT_C,
    MACRO_SER_RESO,
    MACRO_SER_TANK,
    MACRO_SHUNT_RESO,
    MACRO_SHUNT_NOTCH,
]

SERIES_MACROS = {
    MACRO_SER_L,
    MACRO_SER_C,
    MACRO_SER_RESO,
    MACRO_SER_TANK,
}
SHUNT_MACROS = {
    MACRO_SHUNT_L,
    MACRO_SHUNT_C,
    MACRO_SHUNT_RESO,
    MACRO_SHUNT_NOTCH,
}


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


def _expand_ser_l(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, b)
    return comps


def _expand_ser_c(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    return comps


def _expand_shunt_l(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "shunt", L_val, a, gnd)
    return comps


def _expand_shunt_c(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    C_val = (vals + [0.0])[0]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "C", "shunt", C_val, a, gnd)
    return comps


def _expand_ser_reso(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, x)
    _maybe_add_component(comps, "C", "series", C_val, x, b)
    return comps


def _expand_ser_tank(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, b)
    _maybe_add_component(comps, "C", "series", C_val, a, b)
    return comps


def _expand_shunt_reso(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "shunt", L_val, a, gnd)
    _maybe_add_component(comps, "C", "shunt", C_val, a, gnd)
    return comps


def _expand_shunt_notch(a: str, b: str, gnd: str, vals: List[float], inst_idx: int = 0) -> List[ComponentSpec]:
    L_val, C_val = (vals + [0.0, 0.0])[:2]
    x = f"x{inst_idx}_0"
    comps: List[ComponentSpec] = []
    _maybe_add_component(comps, "L", "series", L_val, a, x)
    _maybe_add_component(comps, "C", "series", C_val, x, gnd)
    return comps


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
    MACRO_SER_L: MacroDef(MACRO_SER_L, ("L",), _expand_ser_l),
    MACRO_SER_C: MacroDef(MACRO_SER_C, ("C",), _expand_ser_c),
    MACRO_SHUNT_L: MacroDef(MACRO_SHUNT_L, ("L",), _expand_shunt_l),
    MACRO_SHUNT_C: MacroDef(MACRO_SHUNT_C, ("C",), _expand_shunt_c),
    MACRO_SER_RESO: MacroDef(MACRO_SER_RESO, ("L", "C"), _expand_ser_reso),
    MACRO_SER_TANK: MacroDef(MACRO_SER_TANK, ("L", "C"), _expand_ser_tank),
    MACRO_SHUNT_RESO: MacroDef(MACRO_SHUNT_RESO, ("L", "C"), _expand_shunt_reso),
    MACRO_SHUNT_NOTCH: MacroDef(MACRO_SHUNT_NOTCH, ("L", "C"), _expand_shunt_notch),
}

# Slot type -> token mapping (kept centralized to avoid drift across encoder/decoder/mask).
SLOT_TYPE_TO_TOKEN = {
    "L": VAL_L,
    "C": VAL_C,
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


def _other_node(comp: ComponentSpec, node: str) -> str:
    return comp.node2 if comp.node1 == node else comp.node1


def _collect_shunt_series_lc_branches(
    comps: Sequence[ComponentSpec],
) -> Tuple[Dict[str, List[Tuple[float, float]]], Set[int]]:
    node_to_comps: Dict[str, List[int]] = {}
    for idx, c in enumerate(comps):
        for node in (c.node1, c.node2):
            if node == "gnd":
                continue
            node_to_comps.setdefault(node, []).append(idx)

    branches_by_anchor: Dict[str, List[Tuple[float, float]]] = {}
    branch_ids: Set[int] = set()
    for node, idxs in node_to_comps.items():
        if node in ("in", "out"):
            continue
        if len(idxs) != 2:
            continue
        i1, i2 = idxs
        c1, c2 = comps[i1], comps[i2]
        if c1.role != "series" or c2.role != "series":
            continue
        o1 = _other_node(c1, node)
        o2 = _other_node(c2, node)
        c1_to_gnd = o1 == "gnd"
        c2_to_gnd = o2 == "gnd"
        if c1_to_gnd == c2_to_gnd:
            continue
        comp_to_gnd = c1 if c1_to_gnd else c2
        comp_to_anchor = c2 if c1_to_gnd else c1
        anchor = _other_node(comp_to_anchor, node)
        if anchor == "gnd":
            continue
        types = {comp_to_gnd.ctype, comp_to_anchor.ctype}
        if types != {"L", "C"}:
            continue
        L_val = float(comp_to_gnd.value_si if comp_to_gnd.ctype == "L" else comp_to_anchor.value_si)
        C_val = float(comp_to_gnd.value_si if comp_to_gnd.ctype == "C" else comp_to_anchor.value_si)
        branches_by_anchor.setdefault(anchor, []).append((L_val, C_val))
        branch_ids.update([i1, i2])
    return branches_by_anchor, branch_ids


def _components_to_motif_cells(components: Sequence[ComponentSpec]) -> List[Tuple[str, List[float]]]:
    comps = list(components)
    if not comps:
        return []

    branches_by_anchor, branch_comp_ids = _collect_shunt_series_lc_branches(comps)

    series_pairs: Dict[Tuple[str, str], List[int]] = {}
    for idx, c in enumerate(comps):
        if idx in branch_comp_ids:
            continue
        if c.role != "series":
            continue
        if c.node1 == "gnd" or c.node2 == "gnd":
            continue
        key = tuple(sorted((c.node1, c.node2)))
        series_pairs.setdefault(key, []).append(idx)

    parallel_series: Dict[Tuple[str, str], Tuple[int, int]] = {}
    parallel_ids: Set[int] = set()
    for key, idxs in series_pairs.items():
        if len(idxs) != 2:
            continue
        types = {comps[i].ctype for i in idxs}
        if types != {"L", "C"}:
            continue
        l_idx = idxs[0] if comps[idxs[0]].ctype == "L" else idxs[1]
        c_idx = idxs[0] if comps[idxs[0]].ctype == "C" else idxs[1]
        parallel_series[key] = (l_idx, c_idx)
        parallel_ids.update(idxs)

    series_edges: List[Tuple[str, str, str, List[int]]] = []
    series_adj: Dict[str, List[Tuple[str, int]]] = {}

    def _add_edge(n1: str, n2: str, kind: str, comp_indices: List[int]) -> None:
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
        if c.node1 == "gnd" or c.node2 == "gnd":
            continue
        _add_edge(c.node1, c.node2, "single", [idx])

    if "in" not in series_adj or "out" not in series_adj:
        raise ValueError("Main path parse failed (missing in/out).")

    path_nodes: List[str] = ["in"]
    path_series: List[int] = []
    prev = None
    current = "in"
    visited_edges: Set[int] = set()
    while current != "out":
        neighbors = [item for item in series_adj.get(current, []) if item[0] != prev]
        if len(neighbors) != 1:
            raise ValueError("Main path parse failed (ambiguous path).")
        next_node, edge_id = neighbors[0]
        if edge_id in visited_edges:
            raise ValueError("Main path parse failed (cycle detected).")
        visited_edges.add(edge_id)
        path_series.append(edge_id)
        path_nodes.append(next_node)
        prev, current = current, next_node

    shunt_by_node: Dict[str, Dict[str, float]] = {}
    for idx, c in enumerate(comps):
        if idx in branch_comp_ids:
            continue
        if c.role != "shunt":
            continue
        anchor = c.node2 if c.node1 == "gnd" else c.node1
        if anchor == "gnd":
            continue
        entry = shunt_by_node.setdefault(anchor, {})
        if c.ctype not in entry:
            entry[c.ctype] = float(c.value_si)

    cells: List[Tuple[str, List[float]]] = []
    node_idx = 0
    while node_idx < len(path_nodes):
        node = path_nodes[node_idx]
        shunt_vals = shunt_by_node.get(node, {})
        has_L = _is_valid_value(shunt_vals.get("L"))
        has_C = _is_valid_value(shunt_vals.get("C"))
        if has_L and has_C:
            cells.append((MACRO_SHUNT_RESO, [shunt_vals["L"], shunt_vals["C"]]))
        elif has_L:
            cells.append((MACRO_SHUNT_L, [shunt_vals["L"]]))
        elif has_C:
            cells.append((MACRO_SHUNT_C, [shunt_vals["C"]]))

        for L_val, C_val in branches_by_anchor.get(node, []):
            cells.append((MACRO_SHUNT_NOTCH, [L_val, C_val]))

        if node_idx >= len(path_series):
            break

        edge_id = path_series[node_idx]
        _, _, kind, comp_indices = series_edges[edge_id]
        if kind == "parallel_lc":
            l_idx, c_idx = comp_indices
            L_val = float(comps[l_idx].value_si)
            C_val = float(comps[c_idx].value_si)
            cells.append((MACRO_SER_TANK, [L_val, C_val]))
            node_idx += 1
            continue

        comp = comps[comp_indices[0]]
        if node_idx + 1 < len(path_series):
            mid_node = path_nodes[node_idx + 1]
            next_edge_id = path_series[node_idx + 1]
            _, _, next_kind, next_indices = series_edges[next_edge_id]
            if (
                next_kind == "single"
                and mid_node not in shunt_by_node
                and mid_node not in branches_by_anchor
            ):
                comp2 = comps[next_indices[0]]
                types = {comp.ctype, comp2.ctype}
                if types == {"L", "C"}:
                    L_val = float(comp.value_si if comp.ctype == "L" else comp2.value_si)
                    C_val = float(comp.value_si if comp.ctype == "C" else comp2.value_si)
                    cells.append((MACRO_SER_RESO, [L_val, C_val]))
                    node_idx += 2
                    continue

        if comp.ctype == "L":
            cells.append((MACRO_SER_L, [float(comp.value_si)]))
        elif comp.ctype == "C":
            cells.append((MACRO_SER_C, [float(comp.value_si)]))
        node_idx += 1

    return cells


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
    del filter_type, topology_type
    cells = _components_to_motif_cells(components)
    return _group_cells_by_macro(cells)

# ---- Encoding (components -> tokens) ----


def components_to_dsl_tokens(
    components: Sequence[ComponentSpec],
    *,
    macro_name: str = MACRO_SER_L,
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
    del macro_name

    def _encode_k(k: int, tokens_out: List[str], slot_out: List[float]) -> None:
        if use_varint_k or k > len(K_TOKENS):
            tokens_out.append(K_VAR_START)
            slot_out.append(float("nan"))
            for d in str(max(0, int(k))):
                tokens_out.append(f"<D_{d}>")
                slot_out.append(float("nan"))
            tokens_out.append(K_VAR_END)
            slot_out.append(float("nan"))
        else:
            tokens_out.append(f"<K_{k}>")
            slot_out.append(float("nan"))

    def _emit_value(slot_type: str, value: float, tokens_out: List[str], slot_out: List[float]) -> None:
        if not _is_valid_value(value):
            if allow_incomplete:
                tokens_out.append(VAL_NONE)
                slot_out.append(float("nan"))
                return
            raise ValueError("Missing slot value while allow_incomplete=False.")
        tok = SLOT_TYPE_TO_TOKEN.get(slot_type)
        if tok is None:
            raise ValueError(f"Unknown slot type: {slot_type}")
        tokens_out.append(tok)
        slot_out.append(float(value))

    def _emit_repeat(macro_id: str, vals_per_cell: Sequence[Sequence[float]]) -> None:
        macro = MACRO_LIBRARY[macro_id]
        tokens.extend([REPEAT_START])
        slot_values.append(float("nan"))
        _encode_k(len(vals_per_cell), tokens, slot_values)
        tokens.extend([CASCADE, CALL, macro_id])
        slot_values.extend([float("nan")] * 3)
        for idx_cell, vals in enumerate(vals_per_cell):
            tokens.append(CELL)
            slot_values.append(float("nan"))
            if use_cell_indices and idx_cell < len(CELL_INDEX_TOKENS):
                tokens.append(CELL_INDEX_TOKENS[idx_cell])
                slot_values.append(float("nan"))
            vals_list = list(vals) + [float("nan")] * len(macro.slot_types)
            vals_list = vals_list[: len(macro.slot_types)]
            for slot_type, value in zip(macro.slot_types, vals_list):
                _emit_value(slot_type, value, tokens, slot_values)
        tokens.append(REPEAT_END)
        slot_values.append(float("nan"))

    def _emit_call(macro_id: str, vals: Sequence[float]) -> None:
        macro = MACRO_LIBRARY[macro_id]
        tokens.extend([CALL, macro_id])
        slot_values.extend([float("nan")] * 2)
        vals_list = list(vals) + [float("nan")] * len(macro.slot_types)
        vals_list = vals_list[: len(macro.slot_types)]
        for slot_type, value in zip(macro.slot_types, vals_list):
            _emit_value(slot_type, value, tokens, slot_values)

    tokens: List[str] = []
    slot_values: List[float] = []
    if include_bos:
        tokens.append(BOS)
        slot_values.append(float("nan"))
    if include_order:
        if order is None:
            raise ValueError("include_order=True requires order.")
        tokens.append(f"<ORDER_{int(order)}>")
        slot_values.append(float("nan"))
    tokens.append(MAIN_START)
    slot_values.append(float("nan"))
    tokens.extend([PORT_IN, PORT_OUT, PORT_GND])
    slot_values.extend([float("nan")] * 3)

    if segments is None:
        cells = _components_to_motif_cells(components)
        segments = _group_cells_by_macro(cells)

    for macro_id, cell_vals in segments:
        if macro_id not in MACRO_LIBRARY:
            raise ValueError(f"Unknown macro: {macro_id}")
        vals_per_cell = list(cell_vals) if cell_vals else []
        if not vals_per_cell:
            vals_per_cell = [[float("nan")] * len(MACRO_LIBRARY[macro_id].slot_types)]
        if len(vals_per_cell) == 1:
            _emit_call(macro_id, vals_per_cell[0])
        else:
            _emit_repeat(macro_id, vals_per_cell)

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

    series_indices = [i for i, (macro_name, _) in enumerate(segments) if macro_name in SERIES_MACROS]
    last_series_idx = series_indices[-1] if series_indices else None

    comps: List[ComponentSpec] = []
    current = "in"
    node_idx = 0
    for i, (macro_name, vals_for_macro) in enumerate(segments):
        macro = MACRO_LIBRARY[macro_name]
        if macro_name in SERIES_MACROS:
            if last_series_idx is not None and i == last_series_idx:
                next_node = "out"
            else:
                node_idx += 1
                next_node = f"n{node_idx}"
            comps.extend(macro.expand_fn(current, next_node, "gnd", vals_for_macro, i))
            current = next_node
        else:
            comps.extend(macro.expand_fn(current, current, "gnd", vals_for_macro, i))
    return comps


# ---- Macro parsing helpers ----


def dsl_tokens_to_macro_sequence(tokens: Sequence[str], *, strict: bool = True) -> List[str]:
    """
    Parse DSL tokens into an expanded macro sequence (one macro per cell).
    Raises on malformed sequences when strict=True.
    """
    toks = list(tokens)
    idx = 0

    def _peek() -> str | None:
        return toks[idx] if idx < len(toks) else None

    def _next() -> str | None:
        nonlocal idx
        tok = toks[idx] if idx < len(toks) else None
        idx += 1
        return tok

    def _expect(expected: str) -> bool:
        tok = _next()
        if tok != expected:
            if strict:
                raise ValueError(f"DSL parse error: expected {expected}, got {tok}")
            return False
        return True

    def _parse_k() -> int:
        tok_k = _next()
        if tok_k is None:
            if strict:
                raise ValueError("DSL parse error: missing K token in REPEAT.")
            return 1
        if tok_k == K_VAR_START:
            digits: List[str] = []
            while True:
                tok_digit = _next()
                if tok_digit is None:
                    if strict:
                        raise ValueError("DSL parse error: unterminated <K>...</K>.")
                    return 1
                if tok_digit == K_VAR_END:
                    break
                if tok_digit not in DIGIT_TOKENS:
                    if strict:
                        raise ValueError(f"DSL parse error: invalid digit token {tok_digit}.")
                    return 1
                digits.append(tok_digit.removeprefix("<D_").removesuffix(">"))
            if not digits:
                return 1
            try:
                return max(1, int("".join(digits)))
            except Exception:
                if strict:
                    raise ValueError("DSL parse error: invalid K varint.")
                return 1
        if tok_k in K_TOKENS:
            try:
                return max(1, int(tok_k.removeprefix("<K_").removesuffix(">")))
            except Exception:
                if strict:
                    raise ValueError(f"DSL parse error: invalid K token {tok_k}.")
                return 1
        if strict:
            raise ValueError(f"DSL parse error: invalid K token {tok_k}.")
        return 1

    # Skip BOS / ORDER tokens if present.
    tok = _peek()
    if tok == BOS:
        _next()
        tok = _peek()
    while tok is not None and tok.startswith("<ORDER_"):
        _next()
        tok = _peek()

    if not _expect(MAIN_START):
        return []
    for expected in (PORT_IN, PORT_OUT, PORT_GND):
        if not _expect(expected):
            return []

    macros: List[str] = []
    while True:
        tok = _peek()
        if tok is None:
            if strict:
                raise ValueError("DSL parse error: missing </MAIN>.")
            break
        if tok == MAIN_END:
            _next()
            break
        tok = _next()
        if tok == REPEAT_START:
            k_val = _parse_k()
            if not _expect(CASCADE):
                return []
            if not _expect(CALL):
                return []
            macro = _next()
            if macro not in MACRO_LIBRARY:
                if strict:
                    raise ValueError(f"DSL parse error: unknown macro {macro}.")
                return []
            macro_def = MACRO_LIBRARY[macro]
            slot_types = macro_def.slot_types
            # Detect canonical <CELL> boundaries if present.
            use_cell_tokens = _peek() == CELL
            for _ in range(k_val):
                if use_cell_tokens:
                    if not _expect(CELL):
                        return []
                    while _peek() in CELL_INDEX_TOKENS:
                        _next()
                for slot_type in slot_types:
                    tok_slot = _next()
                    while tok_slot in CELL_INDEX_TOKENS:
                        tok_slot = _next()
                    expected = SLOT_TYPE_TO_TOKEN.get(slot_type)
                    if tok_slot == VAL_NONE:
                        pass
                    elif expected is None or tok_slot != expected:
                        if strict:
                            raise ValueError(f"DSL parse error: expected {expected}, got {tok_slot}.")
                        return []
                macros.append(macro)
            if not _expect(REPEAT_END):
                return []
        elif tok == CALL:
            macro = _next()
            if macro not in MACRO_LIBRARY:
                if strict:
                    raise ValueError(f"DSL parse error: unknown macro {macro}.")
                return []
            macro_def = MACRO_LIBRARY[macro]
            for slot_type in macro_def.slot_types:
                tok_slot = _next()
                while tok_slot in CELL_INDEX_TOKENS:
                    tok_slot = _next()
                expected = SLOT_TYPE_TO_TOKEN.get(slot_type)
                if tok_slot == VAL_NONE:
                    pass
                elif expected is None or tok_slot != expected:
                    if strict:
                        raise ValueError(f"DSL parse error: expected {expected}, got {tok_slot}.")
                    return []
            macros.append(macro)
        else:
            if strict:
                raise ValueError(f"DSL parse error: unexpected token {tok}.")
            return []

    return macros


def count_cells_from_dsl_tokens(tokens: Sequence[str], *, strict: bool = True) -> int:
    """
    Count total cells in a DSL program by expanding macro repeats.
    """
    return len(dsl_tokens_to_macro_sequence(tokens, strict=strict))


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
