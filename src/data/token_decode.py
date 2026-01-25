from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Sequence, Tuple

from .action_codec import action_tokens_to_components
from .vact_struct import vact_struct_tokens_to_components
from .dsl import dsl_tokens_to_components
from .sfci_net_codec import sfci_net_tokens_to_components
from .vact_codec import vact_tokens_to_components


ReprKind = Literal["vact", "vact_struct", "sfci", "action", "dsl", "dsl_value"]


def build_label_value_map(tokenizer) -> Dict[str, float]:
    from . import quantization

    vocab = tokenizer.get_vocab()
    mp: Dict[str, float] = {}
    for tok in vocab.keys():
        if tok.startswith("<VAL_"):
            label = tok.replace("<VAL_", "").replace(">", "")
            try:
                mp[label] = float(quantization.label_to_value(label))
            except Exception:
                continue
    return mp


def decode_components_from_token_ids(
    token_ids: Sequence[int],
    tokenizer,
    *,
    repr_kind: ReprKind,
    label_to_value: Mapping[str, float] | None = None,
    slot_values: Sequence[float] | None = None,
) -> Tuple[list, List[str]]:
    repr_kind = str(repr_kind)
    ids = list(token_ids)
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    if repr_kind == "vact":
        comps = vact_tokens_to_components(tokens, label_to_value=label_to_value, drop_non_component_tokens=True)
        return comps, tokens
    if repr_kind == "vact_struct":
        comps = vact_struct_tokens_to_components(tokens, label_to_value=label_to_value, drop_non_component_tokens=True)
        return comps, tokens
    if repr_kind == "sfci":
        if slot_values is not None:
            if len(slot_values) != len(ids):
                raise ValueError(f"slot_values must align with token_ids: {len(slot_values)} != {len(ids)}")
            full_tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            filtered_tokens: List[str] = []
            filtered_values: List[float] = []
            for tid, tok, v in zip(ids, full_tokens, slot_values):
                if int(tid) in special_ids:
                    continue
                filtered_tokens.append(tok)
                filtered_values.append(float(v))
            comps = sfci_net_tokens_to_components(filtered_tokens, label_to_value=label_to_value, slot_values=filtered_values)
            return comps, filtered_tokens
        comps = sfci_net_tokens_to_components(tokens, label_to_value=label_to_value)
        return comps, tokens
    if repr_kind in ("dsl", "dsl_value"):
        if slot_values is not None:
            if len(slot_values) != len(ids):
                raise ValueError(f"slot_values must align with token_ids: {len(slot_values)} != {len(ids)}")
            full_tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            filtered_tokens: List[str] = []
            filtered_values: List[float] = []
            for tid, tok, v in zip(ids, full_tokens, slot_values):
                if int(tid) in special_ids:
                    continue
                filtered_tokens.append(tok)
                filtered_values.append(float(v))
            comps = dsl_tokens_to_components(filtered_tokens, slot_values=filtered_values)
            return comps, filtered_tokens
        comps = dsl_tokens_to_components(tokens, slot_values=None)
        return comps, tokens
    if repr_kind == "action":
        comps = action_tokens_to_components(tokens, label_to_value=label_to_value)
        return comps, tokens
    raise ValueError(f"Unknown repr_kind: {repr_kind}")
