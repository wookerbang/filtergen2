"""
Baseline: net-centric SFCI style encoding (LaMAGIC-like) for comparison.

Each node is emitted with its incident devices (type-local IDs) to mirror
the hyperedge-list spirit of SFCI, while keeping parsing simple for LC ladders.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Mapping, Sequence, Set, Tuple

from .schema import ComponentSpec
from .dsl import VAL_NONE, VALUE_SLOTS


def _node_order(node: str) -> int:
    if node == "in":
        return 0
    if node.startswith("n"):
        try:
            return 1 + int(node[1:])
        except ValueError:
            return 1_000
    if node == "out":
        return 10_000
    if node == "gnd":
        return 20_000
    return 30_000


def components_to_sfci_net_tokens(
    components: List[ComponentSpec],
    *,
    value_mode: Literal["discrete", "none", "continuous"] = "discrete",
) -> List[str]:
    """
    Net-centric SFCI-like encoding:
      <NODE_x> (<DEV_L0> <ROLE_SERIES> <VAL_label> <PEER_y>)* <ENDNODE>
    Components are listed on both incident nodes; IDs are type-local.
    """
    # assign deterministic IDs per ctype
    type_counters: Dict[str, int] = {}
    comp_meta: List[Tuple[ComponentSpec, int]] = []
    for comp in sorted(components, key=lambda c: (c.ctype, _node_order(c.node1), _node_order(c.node2))):
        idx = type_counters.get(comp.ctype, 0)
        type_counters[comp.ctype] = idx + 1
        comp_meta.append((comp, idx))

    # build adjacency
    node_map: Dict[str, List[Tuple[str, str, str, str, int, float]]] = {}
    for comp, idx in comp_meta:
        if value_mode == "none":
            label = "NONE"
        elif value_mode == "continuous":
            label = "L" if comp.ctype == "L" else "C"
        elif value_mode == "discrete":
            label = comp.std_label or "NA"
        else:
            raise ValueError(f"Unknown value_mode: {value_mode}")
        for node, peer in [(comp.node1, comp.node2), (comp.node2, comp.node1)]:
            value = float(comp.value_si) if value_mode == "continuous" else float("nan")
            node_map.setdefault(node, []).append((comp.ctype, comp.role, label, peer, idx, value))

    tokens: List[str] = []
    for node in sorted(node_map.keys(), key=_node_order):
        tokens.append(f"<NODE_{node}>")
        # order devices on node: type, id
        for ctype, role, label, peer, idx, _ in sorted(node_map[node], key=lambda x: (x[0], x[4])):
            tokens.extend(
                [
                    f"<DEV_{ctype}{idx}>",
                    f"<ROLE_{role.upper()}>",
                    f"<VAL_{label}>",
                    f"<PEER_{peer}>",
                ]
            )
        tokens.append("<ENDNODE>")
    return tokens


def components_to_sfci_net_tokens_and_values(
    components: List[ComponentSpec],
    *,
    value_mode: Literal["continuous", "none", "discrete"] = "continuous",
) -> Tuple[List[str], List[float]]:
    """
    Return SFCI tokens + aligned value targets for continuous value prediction.
    """
    # assign deterministic IDs per ctype
    type_counters: Dict[str, int] = {}
    comp_meta: List[Tuple[ComponentSpec, int]] = []
    for comp in sorted(components, key=lambda c: (c.ctype, _node_order(c.node1), _node_order(c.node2))):
        idx = type_counters.get(comp.ctype, 0)
        type_counters[comp.ctype] = idx + 1
        comp_meta.append((comp, idx))

    node_map: Dict[str, List[Tuple[str, str, str, str, int, float]]] = {}
    for comp, idx in comp_meta:
        if value_mode == "none":
            label = "NONE"
            value = float("nan")
        elif value_mode == "continuous":
            label = "L" if comp.ctype == "L" else "C"
            value = float(comp.value_si)
        elif value_mode == "discrete":
            label = comp.std_label or "NA"
            value = float("nan")
        else:
            raise ValueError(f"Unknown value_mode: {value_mode}")
        for node, peer in [(comp.node1, comp.node2), (comp.node2, comp.node1)]:
            node_map.setdefault(node, []).append((comp.ctype, comp.role, label, peer, idx, value))

    tokens: List[str] = []
    values: List[float] = []
    for node in sorted(node_map.keys(), key=_node_order):
        tokens.append(f"<NODE_{node}>")
        values.append(float("nan"))
        for ctype, role, label, peer, idx, value in sorted(node_map[node], key=lambda x: (x[0], x[4])):
            tokens.extend(
                [
                    f"<DEV_{ctype}{idx}>",
                    f"<ROLE_{role.upper()}>",
                    f"<VAL_{label}>",
                    f"<PEER_{peer}>",
                ]
            )
            values.extend([float("nan"), float("nan"), value, float("nan")])
        tokens.append("<ENDNODE>")
        values.append(float("nan"))
    return tokens, values


def _default_value_for_type(ctype: str) -> float:
    if ctype == "L":
        return 1e-9
    if ctype == "C":
        return 1e-12
    return 1.0


def sfci_net_tokens_to_components(
    tokens: List[str],
    label_to_value: Mapping[str, float] | None = None,
    slot_values: Sequence[float] | None = None,
) -> List[ComponentSpec]:
    """
    Decode net-centric tokens back to components.
    Components are reconstructed once per undirected pair (node, peer, ctype, id).
    """
    comps: List[ComponentSpec] = []
    i = 0
    seen: Set[Tuple[str, str, str, int]] = set()
    current_node = None
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("<NODE_"):
            current_node = tok.replace("<NODE_", "").replace(">", "")
            i += 1
            continue
        if tok == "<ENDNODE>":
            current_node = None
            i += 1
            continue
        if tok.startswith("<DEV_") and current_node is not None:
            if i + 3 >= len(tokens):
                break
            dev = tok.replace("<DEV_", "").replace(">", "")
            # split type+id
            ctype = "".join([ch for ch in dev if not ch.isdigit()])
            try:
                dev_id = int(dev[len(ctype) :])
            except ValueError:
                dev_id = 0
            role = tokens[i + 1].replace("<ROLE_", "").replace(">", "").lower()
            val = tokens[i + 2].replace("<VAL_", "").replace(">", "")
            peer = tokens[i + 3].replace("<PEER_", "").replace(">", "")
            key = tuple(sorted([current_node, peer]) + [ctype, dev_id])
            if key not in seen:
                seen.add(key)
                value = _default_value_for_type(ctype)
                if slot_values is not None and len(slot_values) == len(tokens):
                    v = slot_values[i + 2]
                    if v == v and v > 0.0:
                        value = float(v)
                elif val not in ("NA", "NONE", "L", "C"):
                    if label_to_value is not None:
                        value = float(label_to_value.get(val, value))
                    else:
                        try:
                            from .quantization import label_to_value as _label_to_value
                            value = float(_label_to_value(val))
                        except Exception:
                            value = value
                comps.append(
                    ComponentSpec(
                        ctype=ctype,
                        role=role,
                        value_si=value,
                        std_label=None if val in ("NA", "NONE", "L", "C") else val,
                        node1=current_node,
                        node2=peer,
                    )
                )
            i += 4
        else:
            i += 1
    return comps


def build_sfci_net_vocab(
    value_labels: Sequence[str],
    max_id_per_type: Mapping[str, int] = {"L": 16, "C": 16},
    node_names: Sequence[str] = ("in", "out", "gnd") + tuple(f"n{k}" for k in range(16)),
    order_range: Tuple[int, int] | None = (2, 7),
) -> List[str]:
    """
    构建 net-centric 词表：节点 token、设备 token、角色、数值、peer、ID。
    """
    vocab: Set[str] = set()
    vocab.update(["<ENDNODE>", "<SEP>"])
    if order_range:
        lo, hi = order_range
        for k in range(lo, hi + 1):
            vocab.add(f"<ORDER_{k}>")
    for node in node_names:
        vocab.add(f"<NODE_{node}>")
        vocab.add(f"<PEER_{node}>")
    for ctype, max_id in max_id_per_type.items():
        for i in range(int(max_id) + 1):
            vocab.add(f"<DEV_{ctype}{i}>")
    vocab.update([f"<ROLE_SERIES>", f"<ROLE_SHUNT>"])
    vocab.add(VAL_NONE)
    vocab.update(VALUE_SLOTS)
    for label in value_labels:
        vocab.add(f"<VAL_{label}>")
    return sorted(vocab)
