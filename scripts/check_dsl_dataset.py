#!/usr/bin/env python
"""Check DSL motif extraction quality and token statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dsl import (  # noqa: E402
    VAL_NONE,
    MACRO_LIBRARY,
    dsl_tokens_to_components,
    dsl_tokens_to_macro_sequence,
    components_to_dsl_tokens,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DSL dataset sanity checks (motif DSL).")
    p.add_argument("--data", type=Path, required=True, help="Path to jsonl dataset.")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    missing_dsl = 0
    parse_fail = 0
    roundtrip_fail = 0
    val_none_samples = 0
    val_none_tokens = 0
    macro_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()

    with open(args.data, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tokens = obj.get("dsl_tokens")
            slots = obj.get("dsl_slot_values")
            total += 1
            if tokens is None or slots is None:
                missing_dsl += 1
                continue
            if VAL_NONE in tokens:
                val_none_samples += 1
                val_none_tokens += sum(1 for t in tokens if t == VAL_NONE)
            try:
                macros = dsl_tokens_to_macro_sequence(tokens, strict=True)
            except Exception:
                parse_fail += 1
                continue
            length_counter[len(macros)] += 1
            macro_counter.update(macros)
            try:
                comps = dsl_tokens_to_components(tokens, slot_values=slots)
                new_tokens, _ = components_to_dsl_tokens(comps, include_order=False, allow_incomplete=True)
                new_macros = dsl_tokens_to_macro_sequence(new_tokens, strict=True)
                if macros != new_macros:
                    roundtrip_fail += 1
            except Exception:
                roundtrip_fail += 1

            if args.max_samples and total >= int(args.max_samples):
                break

    print("DSL dataset check")
    print(f"total_samples={total}")
    print(f"missing_dsl={missing_dsl}")
    print(f"parse_fail={parse_fail}")
    print(f"roundtrip_fail={roundtrip_fail}")
    print(f"val_none_samples={val_none_samples}")
    print(f"val_none_tokens={val_none_tokens}")
    if length_counter:
        lengths = sorted(length_counter.items())
        print("length_hist=" + ", ".join(f"{k}:{v}" for k, v in lengths))
    if macro_counter:
        top = macro_counter.most_common(12)
        print("top_macros=" + ", ".join(f"{k}:{v}" for k, v in top))


if __name__ == "__main__":
    main()
