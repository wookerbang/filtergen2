from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.quantization import generate_value_labels  # noqa: E402
from src.data.vact_codec import build_vact_vocab  # noqa: E402
from src.data.vact_struct import build_vact_struct_vocab  # noqa: E402
from src.data.sfci_net_codec import build_sfci_net_vocab  # noqa: E402
from src.data.action_codec import build_action_vocab  # noqa: E402
from src.data.dsl import build_dsl_vocab  # noqa: E402


SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"]


def _dedup_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def build_wordlevel_tokenizer(tokens: List[str], save_dir: Path) -> None:
    tokens = _dedup_preserve_order(tokens)
    vocab = {tok: i for i, tok in enumerate(tokens)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    fast_tok.save_pretrained(save_dir)
    with open(save_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tokenizers (VACT/VACT-Struct/DSL/SFCI/Action).")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/tokenizers"), help="Output root directory.")
    p.add_argument("--series", type=str, default="E24", help="E-series for value labels (E12/E24).")
    p.add_argument("--exp-min", type=int, default=-12, help="Minimum exponent for value labels.")
    p.add_argument("--exp-max", type=int, default=3, help="Maximum exponent for value labels.")
    p.add_argument("--order-min", type=int, default=2, help="Minimum filter order for <ORDER_k> tokens.")
    p.add_argument("--order-max", type=int, default=9, help="Maximum filter order for <ORDER_k> tokens.")
    p.add_argument("--max-nodes", type=int, default=32, help="Number of intermediate nodes to include (n0..n{max}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    # Value labels for L and C
    val_labels = generate_value_labels(series=args.series, kind="L", exp_min=args.exp_min, exp_max=args.exp_max)
    val_labels += generate_value_labels(series=args.series, kind="C", exp_min=args.exp_min, exp_max=args.exp_max)

    node_names = ("in", "out", "gnd") + tuple(f"n{k}" for k in range(args.max_nodes))
    order_range = (args.order_min, args.order_max)

    # VACT vocab
    vact_vocab = SPECIAL_TOKENS + build_vact_vocab(
        value_labels=val_labels,
        node_names=node_names,
        order_range=order_range,
    )
    build_wordlevel_tokenizer(vact_vocab, out_dir / "vact_tokenizer")

    # VACT-Struct vocab (superset)
    vact_struct_vocab = SPECIAL_TOKENS + build_vact_struct_vocab(
        value_labels=val_labels,
        node_names=node_names,
        order_range=order_range,
    )
    build_wordlevel_tokenizer(vact_struct_vocab, out_dir / "vact_struct_tokenizer")

    # DSL vocab (macro/repeat/slots)
    dsl_vocab = SPECIAL_TOKENS + build_dsl_vocab(order_range=order_range)
    build_wordlevel_tokenizer(dsl_vocab, out_dir / "dsl_tokenizer")

    # SFCI net-centric vocab
    sfci_vocab = SPECIAL_TOKENS + build_sfci_net_vocab(
        value_labels=val_labels,
        node_names=node_names,
        order_range=order_range,
    )
    build_wordlevel_tokenizer(sfci_vocab, out_dir / "sfci_tokenizer")

    # Action vocab
    action_vocab = SPECIAL_TOKENS + build_action_vocab(value_labels=val_labels, node_names=node_names)
    build_wordlevel_tokenizer(action_vocab, out_dir / "action_tokenizer")

    print(f"Saved VACT tokenizer to {out_dir / 'vact_tokenizer'}")
    print(f"Saved VACT-Struct tokenizer to {out_dir / 'vact_struct_tokenizer'}")
    print(f"Saved DSL tokenizer to {out_dir / 'dsl_tokenizer'}")
    print(f"Saved SFCI tokenizer to {out_dir / 'sfci_tokenizer'}")
    print(f"Saved Action tokenizer to {out_dir / 'action_tokenizer'}")


if __name__ == "__main__":
    main()
