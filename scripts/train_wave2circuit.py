from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.torch_dataset import FilterDesignDataset
from src.data import quantization
from src.data.vact_codec import CELL_TOKEN, ORDER_PREFIX, SEP_TOKEN
from src.data.vact_struct import (
    CIRCUIT_END,
    CIRCUIT_START,
    CELL_END,
    PORT_IN,
    PORT_OUT,
    SERIES_BLOCK_END,
    SERIES_BLOCK_START,
    SHUNT_BLOCK_END,
    SHUNT_BLOCK_START,
    Z0_50,
)
from src.data.dsl import MACRO_LIBRARY, VALUE_SLOTS, make_dsl_prefix_allowed_tokens_fn, make_macro_ir_prefix_allowed_tokens_fn
from src.models import VACTT5


def make_collate_fn(tokenizer, use_repr: str):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    vocab = tokenizer.get_vocab()
    use_value_targets = use_repr not in ("dsl_value", "macro_ir")
    id_to_value = [float("nan")] * len(vocab)
    if use_value_targets:
        for tok, tid in vocab.items():
            if tok.startswith("<VAL_"):
                try:
                    id_to_value[int(tid)] = float(quantization.label_to_value(tok.replace("<VAL_", "").replace(">", "")))
                except Exception:
                    continue

    def collate(batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        waves = torch.stack([b["wave"] for b in batch])  # (B, C, L)
        scalars = torch.stack([b["scalar"] for b in batch])
        filter_type = scalars[:, 0].long()
        fc_hz = scalars[:, 1]

        if use_repr == "vact":
            tokens_key = "vact_tokens"
        elif use_repr == "vact_struct":
            tokens_key = "vact_struct_tokens"
        elif use_repr == "dsl":
            tokens_key = "dsl_tokens"
        elif use_repr == "dsl_value":
            tokens_key = "dsl_tokens"
        elif use_repr == "macro_ir":
            tokens_key = "macro_ir_tokens"
        else:
            tokens_key = "sfci_tokens"
        seqs = []
        value_lists = []
        for b in batch:
            seq = list(b.get(tokens_key) or b.get("input_ids"))
            if not seq or seq[-1] != eos_id:
                seq.append(eos_id)
            seqs.append(seq)
            if use_repr == "macro_ir":
                value_lists.append(b.get("macro_slot_targets"))
            else:
                value_lists.append(b.get("value_targets"))
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        value_targets = None
        macro_slot_targets = None
        if use_value_targets:
            value_targets = torch.full((len(batch), max_len), float("nan"), dtype=torch.float32)
        if use_repr == "macro_ir":
            slot_count = 0
            for b in batch:
                mts = b.get("macro_slot_targets") or []
                if mts:
                    slot_count = len(mts[0])
                    break
            if slot_count > 0:
                macro_slot_targets = torch.full((len(batch), max_len, slot_count), float("nan"), dtype=torch.float32)
        for i, seq in enumerate(seqs):
            l = len(seq)
            input_ids[i, :l] = torch.tensor(seq, dtype=torch.long)
            if use_value_targets:
                vt_list = value_lists[i] if value_lists[i] is not None else None
                for t, tid in enumerate(seq):
                    if t >= max_len:
                        break
                    if vt_list is not None and t < len(vt_list) and vt_list[t] == vt_list[t]:
                        value_targets[i, t] = float(vt_list[t])
                    elif 0 <= tid < len(id_to_value) and not (id_to_value[tid] != id_to_value[tid]):
                        value_targets[i, t] = float(id_to_value[tid])
        labels = input_ids.clone()
        labels[labels == pad_id] = -100  # ignore pad in loss
        batch_out = {
            "wave": waves,
            "filter_type": filter_type,
            "fc_hz": fc_hz,
            "labels": labels,
        }
        if use_value_targets:
            batch_out["value_targets"] = value_targets
        if use_repr == "macro_ir" and macro_slot_targets is not None:
            for i, mts in enumerate(value_lists):
                if not mts:
                    continue
                mlen = min(len(mts), max_len)
                for t in range(mlen):
                    row = mts[t]
                    if row is None:
                        continue
                    macro_slot_targets[i, t, : len(row)] = torch.tensor(row, dtype=torch.float32)
            batch_out["macro_slot_targets"] = macro_slot_targets
        return batch_out

    return collate


def _shift_right(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Minimal T5-style shift_right used for constrained train-time masking.
    """
    decoder_input_ids = labels.new_full(labels.shape, pad_id)
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids[:, 0] = pad_id
    decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, pad_id)
    return decoder_input_ids


def build_train_time_grammar_masker(tokenizer, *, repr_kind: str):
    """
    Returns a function:
      mask_logits(logits, labels) -> masked_logits
    where logits are masked according to the grammar induced by repr_kind.
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

    type_ids = {vocab[tok] for tok in ("<L>", "<C>") if tok in vocab}
    val_ids = {tid for tok, tid in vocab.items() if tok.startswith("<VAL_")}
    node_ids = {tid for tok, tid in vocab.items() if tok.startswith("<NODE_")}
    order_ids = {tid for tok, tid in vocab.items() if tok.startswith(ORDER_PREFIX)}
    sep_id = vocab.get(SEP_TOKEN)

    role_series_id = vocab.get("<SERIES>")
    role_shunt_id = vocab.get("<SHUNT>")
    role_ids = {x for x in (role_series_id, role_shunt_id) if x is not None}

    def _mask_vact(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dec_in = _shift_right(labels, pad_id)
        masked = logits.clone()
        B, L, V = masked.shape
        if V != vocab_size:
            # tokenizer/model vocab mismatch; fail open
            return masked
        for b in range(B):
            state = 0  # 0:type,1:role,2:val,3:n1,4:n2
            started = False
            for t in range(L):
                tid_prev = int(dec_in[b, t].item())
                if tid_prev in special_ids:
                    pass
                else:
                    if state == 0:
                        if not started and tid_prev in order_ids:
                            pass
                        elif not started and sep_id is not None and tid_prev == sep_id:
                            pass
                        elif tid_prev == vocab.get(CELL_TOKEN):
                            pass
                        elif tid_prev in type_ids:
                            state = 1
                            started = True
                        else:
                            # invalid prefix -> stop masking (fail open for remaining positions)
                            break
                    elif state == 1:
                        if tid_prev in role_ids:
                            state = 2
                        else:
                            break
                    elif state == 2:
                        if tid_prev in val_ids:
                            state = 3
                        else:
                            break
                    elif state == 3:
                        if tid_prev in node_ids:
                            state = 4
                        else:
                            break
                    elif state == 4:
                        if tid_prev in node_ids:
                            state = 0
                        else:
                            break

                if int(labels[b, t].item()) == -100:
                    continue

                allowed: List[int] = []
                if state == 0:
                    allowed = list(type_ids)
                    cell = vocab.get(CELL_TOKEN)
                    if cell is not None:
                        allowed.append(cell)
                    if not started:
                        allowed.extend(list(order_ids))
                        if sep_id is not None:
                            allowed.append(int(sep_id))
                    if eos_id is not None:
                        allowed.append(eos_id)
                elif state == 1:
                    allowed = list(role_ids)
                elif state == 2:
                    allowed = list(val_ids)
                else:
                    allowed = list(node_ids)

                allowed.append(pad_id)
                allow = torch.tensor(sorted(set(allowed)), dtype=torch.long, device=masked.device)
                dis = torch.ones((V,), dtype=torch.bool, device=masked.device)
                dis[allow] = False
                masked[b, t, dis] = -1e9
        return masked

    def _mask_vact_struct(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dec_in = _shift_right(labels, pad_id)
        masked = logits.clone()
        B, L, V = masked.shape
        if V != vocab_size:
            return masked

        # resolve structure token IDs once
        ids = {k: vocab.get(k) for k in [CIRCUIT_START, CIRCUIT_END, CELL_TOKEN, CELL_END, SERIES_BLOCK_START, SERIES_BLOCK_END, SHUNT_BLOCK_START, SHUNT_BLOCK_END, PORT_IN, PORT_OUT, Z0_50]}
        required = ["<CIRCUIT>", "</CIRCUIT>", "<CELL>", "</CELL>", "<SERIES_BLOCK>", "</SERIES_BLOCK>", "<SHUNT_BLOCK>", "</SHUNT_BLOCK>"]
        if any(ids.get(k) is None for k in required):
            return masked

        S_START = 0
        S_AFTER_CIRCUIT = 1
        S_AFTER_PORT_IN = 2
        S_AFTER_NODE_IN = 3
        S_AFTER_PORT_OUT = 4
        S_AFTER_NODE_OUT = 5
        S_BODY = 6
        S_AFTER_CELL = 7
        S_IN_SERIES_EXPECT_TYPE_OR_END = 8
        S_IN_SERIES_EXPECT_ROLE = 9
        S_IN_SERIES_EXPECT_VAL = 10
        S_IN_SERIES_EXPECT_N1 = 11
        S_IN_SERIES_EXPECT_N2 = 12
        S_AFTER_SERIES_END = 13
        S_IN_SHUNT_EXPECT_TYPE_OR_END = 14
        S_IN_SHUNT_EXPECT_ROLE = 15
        S_IN_SHUNT_EXPECT_VAL = 16
        S_IN_SHUNT_EXPECT_N1 = 17
        S_IN_SHUNT_EXPECT_N2 = 18
        S_AFTER_SHUNT_END = 19
        S_DONE = 20

        node_in_id = vocab.get("<NODE_in>")
        node_out_id = vocab.get("<NODE_out>")

        for b in range(B):
            state = S_START
            for t in range(L):
                tid_prev = int(dec_in[b, t].item())
                if tid_prev not in special_ids:
                    if state == S_START:
                        if tid_prev in order_ids:
                            pass
                        elif sep_id is not None and tid_prev == sep_id:
                            pass
                        elif tid_prev == ids[CIRCUIT_START]:
                            state = S_AFTER_CIRCUIT
                        else:
                            break
                    elif state == S_AFTER_CIRCUIT:
                        if tid_prev == ids[PORT_IN]:
                            state = S_AFTER_PORT_IN
                        elif tid_prev == ids[Z0_50]:
                            state = S_BODY
                        elif tid_prev == ids[CELL_TOKEN]:
                            state = S_AFTER_CELL
                        elif tid_prev == ids[CIRCUIT_END]:
                            state = S_DONE
                        else:
                            break
                    elif state == S_AFTER_PORT_IN:
                        if node_in_id is not None and tid_prev == node_in_id:
                            state = S_AFTER_NODE_IN
                        else:
                            break
                    elif state == S_AFTER_NODE_IN:
                        if tid_prev == ids[PORT_OUT]:
                            state = S_AFTER_PORT_OUT
                        else:
                            break
                    elif state == S_AFTER_PORT_OUT:
                        if node_out_id is not None and tid_prev == node_out_id:
                            state = S_AFTER_NODE_OUT
                        else:
                            break
                    elif state == S_AFTER_NODE_OUT:
                        if tid_prev == ids[Z0_50]:
                            state = S_BODY
                        elif tid_prev == ids[CELL_TOKEN]:
                            state = S_AFTER_CELL
                        elif tid_prev == ids[CIRCUIT_END]:
                            state = S_DONE
                        else:
                            break
                    elif state == S_BODY:
                        if tid_prev == ids[CELL_TOKEN]:
                            state = S_AFTER_CELL
                        elif tid_prev == ids[CIRCUIT_END]:
                            state = S_DONE
                        else:
                            break
                    elif state == S_AFTER_CELL:
                        if tid_prev == ids[SERIES_BLOCK_START]:
                            state = S_IN_SERIES_EXPECT_TYPE_OR_END
                        else:
                            break
                    elif state == S_IN_SERIES_EXPECT_TYPE_OR_END:
                        if tid_prev in type_ids:
                            state = S_IN_SERIES_EXPECT_ROLE
                        elif tid_prev == ids[SERIES_BLOCK_END]:
                            state = S_AFTER_SERIES_END
                        else:
                            break
                    elif state == S_IN_SERIES_EXPECT_ROLE:
                        if role_series_id is not None and tid_prev == role_series_id:
                            state = S_IN_SERIES_EXPECT_VAL
                        else:
                            break
                    elif state == S_IN_SERIES_EXPECT_VAL:
                        if tid_prev in val_ids:
                            state = S_IN_SERIES_EXPECT_N1
                        else:
                            break
                    elif state == S_IN_SERIES_EXPECT_N1:
                        if tid_prev in node_ids:
                            state = S_IN_SERIES_EXPECT_N2
                        else:
                            break
                    elif state == S_IN_SERIES_EXPECT_N2:
                        if tid_prev in node_ids:
                            state = S_IN_SERIES_EXPECT_TYPE_OR_END
                        else:
                            break
                    elif state == S_AFTER_SERIES_END:
                        if tid_prev == ids[SHUNT_BLOCK_START]:
                            state = S_IN_SHUNT_EXPECT_TYPE_OR_END
                        else:
                            break
                    elif state == S_IN_SHUNT_EXPECT_TYPE_OR_END:
                        if tid_prev in type_ids:
                            state = S_IN_SHUNT_EXPECT_ROLE
                        elif tid_prev == ids[SHUNT_BLOCK_END]:
                            state = S_AFTER_SHUNT_END
                        else:
                            break
                    elif state == S_IN_SHUNT_EXPECT_ROLE:
                        if role_shunt_id is not None and tid_prev == role_shunt_id:
                            state = S_IN_SHUNT_EXPECT_VAL
                        else:
                            break
                    elif state == S_IN_SHUNT_EXPECT_VAL:
                        if tid_prev in val_ids:
                            state = S_IN_SHUNT_EXPECT_N1
                        else:
                            break
                    elif state == S_IN_SHUNT_EXPECT_N1:
                        if tid_prev in node_ids:
                            state = S_IN_SHUNT_EXPECT_N2
                        else:
                            break
                    elif state == S_IN_SHUNT_EXPECT_N2:
                        if tid_prev in node_ids:
                            state = S_IN_SHUNT_EXPECT_TYPE_OR_END
                        else:
                            break
                    elif state == S_AFTER_SHUNT_END:
                        if tid_prev == ids[CELL_END]:
                            state = S_BODY
                        else:
                            break
                    elif state == S_DONE:
                        pass

                if int(labels[b, t].item()) == -100:
                    continue

                allowed: List[int] = []
                if state == S_START:
                    allowed.extend(list(order_ids))
                    if sep_id is not None:
                        allowed.append(int(sep_id))
                    allowed.append(int(ids[CIRCUIT_START]))
                elif state == S_AFTER_CIRCUIT:
                    allowed.extend([int(ids[PORT_IN]), int(ids[CELL_TOKEN]), int(ids[CIRCUIT_END]), int(ids[Z0_50])])
                elif state == S_AFTER_PORT_IN:
                    if node_in_id is not None:
                        allowed.append(int(node_in_id))
                elif state == S_AFTER_NODE_IN:
                    allowed.append(int(ids[PORT_OUT]))
                elif state == S_AFTER_PORT_OUT:
                    if node_out_id is not None:
                        allowed.append(int(node_out_id))
                elif state == S_AFTER_NODE_OUT:
                    allowed.extend([int(ids[Z0_50]), int(ids[CELL_TOKEN]), int(ids[CIRCUIT_END])])
                elif state == S_BODY:
                    allowed.extend([int(ids[CELL_TOKEN]), int(ids[CIRCUIT_END])])
                elif state == S_AFTER_CELL:
                    allowed.append(int(ids[SERIES_BLOCK_START]))
                elif state == S_IN_SERIES_EXPECT_TYPE_OR_END:
                    allowed.extend(list(type_ids))
                    allowed.append(int(ids[SERIES_BLOCK_END]))
                elif state == S_IN_SERIES_EXPECT_ROLE:
                    if role_series_id is not None:
                        allowed.append(int(role_series_id))
                elif state == S_IN_SERIES_EXPECT_VAL:
                    allowed.extend(list(val_ids))
                elif state in (S_IN_SERIES_EXPECT_N1, S_IN_SERIES_EXPECT_N2):
                    allowed.extend(list(node_ids))
                elif state == S_AFTER_SERIES_END:
                    allowed.append(int(ids[SHUNT_BLOCK_START]))
                elif state == S_IN_SHUNT_EXPECT_TYPE_OR_END:
                    allowed.extend(list(type_ids))
                    allowed.append(int(ids[SHUNT_BLOCK_END]))
                elif state == S_IN_SHUNT_EXPECT_ROLE:
                    if role_shunt_id is not None:
                        allowed.append(int(role_shunt_id))
                elif state == S_IN_SHUNT_EXPECT_VAL:
                    allowed.extend(list(val_ids))
                elif state in (S_IN_SHUNT_EXPECT_N1, S_IN_SHUNT_EXPECT_N2):
                    allowed.extend(list(node_ids))
                elif state == S_AFTER_SHUNT_END:
                    allowed.append(int(ids[CELL_END]))
                elif state == S_DONE:
                    # do not constrain once closed
                    continue

                allowed.append(pad_id)
                allow = torch.tensor(sorted(set(allowed)), dtype=torch.long, device=masked.device)
                dis = torch.ones((V,), dtype=torch.bool, device=masked.device)
                dis[allow] = False
                masked[b, t, dis] = -1e9

        return masked

    if repr_kind == "vact":
        return _mask_vact
    if repr_kind == "vact_struct":
        return _mask_vact_struct
    if repr_kind in ("dsl", "dsl_value"):
        prefix_allowed = make_dsl_prefix_allowed_tokens_fn(tokenizer)

        def _mask_dsl(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            dec_in = _shift_right(labels, pad_id)
            masked = logits.clone()
            B, L, V = masked.shape
            if V != vocab_size:
                return masked
            for b in range(B):
                for t in range(L):
                    if int(labels[b, t].item()) == -100:
                        continue
                    allowed = prefix_allowed(b, dec_in[b, : t + 1])
                    allow = torch.tensor(sorted(set(allowed)), dtype=torch.long, device=masked.device)
                    dis = torch.ones((V,), dtype=torch.bool, device=masked.device)
                    dis[allow] = False
                    masked[b, t, dis] = -1e9
            return masked

        return _mask_dsl
    if repr_kind == "macro_ir":
        prefix_allowed = make_macro_ir_prefix_allowed_tokens_fn(tokenizer)

        def _mask_macro_ir(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            dec_in = _shift_right(labels, pad_id)
            masked = logits.clone()
            B, L, V = masked.shape
            if V != vocab_size:
                return masked
            for b in range(B):
                for t in range(L):
                    if int(labels[b, t].item()) == -100:
                        continue
                    allowed = prefix_allowed(b, dec_in[b, : t + 1])
                    allow = torch.tensor(sorted(set(allowed)), dtype=torch.long, device=masked.device)
                    dis = torch.ones((V,), dtype=torch.bool, device=masked.device)
                    dis[allow] = False
                    masked[b, t, dis] = -1e9
            return masked

        return _mask_macro_ir
    return None


class Wave2CircuitTrainer(Trainer):
    """Custom Trainer that routes waveform/spec inputs into VACTT5."""

    def __init__(self, *args, grammar_masker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grammar_masker = grammar_masker

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(
            wave=inputs["wave"],
            filter_type=inputs["filter_type"],
            fc_hz=inputs["fc_hz"],
            labels=inputs["labels"],
            value_targets=inputs.get("value_targets"),
            macro_slot_targets=inputs.get("macro_slot_targets"),
        )
        if self.grammar_masker is None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            labels = inputs["labels"]
            masked_logits = self.grammar_masker(logits, labels)
            token_loss = F.cross_entropy(
                masked_logits.view(-1, masked_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            value_loss = getattr(outputs, "value_loss", None)
            if value_loss is not None:
                token_loss = token_loss + float(getattr(model, "value_loss_weight", 1.0)) * value_loss
            macro_loss = getattr(outputs, "macro_value_loss", None)
            if macro_loss is not None:
                token_loss = token_loss + float(getattr(model, "macro_value_loss_weight", 1.0)) * macro_loss
            loss = token_loss
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train waveform-conditioned circuit generator (VACT or SFCI).")
    p.add_argument("--data", type=Path, required=True, help="Path to train jsonl.")
    p.add_argument("--eval-data", type=Path, help="Optional path to eval jsonl for periodic eval.")
    p.add_argument("--tokenizer", type=str, required=True, help="Path or name of tokenizer.")
    p.add_argument("--output", type=Path, default=Path("checkpoints/wave2circuit"), help="Checkpoint dir.")
    p.add_argument("--t5-name", type=str, default="t5-base", help="HF model name, e.g., t5-small or t5-base.")
    p.add_argument("--batch-size", type=int, default=64, help="Per-device batch size.")
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--log-steps", type=int, default=50, help="Logging steps.")
    p.add_argument("--eval-steps", type=int, default=200, help="Eval steps when --eval-data is provided.")
    p.add_argument("--save-steps", type=int, default=500, help="Checkpoint save steps.")
    p.add_argument("--save-total-limit", type=int, default=3, help="Max checkpoints to keep.")
    p.add_argument(
        "--use-wave",
        choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"],
        default="real",
        help="Which waveform to use (S21-only options: ideal_s21 / real_s21).",
    )
    p.add_argument(
        "--mix-real-prob",
        type=float,
        default=0.3,
        help="When --use-wave mix, probability of picking real waveform (rest ideal).",
    )
    p.add_argument(
        "--freq-mode",
        choices=["none", "log_fc", "linear_fc", "log_f", "log_f_centered"],
        default="log_f_centered",
        help=(
            "Prepend frequency-position channel: "
            "none (no freq channel), "
            "log_fc (log10(f/fc)), "
            "linear_fc (f/fc), "
            "log_f (log10(f)), "
            "log_f_centered (log10(f) - mean(log10(f)))."
        ),
    )
    p.add_argument(
        "--freq-scale",
        choices=["none", "log_fc", "log_f_mean"],
        default="log_f_mean",
        help=(
            "Optional constant scale channel: "
            "none, "
            "log_fc (repeat log10(fc)), "
            "log_f_mean (repeat mean(log10(f)))."
        ),
    )
    p.add_argument(
        "--spec-mode",
        choices=["none", "type_fc"],
        default="type_fc",
        help="Spec token usage: none (wave-only) or type_fc (prepend filter type + fc token).",
    )
    p.add_argument(
        "--no-s11",
        dest="include_s11",
        action="store_false",
        help="Drop S11 channels from waveform input.",
    )
    p.set_defaults(include_s11=False)
    p.add_argument(
        "--repr",
        choices=["vact", "vact_struct", "dsl", "dsl_value", "macro_ir", "sfci", "action"],
        default="macro_ir",
        help="Which token sequence to train on.",
    )
    p.add_argument(
        "--grammar-mask",
        dest="grammar_mask",
        action="store_true",
        help="Apply FSM grammar mask during training (train/infer aligned).",
    )
    p.add_argument(
        "--no-grammar-mask",
        dest="grammar_mask",
        action="store_false",
        help="Disable FSM grammar mask during training.",
    )
    p.set_defaults(grammar_mask=True)
    p.add_argument("--wave-norm", action="store_true", help="Per-channel standardize waveforms for stability.")
    p.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 training if CUDA is available.")
    p.add_argument("--bf16", action="store_true", help="Use bf16 training if supported.")
    p.add_argument("--value-loss-weight", type=float, default=1.0, help="Weight for continuous value loss.")
    p.add_argument("--macro-order", action="store_true", help="Prefix Macro-IR with <ORDER_k> tokens.")
    p.add_argument("--macro-value-loss-weight", type=float, default=1.0, help="Weight for Macro-IR slot MSE loss.")
    return p.parse_args()


def build_value_token_info(tokenizer) -> tuple[list[int], dict[int, int]]:
    """
    Returns (value_token_ids, slot_type_token_to_idx).
    slot_type_token_to_idx maps typed value tokens (e.g., <VAL_L>) to a small slot_type id.
    """
    vocab = tokenizer.get_vocab()
    val_ids: list[int] = []
    slot_map: dict[int, int] = {}
    slot_order = {tok: i for i, tok in enumerate(VALUE_SLOTS)}
    for tok, tid in vocab.items():
        if tok.startswith("<VAL_"):
            val_ids.append(int(tid))
            if tok in slot_order:
                slot_map[int(tid)] = int(slot_order[tok])
    return val_ids, slot_map


def build_macro_slot_mask(tokenizer) -> tuple[int, torch.Tensor | None]:
    vocab = tokenizer.get_vocab()
    if not MACRO_LIBRARY:
        return 0, None
    slot_count = max(len(m.slot_types) for m in MACRO_LIBRARY.values())
    if slot_count <= 0:
        return 0, None
    mask = torch.zeros((len(vocab), slot_count), dtype=torch.float32)
    for macro_id, macro in MACRO_LIBRARY.items():
        tid = vocab.get(macro_id)
        if tid is None:
            continue
        slen = len(macro.slot_types)
        if slen > 0:
            mask[int(tid), :slen] = 1.0
    return slot_count, mask


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    value_token_ids, slot_type_map = build_value_token_info(tokenizer)

    train_ds = FilterDesignDataset(
        str(args.data),
        tokenizer,
        use_wave=args.use_wave,
        mix_real_prob=args.mix_real_prob,
        use_repr=args.repr,
        normalize_wave=args.wave_norm,
        freq_mode=args.freq_mode,
        freq_scale=args.freq_scale,
        include_s11=args.include_s11,
        macro_order=bool(args.macro_order),
    )
    eval_ds = None
    if args.eval_data:
        eval_ds = FilterDesignDataset(
            str(args.eval_data),
            tokenizer,
            use_wave=args.use_wave,
            mix_real_prob=args.mix_real_prob,
            use_repr=args.repr,
            normalize_wave=args.wave_norm,
            freq_mode=args.freq_mode,
            freq_scale=args.freq_scale,
            include_s11=args.include_s11,
            macro_order=bool(args.macro_order),
        )
    collate_fn = make_collate_fn(tokenizer, use_repr=args.repr)

    sample_wave = train_ds[0]["wave"]
    in_channels = sample_wave.shape[0]
    macro_slot_count = None
    macro_slot_mask = None
    if args.repr == "macro_ir":
        macro_slot_count, macro_slot_mask = build_macro_slot_mask(tokenizer)

    model = VACTT5(
        t5_name=args.t5_name,
        waveform_in_channels=in_channels,
        vocab_size=len(tokenizer),
        spec_mode=args.spec_mode,
        value_loss_weight=args.value_loss_weight,
        value_token_ids=value_token_ids,
        slot_type_token_to_idx=slot_type_map,
        macro_slot_count=macro_slot_count,
        macro_value_loss_weight=args.macro_value_loss_weight,
        macro_slot_mask=macro_slot_mask,
    )
    model.t5.config.eos_token_id = tokenizer.eos_token_id
    model.t5.config.pad_token_id = tokenizer.pad_token_id
    model.t5.config.decoder_start_token_id = tokenizer.pad_token_id

    eval_strategy = "steps" if eval_ds is not None else "no"

    save_steps = args.save_steps
    if eval_ds is not None and save_steps % args.eval_steps != 0:
        # align save/eval cadence to satisfy load_best_model_at_end requirement
        save_steps = args.eval_steps

    training_kwargs = dict(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.log_steps,
        save_steps=save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=eval_strategy,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False,  # tied embeddings -> avoid shared-tensor safetensors error
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_first_step=True,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters)
    eval_key = None
    if "evaluation_strategy" in valid_params:
        eval_key = "evaluation_strategy"
    elif "eval_strategy" in valid_params:
        # Newer transformers renamed evaluation_strategy -> eval_strategy.
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
        eval_key = "eval_strategy"
    if eval_key is None:
        if "evaluate_during_training" in valid_params:
            training_kwargs["evaluate_during_training"] = eval_strategy != "no"
        if training_kwargs.get("load_best_model_at_end"):
            print("[warn] TrainingArguments lacks evaluation strategy; disabling load_best_model_at_end.")
            training_kwargs["load_best_model_at_end"] = False
            training_kwargs.pop("metric_for_best_model", None)
            training_kwargs.pop("greater_is_better", None)
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_params}
    dropped = sorted(set(training_kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"[warn] TrainingArguments does not accept {dropped}; dropping for compatibility.")
    training_args = TrainingArguments(**filtered_kwargs)

    trainer = Wave2CircuitTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        grammar_masker=build_train_time_grammar_masker(tokenizer, repr_kind=args.repr) if args.grammar_mask else None,
    )

    trainer.train()
    trainer.save_model(str(args.output))
    input_cfg = {
        "freq_mode": args.freq_mode,
        "freq_scale": args.freq_scale,
        "include_s11": bool(args.include_s11),
        "spec_mode": args.spec_mode,
        "in_channels": int(in_channels),
        "macro_order": bool(args.macro_order),
    }
    cfg_path = args.output / "input_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w") as f:
        json.dump(input_cfg, f, indent=2, sort_keys=True)
    # Save minimal HF-style artifacts for eval/debug.
    model.t5.config.save_pretrained(str(args.output))
    model.t5.generation_config.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
