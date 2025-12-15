from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.torch_dataset import FilterDesignDataset
from src.data.vact_codec import vact_tokens_to_components
from src.data import quantization
from src.models import VACTT5
from src.data.circuits import components_to_abcd, abcd_to_sparams
from src.data.spice_runner import simulate_real_waveform


def build_value_id_map(tokenizer) -> Dict[int, float]:
    vocab = tokenizer.get_vocab()
    mp: Dict[int, float] = {}
    for tok, tid in vocab.items():
        if tok.startswith("<VAL_"):
            label = tok.replace("<VAL_", "").replace(">", "")
            try:
                mp[int(tid)] = float(quantization.label_to_value(label))
            except Exception:
                continue
    return mp


def build_label_value_map(tokenizer) -> Dict[str, float]:
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


def decode_components(token_ids: List[int], tokenizer, label_map: Dict[str, float]):
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    # Drop leading non-component tokens (e.g., <ORDER_k>, <SEP>) before grouping.
    while tokens and not (tokens[0].startswith("<L>") or tokens[0].startswith("<C>")):
        tokens.pop(0)
    comps = vact_tokens_to_components(tokens, label_to_value=label_map)
    tokens_raw = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    return comps, tokens, tokens_raw


def simulate_s21(comps, freq_hz: np.ndarray, z0: float = 50.0, use_ngspice: bool = False) -> np.ndarray:
    if use_ngspice:
        # minimal spec needed by simulate_real_waveform
        spec = {"z0": z0, "filter_type": "lowpass"}
        s21_db, _ = simulate_real_waveform(comps, spec, freq_hz, use_ngspice=True)
        return s21_db
    A, B, C, D = components_to_abcd(comps, freq_hz, z0)
    s21_db, _ = abcd_to_sparams(A, B, C, D, z0)
    return s21_db


def plot_examples(examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]], out_path: Path, max_rows: int = 5) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped (matplotlib not available): {exc}")
        return
    n = min(len(examples), max_rows)
    cols = 1
    plt.figure(figsize=(8, 3 * n))
    for i in range(n):
        freq, target_s21, pred_s21, tag = examples[i]
        ax = plt.subplot(n, cols, i + 1)
        ax.plot(freq, target_s21, label="Target S21", color="C0")
        ax.plot(freq, pred_s21, label=f"Pred S21 ({tag})", color="C1", linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("S21 (dB)")
        ax.grid(True, ls=":")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke eval: generate circuits and compare S21.")
    ap.add_argument("--data", required=True, type=Path, help="Path to val jsonl.")
    ap.add_argument("--ckpt", required=True, type=Path, help="Checkpoint dir (trainer save).")
    ap.add_argument("--tokenizer", type=Path, help="Tokenizer path (defaults to --ckpt if tokenizer files found).")
    ap.add_argument("--t5-name", type=str, default="t5-small", help="Base T5 model name (for raw state_dict load).")
    ap.add_argument("--num", type=int, default=20, help="Number of samples to eval.")
    ap.add_argument("--use-wave", default="real", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"])
    ap.add_argument("--beam", type=int, default=4, help="Beam size for generation.")
    ap.add_argument("--min-new", type=int, default=0, help="Min new tokens to force generation length.")
    ap.add_argument("--use-ngspice", action="store_true", help="Use ngspice for simulation (requires ngspice installed).")
    ap.add_argument("--wave-norm", action="store_true", help="Normalize waveforms (must match training if enabled).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("smoke_plot.png"))
    ap.add_argument("--debug", action="store_true", help="Print generated tokens/components for samples.")
    ap.add_argument(
        "--dump",
        type=Path,
        help="Optional path to dump per-sample JSONL: filter_type, fc, target_tokens, generated tokens/ids, comps.",
    )
    args = ap.parse_args()

    tok_path = args.tokenizer or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[info] pad_id={tokenizer.pad_token_id}, eos_id={tokenizer.eos_token_id}, unk_id={tokenizer.unk_token_id}")

    # build model
    ds_for_shape = FilterDesignDataset(str(args.data), tokenizer, use_wave=args.use_wave, normalize_wave=args.wave_norm)
    in_channels = ds_for_shape[0]["wave"].shape[0]

    state_path = args.ckpt / "pytorch_model.bin"
    value_map = build_value_id_map(tokenizer)
    model = VACTT5(
        t5_name=args.t5_name,
        value_token_to_value=value_map,
        waveform_in_channels=in_channels,
        vocab_size=len(tokenizer),
    )
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Loaded with missing keys ({len(missing)}): {missing}")
            print(f"Loaded with unexpected keys ({len(unexpected)}): {unexpected}")
    else:
        print(f"Warning: {state_path} not found, using base {args.t5_name} weights.")
    model.t5.config.eos_token_id = tokenizer.eos_token_id
    model.t5.config.pad_token_id = tokenizer.pad_token_id
    model.t5.config.decoder_start_token_id = tokenizer.pad_token_id
    model.to(args.device)
    model.eval()

    val_ds = ds_for_shape
    label_map = build_label_value_map(tokenizer)

    idxs = random.sample(range(len(val_ds)), min(args.num, len(val_ds)))
    parse_ok = sim_ok = 0
    plots = []
    dump_rows = []

    for idx in idxs:
        sample = val_ds[idx]
        raw = val_ds.samples[idx]  # original dict with waveforms
        wave = sample["wave"].unsqueeze(0).to(args.device)
        scalars = sample["scalar"]
        filter_type = scalars[0:1].long().to(args.device)
        fc_hz = scalars[1:2].to(args.device)
        freq = np.array(raw.get("freq_hz", []))
        # choose target S21: prefer real, fallback to ideal
        target_s21 = np.array(raw.get("real_s21_db") or raw.get("ideal_s21_db") or [])

        with torch.no_grad():
            gen_ids = model.generate(
                wave=wave,
                filter_type=filter_type,
                fc_hz=fc_hz,
                num_beams=args.beam,
                do_sample=False,
                max_new_tokens=128,
                min_new_tokens=args.min_new,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )[0].cpu().tolist()

        try:
            comps, toks, toks_raw = decode_components(gen_ids, tokenizer, label_map)
            if args.debug:
                print(f"[debug] sample {idx}: len(gen_ids)={len(gen_ids)}, len(tokens)={len(toks)}, len(comps)={len(comps)}")
                print(f"[debug]  gen_ids: {gen_ids[:40]}")
                print(f"[debug]  tokens_raw: {' '.join(toks_raw[:60])}")
                print(f"[debug]  tokens: {' '.join(toks[:60])}")
                for j, c in enumerate(comps[:8]):
                    print(f"[debug]  comp{j}: {c.ctype} {c.role} {c.value_si:.3e} {c.node1}->{c.node2} label={c.std_label}")
            target_tokens = list(raw.get("vact_tokens") or raw.get("sfci_tokens") or [])
            dump_rows.append(
                {
                    "idx": idx,
                    "filter_type": raw.get("filter_type"),
                    "fc_hz": float(raw.get("fc_hz", 0)),
                    "target_tokens": target_tokens,
                    "gen_tokens": toks,
                    "gen_tokens_raw": toks_raw,
                    "gen_token_ids": gen_ids,
                    "unk_in_gen": sum(1 for t in gen_ids if t == tokenizer.unk_token_id),
                    "num_components": len(comps),
                    "target_fc_hz": float(raw.get("fc_hz", 0)),
                    "target_filter_type": raw.get("filter_type"),
                }
            )
            parse_ok += 1
            pred_s21 = simulate_s21(comps, freq, use_ngspice=args.use_ngspice)
            sim_ok += 1
            plots.append((freq, target_s21, pred_s21, f"beam{args.beam}"))
        except Exception as e:
            print(f"Sample {idx} failed: {e}")
            continue

    total = len(idxs)
    print(f"Parse success: {parse_ok}/{total}")
    print(f"Sim success:   {sim_ok}/{total}")

    if plots:
        plot_examples(plots, args.out, max_rows=5)
    if args.dump:
        import json

        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dumped predictions to {args.dump}")


if __name__ == "__main__":
    main()
