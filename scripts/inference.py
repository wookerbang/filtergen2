from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.torch_dataset import FilterDesignDataset
from src.data.vact_codec import make_vact_syntax_prefix_allowed_tokens_fn
from src.data.vact_struct import make_vact_struct_prefix_allowed_tokens_fn
from src.data.dsl import VALUE_SLOTS, make_dsl_prefix_allowed_tokens_fn
from src.data.token_decode import build_label_value_map, decode_components_from_token_ids
from src.models import VACTT5
from src.physics import FastTrackEngine
from src.eval.yield_analysis import build_spec_masks, estimate_yield_mc, estimate_yield_sequential, prepare_yield_spec


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verifier-guided best-of-K inference with refinement.")
    p.add_argument("--data", required=True, type=Path, help="Path to dataset jsonl (val/test).")
    p.add_argument("--ckpt", required=True, type=Path, help="Checkpoint dir (trainer save).")
    p.add_argument("--tokenizer", type=Path, help="Tokenizer path (defaults to --ckpt).")
    p.add_argument("--t5-name", type=str, default="t5-small", help="Base T5 model name (for raw state_dict load).")
    p.add_argument(
        "--repr",
        choices=["vact", "vact_struct", "dsl", "action"],
        default="vact_struct",
    )
    p.add_argument("--num", type=int, default=50, help="Number of samples to run.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample selection.")
    p.add_argument("--use-wave", default="ideal", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"])
    p.add_argument("--target", default="ideal", choices=["ideal", "real"], help="Target curve for verifier loss.")
    p.add_argument("--wave-norm", action="store_true", help="Normalize waveforms (must match training if enabled).")
    p.add_argument(
        "--freq-mode",
        choices=["none", "log_fc", "linear_fc", "log_f", "log_f_centered"],
        default="log_fc",
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
        default="none",
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
        default="none",
        help="Spec token usage: none (wave-only) or type_fc (prepend filter type + fc token).",
    )
    p.add_argument(
        "--no-s11",
        dest="include_s11",
        action="store_false",
        help="Drop S11 channels from waveform input.",
    )
    p.set_defaults(include_s11=True)
    p.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help="Allow input config mismatch with checkpoint (may skip weights).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # generation
    p.add_argument("--kmax", type=int, default=16, help="Generate this many candidates per sample.")
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of beam search.")
    p.add_argument("--num-beams", type=int, default=4, help="Beam size (used when --do-sample is off).")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p (used when --do-sample).")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (used when --do-sample).")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for generation (>1 discourages repeats).")
    p.add_argument("--max-new", type=int, default=256, help="Max new tokens.")
    p.add_argument("--syntax-mask", action="store_true", help="Apply representation grammar mask during decoding.")
    p.add_argument("--value-mode", choices=["standard", "precision"], default="precision", help="Numeric inference mode for DSL slots.")

    # verifier / refinement
    p.add_argument("--verify-top", type=int, default=4, help="Refine top-M candidates after verifier scoring.")
    p.add_argument("--refine-steps", type=int, default=50, help="Refinement steps.")
    p.add_argument("--refine-lr", type=float, default=5e-2, help="Refinement learning rate.")
    p.add_argument("--refine-max-ratio", type=float, default=2.0, help="Clamp value range: v in [v0/r, v0*r].")
    p.add_argument("--q", type=float, default=50.0, help="Finite-Q loss model (applied to both L and C unless overridden).")
    p.add_argument("--q-l", type=float, default=None, help="Override Q for inductors (None -> use --q).")
    p.add_argument("--q-c", type=float, default=None, help="Override Q for capacitors (None -> use --q).")
    p.add_argument(
        "--q-model",
        type=str,
        default="freq_dependent",
        choices=["freq_dependent", "fixed_ref"],
        help="Fast Track Q model: freq_dependent (high fidelity) or fixed_ref (SPICE-like).",
    )
    p.add_argument("--passband-min-db", type=float, default=-3.0, help="Passband lower bound for hinge loss.")
    p.add_argument("--stopband-max-db", type=float, default=-40.0, help="Stopband upper bound for hinge loss.")
    p.add_argument("--guide-weight", type=float, default=1e-2, help="MSE guidance weight for hinge loss.")
    p.add_argument("--hinge-power", type=float, default=2.0, help="Hinge power (L2 when 2).")
    p.add_argument("--snap-series", type=str, default="E24", help="Snap-to standard series (E24/E12/none).")

    # yield evaluation (Monte Carlo)
    p.add_argument("--yield-n", type=int, default=0, help="MC sample count for yield (0 disables).")
    p.add_argument("--yield-seq", action="store_true", help="Use sequential sampling for yield.")
    p.add_argument("--yield-n-min", type=int, default=200, help="Min samples for sequential yield.")
    p.add_argument("--yield-n-max", type=int, default=2000, help="Max samples for sequential yield.")
    p.add_argument("--yield-batch", type=int, default=200, help="Batch size for sequential yield.")
    p.add_argument("--yield-ci", type=float, default=0.02, help="Target CI half-width (sequential).")
    p.add_argument("--yield-ci-method", choices=["wilson", "agresti_coull"], default="wilson")
    p.add_argument("--yield-score", choices=["mean", "ci_lower"], default="mean", help="Selection score for yield rerank.")
    p.add_argument("--yield-tol", type=float, default=0.05, help="Tolerance fraction for yield Monte Carlo.")
    p.add_argument("--yield-tol-l", type=float, default=None, help="Override L tolerance (None -> --yield-tol).")
    p.add_argument("--yield-tol-c", type=float, default=None, help="Override C tolerance (None -> --yield-tol).")
    p.add_argument("--yield-dist", choices=["uniform", "normal"], default="uniform", help="Tolerance distribution.")
    p.add_argument("--yield-sigma-frac", type=float, default=None, help="Sigma fraction for normal tolerance.")
    p.add_argument("--yield-trunc-sigma", type=float, default=3.0, help="Truncation sigma for normal tolerance.")
    p.add_argument("--yield-global-sigma", type=float, default=0.0, help="Global (lot) sigma fraction for L/C.")
    p.add_argument("--yield-global-sigma-l", type=float, default=None, help="Override global sigma for inductors.")
    p.add_argument("--yield-global-sigma-c", type=float, default=None, help="Override global sigma for capacitors.")
    p.add_argument("--yield-global-trunc-sigma", type=float, default=3.0, help="Truncation sigma for global shift.")
    p.add_argument("--yield-rl-min-db", type=float, default=None, help="Return loss minimum (positive, dB).")
    p.add_argument("--yield-ignore-ripple", action="store_true", help="Ignore ripple spec in yield checks.")
    p.add_argument("--yield-ripple-db", type=float, default=None, help="Override ripple spec for yield (dB).")
    p.add_argument("--yield-stage2-top", type=int, default=0, help="Rerun yield on top-N candidates with larger budget.")
    p.add_argument("--yield-stage2-n", type=int, default=0, help="Stage-2 MC samples (or seq n_max).")
    p.add_argument(
        "--allow-mask-fallback",
        action="store_true",
        help="Allow spec-based mask fallback when dataset masks are missing (demo only).",
    )

    p.add_argument("--dump", type=Path, help="Optional JSONL dump of per-sample best candidate + score.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok_path = args.tokenizer or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = FilterDesignDataset(
        str(args.data),
        tokenizer,
        use_wave=args.use_wave,
        normalize_wave=args.wave_norm,
        use_repr="vact",
        freq_mode=args.freq_mode,
        freq_scale=args.freq_scale,
        include_s11=args.include_s11,
    )
    idxs = random.sample(range(len(ds)), min(int(args.num), len(ds)))

    def _build_value_token_info(tok):
        vocab = tok.get_vocab()
        val_ids = []
        slot_map = {}
        slot_order = {t: i for i, t in enumerate(VALUE_SLOTS)}
        for t, tid in vocab.items():
            if t.startswith("<VAL_"):
                val_ids.append(int(tid))
                if t in slot_order:
                    slot_map[int(tid)] = int(slot_order[t])
        return val_ids, slot_map

    value_token_ids, slot_type_map = _build_value_token_info(tokenizer)

    in_channels = ds[0]["wave"].shape[0]
    cfg_path = args.ckpt / "input_config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = json.load(f)
        mismatches = []
        for key, expected in (
            ("freq_mode", args.freq_mode),
            ("freq_scale", args.freq_scale),
            ("include_s11", bool(args.include_s11)),
            ("spec_mode", args.spec_mode),
        ):
            if key in cfg and cfg[key] != expected:
                mismatches.append((key, cfg[key], expected))
        if "in_channels" in cfg and int(cfg["in_channels"]) != int(in_channels):
            mismatches.append(("in_channels", cfg["in_channels"], int(in_channels)))
        if mismatches:
            lines = ["Input config mismatch with checkpoint:"]
            lines.extend([f"- {k}: ckpt={v_ckpt} current={v_cur}" for k, v_ckpt, v_cur in mismatches])
            lines.append("Align flags or pass --allow-input-mismatch.")
            msg = "\n".join(lines)
            if args.allow_input_mismatch:
                print(f"[warn] {msg}")
            else:
                raise ValueError(msg)
    model = VACTT5(
        t5_name=args.t5_name,
        waveform_in_channels=in_channels,
        vocab_size=len(tokenizer),
        spec_mode=args.spec_mode,
        value_token_ids=value_token_ids,
        slot_type_token_to_idx=slot_type_map,
    )
    state_path = args.ckpt / "pytorch_model.bin"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.t5.config.eos_token_id = tokenizer.eos_token_id
    model.t5.config.pad_token_id = tokenizer.pad_token_id
    model.t5.config.decoder_start_token_id = tokenizer.pad_token_id
    model.to(args.device).eval()

    label_map = build_label_value_map(tokenizer)
    prefix_allowed = None
    if args.syntax_mask:
        if args.repr == "vact":
            prefix_allowed = make_vact_syntax_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr == "vact_struct":
            prefix_allowed = make_vact_struct_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr == "dsl":
            prefix_allowed = make_dsl_prefix_allowed_tokens_fn(tokenizer)

    q_l = args.q if args.q_l is None else args.q_l
    q_c = args.q if args.q_c is None else args.q_c
    q_model = str(args.q_model)
    engine: FastTrackEngine | None = None

    dump_rows: list[Dict[str, object]] = []

    for idx in idxs:
        sample = ds[idx]
        raw = ds.samples[idx]
        freq = np.asarray(raw.get("freq_hz", []), dtype=float)
        z0 = float(raw.get("z0", 50.0))
        target_s21 = np.asarray(
            raw.get("ideal_s21_db") if args.target == "ideal" else raw.get("real_s21_db"),
            dtype=float,
        )
        mask_min = raw.get("mask_min_db")
        mask_max = raw.get("mask_max_db")
        mask_min_db = None
        mask_max_db = None
        if mask_min is not None and mask_max is not None:
            mask_min_db = np.asarray(mask_min, dtype=float)
            mask_max_db = np.asarray(mask_max, dtype=float)
            if mask_min_db.shape != freq.shape or mask_max_db.shape != freq.shape:
                mask_min_db = None
                mask_max_db = None
        if (mask_min_db is None or mask_max_db is None) and freq.size:
            if args.allow_mask_fallback:
                try:
                    mask_min_db, mask_max_db, _, _ = build_spec_masks(raw, freq)
                except Exception:
                    mask_min_db = None
                    mask_max_db = None
            else:
                print(f"[warn] mask missing for idx={idx}; enable --allow-mask-fallback for demo runs.")
                continue
        if freq.size == 0 or target_s21.size == 0:
            continue
        if engine is None or float(engine.z0) != z0:
            engine = FastTrackEngine(z0=z0, device=args.device, dtype=torch.float64)

        wave = sample["wave"].unsqueeze(0).to(args.device)
        scalars = sample["scalar"]
        filter_type = scalars[0:1].long().to(args.device)
        fc_hz = scalars[1:2].to(args.device)

        gen_kwargs = dict(
            wave=wave,
            filter_type=filter_type,
            fc_hz=fc_hz,
            max_new_tokens=int(args.max_new),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed,
            num_return_sequences=int(args.kmax),
            repetition_penalty=float(args.repetition_penalty),
        )
        if args.do_sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    num_beams=1,
                    top_p=float(args.top_p),
                    temperature=float(args.temperature),
                )
            )
        else:
            gen_kwargs.update(dict(do_sample=False, num_beams=max(int(args.num_beams), int(args.kmax))))

        with torch.no_grad():
            outs = model.generate(**gen_kwargs)
        seqs = outs.cpu().tolist()

        slot_values_seqs = None
        if args.repr == "dsl":
            seq_lens = [len(s) for s in seqs]
            max_len = max(seq_lens) if seq_lens else 0
            if max_len > 0:
                pad_id = int(tokenizer.pad_token_id)
                seq_tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=args.device)
                for i, s in enumerate(seqs):
                    seq_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=args.device)
                wave_rep = wave.repeat(len(seqs), 1, 1)
                filter_rep = filter_type.repeat(len(seqs))
                fc_rep = fc_hz.repeat(len(seqs))
                with torch.no_grad():
                    pred_vals = model.predict_values(
                        wave_rep,
                        filter_rep,
                        fc_rep,
                        token_ids=seq_tensor,
                        mode=args.value_mode,
                    )
                pred_vals = pred_vals.detach().cpu().tolist()
                slot_values_seqs = [pred_vals[i][: seq_lens[i]] for i in range(len(seqs))]

        cand_records: list[tuple[float, list, list[str] | None]] = []
        for i_seq, seq in enumerate(seqs):
            try:
                slot_values = slot_values_seqs[i_seq] if slot_values_seqs is not None else None
                comps, toks = decode_components_from_token_ids(
                    seq,
                    tokenizer,
                    repr_kind=args.repr,
                    label_to_value=label_map,
                    slot_values=slot_values,
                )
                if not comps:
                    cand_records.append((float("inf"), None, None))
                    continue
                # Verifier loss (mask-driven) without refinement.
                res = engine.refine(
                    comps,
                    freq_hz=freq,
                    target_s21_db=target_s21,
                    mask_min_db=mask_min_db,
                    mask_max_db=mask_max_db,
                    steps=0,
                    lr=float(args.refine_lr),
                    optimizer="adam",
                    q_L=q_l,
                    q_C=q_c,
                    q_model=q_model,
                    max_ratio=float(args.refine_max_ratio),
                    loss_kind="spec_hinge",
                    passband_min_db=float(args.passband_min_db),
                    stopband_max_db=float(args.stopband_max_db),
                    guide_weight=float(args.guide_weight),
                    hinge_power=float(args.hinge_power),
                    snap_series=None,
                )
                cand_records.append((float(res.initial_loss), comps, toks))
            except Exception:
                cand_records.append((float("inf"), None, None))

        # pick top-M by verifier loss
        scored = [(score, j) for j, (score, comps, toks) in enumerate(cand_records) if np.isfinite(score) and comps]
        scored.sort(key=lambda x: float(x[0]))
        refine_idxs = [j for _, j in scored[: max(1, int(args.verify_top))]]

        refined_records: list[dict[str, object]] = []
        for j in refine_idxs:
            score0, comps0, toks0 = cand_records[j]
            if comps0 is None:
                continue
            res = engine.refine(
                comps0,
                freq_hz=freq,
                target_s21_db=target_s21,
                mask_min_db=mask_min_db,
                mask_max_db=mask_max_db,
                steps=int(args.refine_steps),
                lr=float(args.refine_lr),
                optimizer="adam",
                q_L=q_l,
                q_C=q_c,
                q_model=q_model,
                max_ratio=float(args.refine_max_ratio),
                loss_kind="spec_hinge",
                passband_min_db=float(args.passband_min_db),
                stopband_max_db=float(args.stopband_max_db),
                guide_weight=float(args.guide_weight),
                hinge_power=float(args.hinge_power),
                snap_series=None if str(args.snap_series).lower() == "none" else str(args.snap_series),
            )
            final_score = res.snapped_loss if res.snapped_loss is not None else res.final_loss
            refined_records.append(
                {
                    "score": float(final_score),
                    "components": res.refined_components,
                    "tokens": toks0,
                }
            )

        if not refined_records:
            continue

        best_score = float("inf")
        best_comp = None
        best_tokens = None
        best_yield = None

        yield_enabled = bool(args.yield_seq) or int(args.yield_n) > 0
        yield_spec = None
        if yield_enabled:
            if args.yield_ignore_ripple:
                ripple_db = None
            elif args.yield_ripple_db is not None:
                ripple_db = float(args.yield_ripple_db)
            else:
                ripple_db = float(raw.get("ripple_db")) if raw.get("ripple_db") is not None else None
            try:
                yield_spec = prepare_yield_spec(
                    freq_hz=freq,
                    mask_min_db=mask_min_db,
                    mask_max_db=mask_max_db,
                    spec=raw,
                    passband_ripple_max_db=ripple_db,
                    return_loss_min_db=args.yield_rl_min_db,
                    require_masks=True,
                )
            except Exception as exc:
                print(f"[warn] yield spec build failed for idx={idx}: {exc}")
                yield_spec = None
                yield_enabled = False

        if yield_enabled and yield_spec is not None:
            tol_L = args.yield_tol if args.yield_tol_l is None else args.yield_tol_l
            tol_C = args.yield_tol if args.yield_tol_c is None else args.yield_tol_c
            tol_map = {"L": float(tol_L), "C": float(tol_C)}
            global_sigma_L = args.yield_global_sigma if args.yield_global_sigma_l is None else args.yield_global_sigma_l
            global_sigma_C = args.yield_global_sigma if args.yield_global_sigma_c is None else args.yield_global_sigma_c
            global_trunc_sigma = float(args.yield_global_trunc_sigma)
            base_seed = int(args.seed) + int(idx) * 1009
            for rec in refined_records:
                comps = rec["components"]
                rng = np.random.default_rng(base_seed)
                if args.yield_seq:
                    y_res = estimate_yield_sequential(
                        components=comps,
                        freq_hz=freq,
                        spec=yield_spec,
                        n_min=int(args.yield_n_min),
                        n_max=int(args.yield_n_max),
                        batch=int(args.yield_batch),
                        target_half_width=float(args.yield_ci),
                        rng=rng,
                        engine=engine,
                        z0=z0,
                        q_L=q_l,
                        q_C=q_c,
                        q_model=q_model,
                        tol_L=float(tol_L),
                        tol_C=float(tol_C),
                        tol_map=tol_map,
                        global_sigma_L=global_sigma_L,
                        global_sigma_C=global_sigma_C,
                        global_trunc_sigma=global_trunc_sigma,
                        dist=str(args.yield_dist),
                        sigma_frac=args.yield_sigma_frac,
                        trunc_sigma=float(args.yield_trunc_sigma),
                        ci_method=str(args.yield_ci_method),
                    )
                else:
                    y_res = estimate_yield_mc(
                        components=comps,
                        freq_hz=freq,
                        spec=yield_spec,
                        n=int(args.yield_n),
                        rng=rng,
                        engine=engine,
                        z0=z0,
                        q_L=q_l,
                        q_C=q_c,
                        q_model=q_model,
                        tol_L=float(tol_L),
                        tol_C=float(tol_C),
                        tol_map=tol_map,
                        global_sigma_L=global_sigma_L,
                        global_sigma_C=global_sigma_C,
                        global_trunc_sigma=global_trunc_sigma,
                        dist=str(args.yield_dist),
                        sigma_frac=args.yield_sigma_frac,
                        trunc_sigma=float(args.yield_trunc_sigma),
                        ci_method=str(args.yield_ci_method),
                    )
                rec["yield"] = y_res
                if str(args.yield_score) == "ci_lower":
                    rec["yield_score"] = float(y_res.ci_low)
                else:
                    rec["yield_score"] = float(y_res.yield_hat)

            refined_records.sort(key=lambda r: (float(r.get("yield_score", 0.0)), -float(r.get("score", 0.0))), reverse=True)
            stage2_top = int(args.yield_stage2_top)
            stage2_n = int(args.yield_stage2_n)
            if stage2_top > 0 and stage2_n > 0:
                for rec in refined_records[: min(stage2_top, len(refined_records))]:
                    comps = rec["components"]
                    rng = np.random.default_rng(base_seed)
                    if args.yield_seq:
                        n_max2 = max(int(stage2_n), 1)
                        n_min2 = min(int(args.yield_n_min), n_max2)
                        y_res = estimate_yield_sequential(
                            components=comps,
                            freq_hz=freq,
                            spec=yield_spec,
                            n_min=int(n_min2),
                            n_max=int(n_max2),
                            batch=int(args.yield_batch),
                            target_half_width=float(args.yield_ci),
                            rng=rng,
                            engine=engine,
                            z0=z0,
                            q_L=q_l,
                            q_C=q_c,
                            q_model=q_model,
                            tol_L=float(tol_L),
                            tol_C=float(tol_C),
                            tol_map=tol_map,
                            global_sigma_L=global_sigma_L,
                            global_sigma_C=global_sigma_C,
                            global_trunc_sigma=global_trunc_sigma,
                            dist=str(args.yield_dist),
                            sigma_frac=args.yield_sigma_frac,
                            trunc_sigma=float(args.yield_trunc_sigma),
                            ci_method=str(args.yield_ci_method),
                        )
                    else:
                        y_res = estimate_yield_mc(
                            components=comps,
                            freq_hz=freq,
                            spec=yield_spec,
                            n=int(stage2_n),
                            rng=rng,
                            engine=engine,
                            z0=z0,
                            q_L=q_l,
                            q_C=q_c,
                            q_model=q_model,
                            tol_L=float(tol_L),
                            tol_C=float(tol_C),
                            tol_map=tol_map,
                            global_sigma_L=global_sigma_L,
                            global_sigma_C=global_sigma_C,
                            global_trunc_sigma=global_trunc_sigma,
                            dist=str(args.yield_dist),
                            sigma_frac=args.yield_sigma_frac,
                            trunc_sigma=float(args.yield_trunc_sigma),
                            ci_method=str(args.yield_ci_method),
                        )
                    rec["yield"] = y_res
                    if str(args.yield_score) == "ci_lower":
                        rec["yield_score"] = float(y_res.ci_low)
                    else:
                        rec["yield_score"] = float(y_res.yield_hat)
                refined_records.sort(key=lambda r: (float(r.get("yield_score", 0.0)), -float(r.get("score", 0.0))), reverse=True)
            best = refined_records[0]
            best_comp = best["components"]
            best_tokens = best["tokens"]
            best_yield = best.get("yield")
            best_score = float(best.get("score", float("inf")))
        else:
            for rec in refined_records:
                score = float(rec.get("score", float("inf")))
                if score < best_score:
                    best_score = score
                    best_comp = rec["components"]
                    best_tokens = rec["tokens"]

        if best_comp is None:
            continue

        dump_rows.append(
            {
                "idx": idx,
                "sample_id": raw.get("sample_id"),
                "filter_type": raw.get("filter_type"),
                "fc_hz": float(raw.get("fc_hz", 0.0)),
                "best_score": float(best_score),
                "num_components": len(best_comp),
                "gen_tokens": (best_tokens or [])[:200],
                "yield_hat": float(best_yield.yield_hat) if best_yield is not None else None,
                "yield_ci_low": float(best_yield.ci_low) if best_yield is not None else None,
                "yield_ci_high": float(best_yield.ci_high) if best_yield is not None else None,
                "yield_n": int(best_yield.num_total) if best_yield is not None else None,
                "yield_fail_counts": best_yield.fail_counts if best_yield is not None else None,
            }
        )

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dumped to {args.dump}")
    else:
        print(f"Samples processed: {len(dump_rows)}")


if __name__ == "__main__":
    main()
