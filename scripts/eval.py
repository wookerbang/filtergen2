from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.circuits import Circuit
from src.data.torch_dataset import FilterDesignDataset
from src.data.token_decode import build_label_value_map, decode_components_from_token_ids
from src.data.vact_codec import make_vact_syntax_prefix_allowed_tokens_fn
from src.data.vact_struct import make_vact_struct_prefix_allowed_tokens_fn
from src.data.dsl import VALUE_SLOTS, make_dsl_prefix_allowed_tokens_fn
from src.data.spice_runner import run_ac_analysis_with_ngspice
from src.eval.metrics import waveform_error
from src.models import VACTT5
from src.physics import FastTrackEngine


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _ref_freq_hz(freq_hz: np.ndarray, fc_hz: float | None) -> float:
    if fc_hz is not None and np.isfinite(fc_hz):
        return float(fc_hz)
    f_min = float(np.min(freq_hz))
    f_max = float(np.max(freq_hz))
    return float(np.sqrt(f_min * f_max))


def _simulate_s21_db(
    *,
    components,
    freq_hz: np.ndarray,
    z0: float,
    fc_hz: float | None,
    use_ngspice: bool,
    q_L: float | None,
    q_C: float | None,
    q_model: str,
    fast_engine: FastTrackEngine,
) -> np.ndarray:
    use_spice = bool(use_ngspice) and ((q_L is None and q_C is None) or str(q_model) == "fixed_ref")
    if use_spice:
        ref_freq_hz = _ref_freq_hz(freq_hz, fc_hz)
        circuit = Circuit(components, z0=z0, in_port=("in", "gnd"), out_port=("out", "gnd"))
        try:
            s21_db, _ = run_ac_analysis_with_ngspice(
                circuit,
                freq_hz,
                z0,
                q_L=q_L,
                q_C=q_C,
                ref_freq_hz=ref_freq_hz,
            )
            return s21_db
        except RuntimeError:
            pass
    ref_freq_hz = _ref_freq_hz(freq_hz, fc_hz) if str(q_model) == "fixed_ref" and (q_L is not None or q_C is not None) else None
    return fast_engine.simulate_s21_db(
        components,
        freq_hz,
        q_L=q_L,
        q_C=q_C,
        q_model=str(q_model),
        ref_freq_hz=ref_freq_hz,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best-of-K evaluation with simulation verifier (VACT/VACT-Struct/DSL).")
    p.add_argument("--data", required=True, type=Path, help="Path to dataset jsonl (val/test).")
    p.add_argument("--ckpt", required=True, type=Path, help="Checkpoint dir (trainer save).")
    p.add_argument("--tokenizer", type=Path, help="Tokenizer path (defaults to --ckpt).")
    p.add_argument("--t5-name", type=str, default="t5-small", help="Base T5 model name (for raw state_dict load).")
    p.add_argument(
        "--repr",
        choices=["vact", "vact_struct", "dsl", "action"],
        default="vact_struct",
        help="Target representation to decode.",
    )
    p.add_argument("--num", type=int, default=200, help="Number of samples to eval.")
    p.add_argument("--sample-id", type=str, help="Evaluate a single sample_id (overrides --num).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample selection.")
    p.add_argument("--use-wave", default="real", choices=["ideal", "real", "both", "ideal_s21", "real_s21", "mix"])
    p.add_argument(
        "--target-wave",
        default="auto",
        choices=["auto", "ideal", "real"],
        help="Target S21 selection: auto (prefer real if present), ideal, or real.",
    )
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
    p.add_argument("--k-eval", type=str, default="1,4,16", help="Comma-separated K values for best-of-K curves.")
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of beam search.")
    p.add_argument("--num-beams", type=int, default=4, help="Beam size (used when --do-sample is off).")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling p (used when --do-sample).")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (used when --do-sample).")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for generation (>1 discourages repeats).")
    p.add_argument("--max-new", type=int, default=256, help="Max new tokens.")
    p.add_argument("--syntax-mask", action="store_true", help="Apply representation grammar mask during decoding.")
    p.add_argument("--value-mode", choices=["standard", "precision"], default="precision", help="Numeric inference mode for DSL slots.")

    # simulation + metric
    p.add_argument("--use-ngspice", dest="use_ngspice", action="store_true", help="Use ngspice for evaluation (fallback to Fast Track).")
    p.add_argument("--no-ngspice", dest="use_ngspice", action="store_false", help="Disable ngspice; use Fast Track only.")
    p.set_defaults(use_ngspice=True)
    p.add_argument("--q", type=float, default=50.0, help="Finite-Q loss model (applied to both L and C unless overridden).")
    p.add_argument("--q-l", type=float, default=None, help="Override Q for inductors (None -> use --q).")
    p.add_argument("--q-c", type=float, default=None, help="Override Q for capacitors (None -> use --q).")
    p.add_argument(
        "--q-model",
        type=str,
        default="freq_dependent",
        choices=["freq_dependent", "fixed_ref"],
        help="Q modeling for eval: freq_dependent (high fidelity) or fixed_ref (SPICE-style).",
    )
    p.add_argument("--error", choices=["mae_lin", "rmse_lin", "mae_db", "rmse_db", "maxe_lin", "maxe_db"], default="mae_lin")
    p.add_argument("--taus", type=str, default="0.01,0.02,0.05", help="Comma-separated τ thresholds for success@τ.")
    p.add_argument("--debug-nan", action="store_true", help="Print debug info when simulated S21 contains NaN/Inf.")

    # refinement (topology fixed)
    p.add_argument("--refine-steps", type=int, default=0, help="Few-step differentiable refinement steps (0 disables).")
    p.add_argument("--refine-top", type=int, default=1, help="Refine top-N candidates per sample.")
    p.add_argument("--refine-lr", type=float, default=5e-2, help="Refinement learning rate.")
    p.add_argument("--refine-max-ratio", type=float, default=2.0, help="Clamp value range: v in [v0/r, v0*r].")

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

    # dataset for waveform/spec conditioning + raw targets
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
    if args.sample_id:
        idxs = [i for i, s in enumerate(ds.samples) if s.get("sample_id") == args.sample_id]
        if not idxs:
            raise ValueError(f"sample_id not found in dataset: {args.sample_id}")
        idxs = idxs[:1]
    else:
        idxs = random.sample(range(len(ds)), min(int(args.num), len(ds)))

    # model
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
        # Be forgiving to config drift (e.g., type_vocab_size changes).
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped.append(k)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[warn] skipped {len(skipped)} mismatched keys (e.g., config drift): {skipped[:6]}")
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.t5.config.eos_token_id = tokenizer.eos_token_id
    model.t5.config.pad_token_id = tokenizer.pad_token_id
    model.t5.config.decoder_start_token_id = tokenizer.pad_token_id
    model.to(args.device).eval()

    label_map = build_label_value_map(tokenizer)

    taus = _parse_csv_floats(args.taus)
    k_eval = sorted(set(_parse_csv_ints(args.k_eval)))
    kmax = int(args.kmax)
    if kmax < max(k_eval):
        raise ValueError(f"--kmax ({kmax}) must be >= max(--k-eval) ({max(k_eval)})")

    q_l = args.q if args.q_l is None else args.q_l
    q_c = args.q if args.q_c is None else args.q_c
    fast_engine: FastTrackEngine | None = None

    prefix_allowed = None
    if args.syntax_mask:
        if args.repr == "vact":
            prefix_allowed = make_vact_syntax_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr == "vact_struct":
            prefix_allowed = make_vact_struct_prefix_allowed_tokens_fn(tokenizer)
        elif args.repr == "dsl":
            prefix_allowed = make_dsl_prefix_allowed_tokens_fn(tokenizer)

    # metrics accumulators
    total = 0
    valid_any = 0  # at least one candidate decoded
    sim_any = 0  # at least one candidate simulated
    success_counts = {(k, tau): 0 for k in k_eval for tau in taus}

    dump_rows = []

    for idx in idxs:
        total += 1
        sample = ds[idx]
        raw = ds.samples[idx]
        freq = np.asarray(raw.get("freq_hz", []), dtype=float)
        if args.target_wave == "ideal":
            target_s21 = np.asarray(raw.get("ideal_s21_db") or raw.get("real_s21_db") or [], dtype=float)
        elif args.target_wave == "real":
            target_s21 = np.asarray(raw.get("real_s21_db") or raw.get("ideal_s21_db") or [], dtype=float)
        else:
            target_s21 = np.asarray(raw.get("real_s21_db") or raw.get("ideal_s21_db") or [], dtype=float)
        if freq.size == 0 or target_s21.size == 0:
            continue
        if args.debug_nan:
            bad_tgt = ~np.isfinite(target_s21)
            if bool(np.any(bad_tgt)):
                bad_idx = np.where(bad_tgt)[0]
                preview = bad_idx[:5]
                preview_vals = [float(target_s21[j]) for j in preview]
                preview_freq = [float(freq[j]) for j in preview]
                print(
                    "[nan_target]"
                    f" sample_id={raw.get('sample_id')}"
                    f" bad={len(bad_idx)}/{len(target_s21)}"
                    f" freq={preview_freq}"
                    f" s21={preview_vals}"
                )
        z0 = float(raw.get("z0", 50.0))
        fc_hz_raw = raw.get("fc_hz")
        if fast_engine is None or float(fast_engine.z0) != z0:
            fast_engine = FastTrackEngine(z0=z0, device=args.device, dtype=torch.float64)

        wave = sample["wave"].unsqueeze(0).to(args.device)
        scalars = sample["scalar"]
        filter_type = scalars[0:1].long().to(args.device)
        fc_hz_tensor = scalars[1:2].to(args.device)

        gen_kwargs = dict(
            wave=wave,
            filter_type=filter_type,
            fc_hz=fc_hz_tensor,
            max_new_tokens=int(args.max_new),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed,
            num_return_sequences=kmax,
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
            gen_kwargs.update(dict(do_sample=False, num_beams=max(int(args.num_beams), kmax)))

        with torch.no_grad():
            outs = model.generate(**gen_kwargs)
        seqs = outs.cpu().tolist()

        slot_values_seqs = None
        if args.repr == "dsl":
            # Second pass: predict numeric values from decoder hidden states.
            seq_lens = [len(s) for s in seqs]
            max_len = max(seq_lens) if seq_lens else 0
            if max_len > 0:
                pad_id = int(tokenizer.pad_token_id)
                seq_tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=args.device)
                for i, s in enumerate(seqs):
                    seq_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=args.device)
                wave_rep = wave.repeat(len(seqs), 1, 1)
                filter_rep = filter_type.repeat(len(seqs))
                fc_rep = fc_hz_tensor.repeat(len(seqs))
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

        # Candidate list aligned with generation order (length == kmax).
        # Each entry: (err_value, WaveformError|None, comps|None, tokens|None)
        cand_records = []
        any_decoded = False
        any_simulated = False
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
                    cand_records.append((float("inf"), None, None, None))
                    continue
                any_decoded = True
                s21_db = _simulate_s21_db(
                    components=comps,
                    freq_hz=freq,
                    z0=z0,
                    fc_hz=fc_hz_raw,
                    use_ngspice=bool(args.use_ngspice),
                    q_L=q_l,
                    q_C=q_c,
                    q_model=str(args.q_model),
                    fast_engine=fast_engine,
                )
                any_simulated = True
                if args.debug_nan:
                    bad_mask = ~np.isfinite(s21_db)
                    if bool(np.any(bad_mask)):
                        bad_idx = np.where(bad_mask)[0]
                        preview = bad_idx[:5]
                        preview_vals = [float(s21_db[j]) for j in preview]
                        preview_freq = [float(freq[j]) for j in preview]
                        comp_vals = [float(c.value_si) for c in comps]
                        v_min = float(np.min(comp_vals)) if comp_vals else float("nan")
                        v_max = float(np.max(comp_vals)) if comp_vals else float("nan")
                        print(
                            "[nan_s21]"
                            f" sample_id={raw.get('sample_id')}"
                            f" cand={i_seq}"
                            f" bad={len(bad_idx)}/{len(s21_db)}"
                            f" freq={preview_freq}"
                            f" s21={preview_vals}"
                            f" val_range=[{v_min:.3e},{v_max:.3e}]"
                        )
                e0 = waveform_error(s21_db, target_s21, kind=args.error)
                err_val = float(e0.value)
                if not np.isfinite(err_val):
                    err_val = float("inf")
                cand_records.append((err_val, e0, comps, toks))
            except Exception:
                cand_records.append((float("inf"), None, None, None))

        if any_decoded:
            valid_any += 1
        if any_simulated:
            sim_any += 1

        # refine top-N candidates, then re-rank by refined error.
        if args.refine_steps and int(args.refine_steps) > 0:
            refined_records = list(cand_records)
            # pick top-N by current verifier error (excluding inf failures)
            top_n = max(0, int(args.refine_top))
            scored = [(score0, j) for j, (score0, err_obj, comps0, toks0) in enumerate(refined_records) if np.isfinite(score0) and comps0]
            scored.sort(key=lambda x: float(x[0]))
            refine_idxs = [j for _, j in scored[:top_n]]
            loss_kind = "mae_db" if args.error == "mae_db" else "mse_db"
            for j in refine_idxs:
                score0, err_obj, comps0, toks0 = refined_records[j]
                if comps0 is None:
                    continue
                try:
                    res = fast_engine.refine(
                        comps0,
                        freq_hz=freq,
                        target_s21_db=target_s21,
                        steps=int(args.refine_steps),
                        lr=float(args.refine_lr),
                        optimizer="adam",
                        q_L=q_l,
                        q_C=q_c,
                        q_model=str(args.q_model),
                        ref_freq_hz=_ref_freq_hz(freq, fc_hz_raw) if str(args.q_model) == "fixed_ref" and (q_l is not None or q_c is not None) else None,
                        max_ratio=float(args.refine_max_ratio),
                        loss_kind=loss_kind,
                        snap_series=None,
                    )
                    comps_r = res.refined_components
                    s21_db_r = _simulate_s21_db(
                        components=comps_r,
                        freq_hz=freq,
                        z0=z0,
                        fc_hz=fc_hz_raw,
                        use_ngspice=bool(args.use_ngspice),
                        q_L=q_l,
                        q_C=q_c,
                        q_model=str(args.q_model),
                        fast_engine=fast_engine,
                    )
                    err_r = waveform_error(s21_db_r, target_s21, kind=args.error)
                    err_val = float(err_r.value)
                    if not np.isfinite(err_val):
                        err_val = float("inf")
                    refined_records[j] = (err_val, err_r, comps_r, toks0)
                except Exception:
                    pass
            cand_records = refined_records

        # compute best-of-K success in *generation order*
        err_values = [float(r[0]) for r in cand_records]
        best_k_val = {}
        for k in k_eval:
            kk = min(int(k), len(err_values))
            best_k_val[k] = float(np.min(err_values[:kk])) if kk > 0 else float("inf")

        for k in k_eval:
            for tau in taus:
                if np.isfinite(best_k_val[k]) and best_k_val[k] <= float(tau):
                    success_counts[(k, tau)] += 1

        # dump only the best candidate
        best_idx = int(np.argmin(err_values)) if err_values else 0
        best = cand_records[best_idx]
        if args.dump:
            dump_rows.append(
                {
                    "idx": idx,
                    "sample_id": raw.get("sample_id"),
                    "filter_type": raw.get("filter_type"),
                    "fc_hz": float(raw.get("fc_hz", 0.0)),
                    "error_kind": best[1].kind if best[1] is not None else args.error,
                    "best_error": float(best[0]),
                    "num_components": len(best[2] or []),
                    "gen_tokens": (best[3] or [])[:200],
                }
            )

    print(f"Samples evaluated: {total}")
    print(f"Validity@Kmax:    {valid_any}/{total}")
    print(f"Simulated@Kmax:   {sim_any}/{total}")
    for k in k_eval:
        for tau in taus:
            num = success_counts[(k, tau)]
            den = max(total, 1)
            print(f"success@{tau:g} best-of-{k}: {num}/{den} = {num/den:.3f}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as f:
            for row in dump_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dumped to {args.dump}")


if __name__ == "__main__":
    main()
