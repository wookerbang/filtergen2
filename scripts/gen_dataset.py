"""Simple CLI to run the data generation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import build_dataset


def _load_json_arg(arg: str | None, *, label: str) -> dict | None:
    if arg is None:
        return None
    path = Path(arg)
    try:
        payload = path.read_text() if path.exists() else arg
    except OSError as exc:
        raise ValueError(f"Failed to read --{label}: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --{label} JSON: {exc}") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LC filter dataset jsonl.")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate.")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/demo"), help="Directory to write jsonl.")
    p.add_argument("--split", type=str, default="train", help="Split name, used in <split>.jsonl.")
    p.add_argument(
        "--use-ngspice",
        action="store_true",
        help="Use ngspice for wave simulation when possible (otherwise fall back to Fast Track).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for spec sampling.")
    p.add_argument(
        "--scenario",
        type=str,
        default="random",
        help="Scenario template (general/anti_jamming/coexistence/wideband_rejection/random_basic/lowpass/highpass/bandpass/bandstop) or 'random'.",
    )
    p.add_argument(
        "--scenario-weights",
        type=str,
        help='Optional JSON mapping of scenario -> weight, e.g. \'{"general":0.3,"wideband_rejection":0.3}\'',
    )
    p.add_argument("--q", type=float, default=50.0, help="Finite-Q loss model (applied to both L and C unless overridden).")
    p.add_argument("--q-l", type=float, default=None, help="Override Q for inductors (None -> use --q).")
    p.add_argument("--q-c", type=float, default=None, help="Override Q for capacitors (None -> use --q).")
    p.add_argument(
        "--q-model",
        type=str,
        default="freq_dependent",
        choices=["freq_dependent", "fixed_ref"],
        help="Q modeling for real waveforms: freq_dependent (Fast Track) or fixed_ref (SPICE-style).",
    )
    p.add_argument(
        "--quantize",
        choices=["E24", "E12", "none"],
        default="none",
        help="Quantize component values to E-series (none keeps continuous values).",
    )
    p.add_argument("--vact", dest="vact", action="store_true", help="Emit VACT-Seq tokens.")
    p.add_argument("--no-vact", dest="vact", action="store_false", help="Disable VACT-Seq token emission.")
    p.set_defaults(vact=False)
    p.add_argument("--vact-cell", dest="vact_cell", action="store_true", help="Insert <CELL> markers in VACT.")
    p.add_argument("--no-vact-cell", dest="vact_cell", action="store_false", help="Disable <CELL> markers in VACT.")
    p.set_defaults(vact_cell=False)
    p.add_argument("--vact-struct", dest="vact_struct", action="store_true", help="Emit VACT-Struct tokens.")
    p.add_argument("--no-vact-struct", dest="vact_struct", action="store_false", help="Disable VACT-Struct token emission.")
    p.set_defaults(vact_struct=False)
    p.add_argument("--actions", dest="actions", action="store_true", help="Emit action-construction tokens.")
    p.add_argument("--no-actions", dest="actions", action="store_false", help="Disable action-construction tokens.")
    p.set_defaults(actions=False)
    p.add_argument("--dsl", dest="dsl", action="store_true", help="Emit DSL tokens (macro/repeat).")
    p.add_argument("--no-dsl", dest="dsl", action="store_false", help="Disable DSL token emission.")
    p.set_defaults(dsl=False)
    p.add_argument("--sfci", dest="sfci", action="store_true", help="Emit SFCI tokens.")
    p.add_argument("--no-sfci", dest="sfci", action="store_false", help="Disable SFCI token emission.")
    p.set_defaults(sfci=True)
    p.add_argument(
        "--sfci-value-mode",
        choices=["discrete", "none", "continuous"],
        default="none",
        help="SFCI value tokens: discrete labels, none (<VAL_NONE>), or continuous placeholders.",
    )
    p.add_argument(
        "--sfci-values",
        dest="sfci_value_mode",
        action="store_const",
        const="discrete",
        help="Alias for --sfci-value-mode discrete.",
    )
    p.add_argument(
        "--no-sfci-values",
        dest="sfci_value_mode",
        action="store_const",
        const="none",
        help="Alias for --sfci-value-mode none.",
    )
    p.add_argument("--dsl-order", dest="dsl_order", action="store_true", help="Prepend <ORDER_k> in DSL tokens.")
    p.add_argument("--no-dsl-order", dest="dsl_order", action="store_false", help="Disable <ORDER_k> in DSL tokens.")
    p.set_defaults(dsl_order=True)
    p.add_argument("--dsl-cell-indices", dest="dsl_cell_indices", action="store_true", help="Emit <CELL_IDX_i> in DSL.")
    p.add_argument("--no-dsl-cell-indices", dest="dsl_cell_indices", action="store_false", help="Disable <CELL_IDX_i> in DSL.")
    p.set_defaults(dsl_cell_indices=False)
    p.add_argument("--dsl-strict", dest="dsl_strict", action="store_true", help="Drop samples with <VAL_NONE> or DSL parse failures.")
    p.add_argument("--no-dsl-strict", dest="dsl_strict", action="store_false", help="Allow <VAL_NONE> in DSL tokens.")
    p.set_defaults(dsl_strict=False)
    p.add_argument(
        "--il-check",
        dest="il_check",
        action="store_true",
        help="Reject circuits with high insertion loss (default: off).",
    )
    p.add_argument(
        "--no-il-check",
        dest="il_check",
        action="store_false",
        help="Disable insertion loss rejection sanity check.",
    )
    p.set_defaults(il_check=False)
    p.add_argument(
        "--filter-type",
        choices=["lowpass", "highpass", "bandpass", "bandstop"],
        help="Fix filter_type for all samples (scenario must be compatible).",
    )
    p.add_argument(
        "--prototype-type",
        choices=["cheby1", "butter"],
        help="Fix prototype type for all samples.",
    )
    p.add_argument(
        "--topology-type",
        choices=["pi", "t"],
        help="Fix topology type for all samples.",
    )
    p.add_argument("--max-nodes", type=int, default=32, help="Max internal nodes after canonicalization (n1..nK).")
    p.add_argument(
        "--narrow-freq-grid",
        action="store_true",
        help="Use a narrow frequency grid around fc for LP/HP (BP/BS unchanged).",
    )
    p.add_argument(
        "--narrow-freq-span",
        type=float,
        default=0.5,
        help="Half-span around fc for narrow grid (f in [fc*(1-span), fc*(1+span)]).",
    )
    p.add_argument("--bp-order-lp", type=int, help="Force LP order for cascade BP (requires bandpass).")
    p.add_argument("--bp-order-hp", type=int, help="Force HP order for cascade BP (requires bandpass).")
    p.add_argument(
        "--bp-cascade-order",
        type=str,
        choices=["random", "lp_hp", "hp_lp"],
        help="Fix cascade order for BP (lp_hp or hp_lp).",
    )
    p.add_argument(
        "--spec-fixed",
        type=str,
        help=(
            "Optional JSON mapping to override spec fields, e.g. "
            "'{\"order\":4,\"fc_hz\":1e9,\"bw_frac\":0.2,\"ripple_db\":0.1}'."
        ),
    )
    p.add_argument(
        "--spec-ranges",
        type=str,
        help=(
            "Optional JSON mapping of ranges, e.g. "
            "'{\"order\":[2,6],\"fc_hz\":[1e8,1e9],\"bw_frac\":[0.05,0.2],\"ripple_db\":[0.1,0.5]}'."
        ),
    )
    p.add_argument(
        "--spec-profile",
        type=str,
        help=(
            "Optional JSON (or path to JSON) mapping scenario -> {fixed:{}, ranges:{}} "
            "for per-scenario specs."
        ),
    )
    p.add_argument(
        "--mask-mode",
        choices=["data", "spec"],
        default="data",
        help="Mask generation mode: data-driven masks or spec masks.",
    )
    p.add_argument(
        "--ensure-spec",
        action="store_true",
        help="Reject samples that do not satisfy spec masks (re-sample until enough samples).",
    )
    p.add_argument(
        "--ensure-spec-wave",
        choices=["real", "ideal"],
        default="real",
        help="Waveform used for spec compliance checks.",
    )
    p.add_argument(
        "--ensure-max-tries",
        type=int,
        default=0,
        help="Max attempts when --ensure-spec is on (0 = no limit).",
    )
    p.add_argument(
        "--ensure-spec-strategy",
        choices=["resample", "struct", "order", "mixed"],
        default="mixed",
        help="Strategy for spec-compliant generation when --ensure-spec is enabled.",
    )
    p.add_argument(
        "--ensure-struct-tries",
        type=int,
        default=2,
        help="Attempts to resample structure before falling back (mixed strategy).",
    )
    p.add_argument(
        "--ensure-order-tries",
        type=int,
        default=2,
        help="Attempts to resample order before falling back (mixed strategy).",
    )
    p.add_argument(
        "--ensure-order-bias",
        type=float,
        default=0.7,
        help="Bias toward higher order when resampling order (0..1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_l = args.q if args.q_l is None else args.q_l
    q_c = args.q if args.q_c is None else args.q_c
    scenario_weights = None
    if args.scenario_weights:
        try:
            scenario_weights = json.loads(args.scenario_weights)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --scenario-weights JSON: {exc}") from exc
    spec_fixed = _load_json_arg(args.spec_fixed, label="spec-fixed")
    spec_ranges = _load_json_arg(args.spec_ranges, label="spec-ranges")
    spec_profile = _load_json_arg(args.spec_profile, label="spec-profile")
    if args.bp_order_lp is not None or args.bp_order_hp is not None or args.bp_cascade_order is not None:
        if spec_fixed is None:
            spec_fixed = {}
        if args.bp_order_lp is not None:
            spec_fixed["bp_order_lp"] = int(args.bp_order_lp)
        if args.bp_order_hp is not None:
            spec_fixed["bp_order_hp"] = int(args.bp_order_hp)
        if args.bp_cascade_order is not None:
            spec_fixed["bp_cascade_order"] = str(args.bp_cascade_order)
    path = build_dataset(
        num_samples=args.num_samples,
        output_dir=str(args.output_dir),
        split=args.split,
        use_ngspice=bool(args.use_ngspice),
        seed=args.seed,
        scenario=str(args.scenario),
        scenario_weights=scenario_weights,
        emit_vact_tokens=bool(args.vact),
        emit_vact_cells=bool(args.vact_cell),
        emit_vact_struct=bool(args.vact_struct),
        emit_actions=bool(args.actions),
        emit_dsl=bool(args.dsl),
        emit_sfci=bool(args.sfci),
        sfci_value_mode=str(args.sfci_value_mode),
        dsl_include_order=bool(args.dsl_order),
        dsl_use_cell_indices=bool(args.dsl_cell_indices),
        dsl_strict=bool(args.dsl_strict),
        max_nodes=int(args.max_nodes),
        q_L=q_l,
        q_C=q_c,
        q_model=str(args.q_model),
        check_insertion_loss=bool(args.il_check),
        filter_type_override=args.filter_type,
        prototype_type_override=args.prototype_type,
        topology_type_override=args.topology_type,
        spec_fixed=spec_fixed,
        spec_ranges=spec_ranges,
        spec_profile=spec_profile,
        narrow_freq_grid=bool(args.narrow_freq_grid),
        narrow_freq_span=float(args.narrow_freq_span),
        quantize_series=None if str(args.quantize).lower() == "none" else str(args.quantize),
        mask_mode=str(args.mask_mode),
        ensure_spec=bool(args.ensure_spec),
        ensure_spec_wave=str(args.ensure_spec_wave),
        ensure_max_tries=int(args.ensure_max_tries),
        ensure_spec_strategy=str(args.ensure_spec_strategy),
        ensure_struct_tries=int(args.ensure_struct_tries),
        ensure_order_tries=int(args.ensure_order_tries),
        ensure_order_bias=float(args.ensure_order_bias),
    )
    print(f"Dataset written to {path}")


if __name__ == "__main__":
    main()
