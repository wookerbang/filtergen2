"""Simple CLI to run the data generation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import build_dataset


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
        help="Scenario template (general/anti_jamming/coexistence/wideband_rejection/random_basic) or 'random'.",
    )
    p.add_argument("--q", type=float, default=50.0, help="Finite-Q loss model (applied to both L and C unless overridden).")
    p.add_argument("--q-l", type=float, default=None, help="Override Q for inductors (None -> use --q).")
    p.add_argument("--q-c", type=float, default=None, help="Override Q for capacitors (None -> use --q).")
    p.add_argument("--tol", type=float, default=0.05, help="Component tolerance fraction for input waveforms (e.g. 0.05 = ±5%%).")
    p.add_argument(
        "--q-model",
        type=str,
        default="freq_dependent",
        choices=["freq_dependent", "fixed_ref"],
        help="Q modeling for real waveforms: freq_dependent (Fast Track) or fixed_ref (SPICE-style).",
    )
    p.add_argument("--vact-cell", dest="vact_cell", action="store_true", help="Insert <CELL> markers in VACT.")
    p.add_argument("--no-vact-cell", dest="vact_cell", action="store_false", help="Disable <CELL> markers in VACT.")
    p.set_defaults(vact_cell=True)
    p.add_argument("--vact-struct", dest="vact_struct", action="store_true", help="Emit VACT-Struct tokens (default: on).")
    p.add_argument("--no-vact-struct", dest="vact_struct", action="store_false", help="Disable VACT-Struct token emission.")
    p.set_defaults(vact_struct=True)
    p.add_argument("--actions", dest="actions", action="store_true", help="Emit action-construction tokens (default: on).")
    p.add_argument("--no-actions", dest="actions", action="store_false", help="Disable action-construction tokens.")
    p.set_defaults(actions=True)
    p.add_argument("--dsl", dest="dsl", action="store_true", help="Emit DSL tokens (macro/repeat).")
    p.add_argument("--no-dsl", dest="dsl", action="store_false", help="Disable DSL token emission.")
    p.set_defaults(dsl=True)
    p.add_argument(
        "--il-check",
        dest="il_check",
        action="store_true",
        help="Reject circuits with high insertion loss (default: on).",
    )
    p.add_argument(
        "--no-il-check",
        dest="il_check",
        action="store_false",
        help="Disable insertion loss rejection sanity check.",
    )
    p.set_defaults(il_check=True)
    p.add_argument("--max-nodes", type=int, default=32, help="Max internal nodes after canonicalization (n1..nK).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_l = args.q if args.q_l is None else args.q_l
    q_c = args.q if args.q_c is None else args.q_c
    path = build_dataset(
        num_samples=args.num_samples,
        output_dir=str(args.output_dir),
        split=args.split,
        use_ngspice=bool(args.use_ngspice),
        seed=args.seed,
        scenario=str(args.scenario),
        emit_vact_cells=bool(args.vact_cell),
        emit_vact_struct=bool(args.vact_struct),
        emit_actions=bool(args.actions),
        emit_dsl=bool(args.dsl),
        max_nodes=int(args.max_nodes),
        q_L=q_l,
        q_C=q_c,
        tol_frac=float(args.tol),
        q_model=str(args.q_model),
        check_insertion_loss=bool(args.il_check),
    )
    print(f"Dataset written to {path}")


if __name__ == "__main__":
    main()
