"""Plot best-of-K success rates from eval.py stdout logs.

Parses lines like:
  success@0.05 best-of-16: 67/200 = 0.335

Usage (stdin):
  cat outputs/eval_100k_t5base_sample.log | python scripts/plot_eval_success_rates.py --output outputs/success.png

Usage (file):
  python scripts/plot_eval_success_rates.py --input outputs/eval_100k_t5base_sample.log --output outputs/success.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

_LINE_RE = re.compile(
    r"success@(?P<tau>[0-9]*\.?[0-9]+)\s+best-of-(?P<k>[0-9]+):\s+(?P<num>[0-9]+)/(?P<den>[0-9]+)\s+=\s+(?P<rate>[0-9]*\.?[0-9]+)"
)


def _read_lines(path: Path | None) -> List[str]:
    if path is None:
        return [ln.rstrip("\n") for ln in sys.stdin.read().splitlines()]
    return [ln.rstrip("\n") for ln in path.read_text().splitlines()]


def parse_success_table(lines: Iterable[str]) -> Tuple[Dict[float, Dict[int, float]], int | None]:
    table: Dict[float, Dict[int, float]] = {}
    den: int | None = None
    for line in lines:
        m = _LINE_RE.search(line)
        if not m:
            continue
        tau = float(m.group("tau"))
        k = int(m.group("k"))
        num = int(m.group("num"))
        den_i = int(m.group("den"))
        rate = float(m.group("rate"))
        # Sanity: prefer the computed ratio if log had rounding.
        rate = float(num / den_i) if den_i > 0 else rate
        table.setdefault(tau, {})[k] = rate
        if den is None:
            den = den_i
        elif den != den_i:
            den = None
    return table, den


def plot_success_vs_k(
    *,
    table: Dict[float, Dict[int, float]],
    title: str,
    output: Path | None,
    show_values: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if not table:
        raise SystemExit("No success@tau best-of-K lines found in input.")

    taus = sorted(table.keys())
    ks = sorted({k for by_k in table.values() for k in by_k.keys()})
    if not ks:
        raise SystemExit("No K values parsed from input.")

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
    for tau in taus:
        by_k = table[tau]
        y = [by_k.get(k, float("nan")) for k in ks]
        ax.plot(ks, y, marker="o", linewidth=2, label=f"τ={tau:g}")
        if show_values:
            for k, yy in zip(ks, y):
                if not np.isfinite(yy):
                    continue
                ax.text(k, yy, f" {yy:.3f}", va="center", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("best-of-K")
    ax.set_ylabel("success rate")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot best-of-K success rates from eval.py logs.")
    p.add_argument("--input", type=Path, help="Optional log file. If omitted, reads from stdin.")
    p.add_argument("--output", type=Path, help="Optional output PNG path. If omitted, opens a window.")
    p.add_argument("--title", type=str, default="best-of-K success rates", help="Figure title.")
    p.add_argument("--no-values", action="store_true", help="Do not annotate point values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lines = _read_lines(args.input)
    table, den = parse_success_table(lines)
    title = args.title
    if den is not None:
        title = f"{title} (N={den})"
    plot_success_vs_k(table=table, title=title, output=args.output, show_values=not args.no_values)


if __name__ == "__main__":
    main()
