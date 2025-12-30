"""Plot summary statistics for eval.py --dump JSONL outputs.

Example:
  python scripts/plot_eval_dump_summary.py \
    --jsonl outputs/eval_100k_t5base_sample.jsonl \
    --output outputs/eval_100k_t5base_sample_summary.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _as_float_array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    out.fill(np.nan)
    for i, r in enumerate(rows):
        v = r.get(key)
        try:
            out[i] = float(v)
        except Exception:
            out[i] = np.nan
    return out


def _as_str_list(rows: Sequence[Mapping[str, Any]], key: str) -> List[str]:
    out: List[str] = []
    for r in rows:
        v = r.get(key)
        out.append(str(v) if v is not None else "unknown")
    return out


def compute_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    taus: Sequence[float],
    tau_bar: float,
) -> Dict[str, Any]:
    n_total = int(len(rows))
    errors = _as_float_array(rows, "best_error")
    num_components = _as_float_array(rows, "num_components")
    filter_types = _as_str_list(rows, "filter_type")

    finite_err = np.isfinite(errors)
    n_finite = int(np.sum(finite_err))
    n_invalid = int(n_total - n_finite)

    if n_total == 0:
        return {
            "n_total": 0,
            "n_finite_error": 0,
            "n_invalid_error": 0,
            "median_error": float("nan"),
            "median_num_components": float("nan"),
            "success_at": {float(t): float("nan") for t in taus},
            "filter_type_counts": {},
            "filter_type_success_at_tau_bar": {},
        }

    success_at = {}
    for tau in taus:
        tau_f = float(tau)
        success_at[tau_f] = float(np.sum(finite_err & (errors <= tau_f)) / n_total)

    median_error = float(np.median(errors[finite_err])) if n_finite else float("nan")

    finite_comp = np.isfinite(num_components) & finite_err
    median_num_components = float(np.median(num_components[finite_comp])) if np.any(finite_comp) else float("nan")

    ft_counts = Counter(filter_types)
    ft_success = {}
    tau_bar = float(tau_bar)
    for t, ct in ft_counts.items():
        idxs = [i for i, tt in enumerate(filter_types) if tt == t]
        if not idxs:
            continue
        e = errors[idxs]
        ok = np.isfinite(e) & (e <= tau_bar)
        ft_success[t] = float(np.sum(ok) / len(idxs))

    return {
        "n_total": n_total,
        "n_finite_error": n_finite,
        "n_invalid_error": n_invalid,
        "median_error": median_error,
        "median_num_components": median_num_components,
        "taus": [float(t) for t in taus],
        "tau_bar": tau_bar,
        "success_at": success_at,
        "filter_type_counts": dict(ft_counts),
        "filter_type_success_at_tau_bar": ft_success,
    }


def plot_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    title: str,
    output: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    errors = _as_float_array(rows, "best_error")
    num_components = _as_float_array(rows, "num_components")
    filter_types = _as_str_list(rows, "filter_type")

    n_total = int(metrics["n_total"])
    n_invalid = int(metrics["n_invalid_error"])
    taus = [float(t) for t in metrics["taus"]]
    tau_bar = float(metrics["tau_bar"])
    success_at: Mapping[float, float] = metrics["success_at"]

    finite_err = np.isfinite(errors)
    finite_errors = np.sort(errors[finite_err])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_err_hist, ax_err_cdf, ax_comp_hist, ax_ft_bar = axes.ravel()

    # best_error histogram
    if finite_errors.size:
        ax_err_hist.hist(finite_errors, bins=30, range=(0.0, 1.0), color="tab:blue", alpha=0.85)
        med = float(metrics["median_error"])
        if np.isfinite(med):
            ax_err_hist.axvline(med, color="k", linestyle="--", linewidth=1, label=f"median={med:.3f}")
        for tau in taus:
            ax_err_hist.axvline(float(tau), color="tab:red", linestyle=":", linewidth=1)
    ax_err_hist.set_title("best_error histogram")
    ax_err_hist.set_xlabel("best_error")
    ax_err_hist.set_ylabel("count")
    ax_err_hist.grid(True, which="both", ls=":", alpha=0.6)
    if finite_errors.size:
        ax_err_hist.legend(loc="upper right")

    # best_error CDF (normalized by total N to reflect invalid samples)
    if finite_errors.size:
        y = np.arange(1, finite_errors.size + 1, dtype=float) / max(float(n_total), 1.0)
        ax_err_cdf.plot(finite_errors, y, color="tab:blue", linewidth=2)
        for tau in taus:
            tau_f = float(tau)
            s = float(success_at.get(tau_f, float("nan")))
            if np.isfinite(s):
                ax_err_cdf.scatter([tau_f], [s], color="tab:red", s=35, zorder=3)
                ax_err_cdf.text(tau_f, s, f" {s:.3f}", va="center", fontsize=9)
    ax_err_cdf.set_title("best_error CDF")
    ax_err_cdf.set_xlabel("best_error")
    ax_err_cdf.set_ylabel("P(error ≤ x)")
    ax_err_cdf.set_xlim(0.0, 1.0)
    ax_err_cdf.set_ylim(0.0, 1.0)
    ax_err_cdf.grid(True, which="both", ls=":", alpha=0.6)
    if n_invalid:
        ax_err_cdf.text(
            0.98,
            0.02,
            f"invalid={n_invalid}/{n_total}",
            ha="right",
            va="bottom",
            fontsize=9,
            transform=ax_err_cdf.transAxes,
        )

    # num_components histogram (finite errors only)
    finite_comp = np.isfinite(num_components) & finite_err
    finite_comps = num_components[finite_comp].astype(int) if np.any(finite_comp) else np.array([], dtype=int)
    if finite_comps.size:
        lo = int(np.min(finite_comps))
        hi = int(np.max(finite_comps))
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        ax_comp_hist.hist(finite_comps, bins=bins, color="tab:green", alpha=0.85)
        medc = float(metrics["median_num_components"])
        if np.isfinite(medc):
            ax_comp_hist.axvline(medc, color="k", linestyle="--", linewidth=1, label=f"median={medc:.1f}")
    ax_comp_hist.set_title("num_components histogram (valid only)")
    ax_comp_hist.set_xlabel("num_components")
    ax_comp_hist.set_ylabel("count")
    ax_comp_hist.grid(True, which="both", ls=":", alpha=0.6)
    if finite_comps.size:
        ax_comp_hist.legend(loc="upper right")

    # success@tau_bar by filter_type
    ft_counts = Counter(filter_types)
    # Keep a stable order for readability.
    ft_order = ["lowpass", "highpass", "bandpass", "bandstop"]
    types = [t for t in ft_order if t in ft_counts] + [t for t in sorted(ft_counts) if t not in ft_order]
    rates = []
    for t in types:
        idxs = [i for i, tt in enumerate(filter_types) if tt == t]
        e = errors[idxs]
        ok = np.isfinite(e) & (e <= tau_bar)
        rates.append(float(np.sum(ok) / len(idxs)) if idxs else float("nan"))

    ax_ft_bar.bar(types, rates, color="tab:purple", alpha=0.85)
    overall = float(success_at.get(float(tau_bar), float("nan")))
    if np.isfinite(overall):
        ax_ft_bar.axhline(overall, color="k", linestyle="--", linewidth=1, alpha=0.6, label=f"overall={overall:.3f}")
    for i, t in enumerate(types):
        ax_ft_bar.text(i, min(max(rates[i], 0.0) + 0.03, 0.98), f"n={ft_counts[t]}", ha="center", fontsize=9)
    ax_ft_bar.set_title(f"success@{tau_bar:g} by filter_type")
    ax_ft_bar.set_ylabel("rate")
    ax_ft_bar.set_ylim(0.0, 1.0)
    ax_ft_bar.grid(True, axis="y", ls=":", alpha=0.6)
    ax_ft_bar.tick_params(axis="x", rotation=20)
    if np.isfinite(overall):
        ax_ft_bar.legend(loc="upper right")

    med_err = float(metrics["median_error"])
    med_comp = float(metrics["median_num_components"])
    succ_005 = float(success_at.get(0.05, float("nan")))
    summary = f"N={n_total} invalid={n_invalid}/{n_total} median_error={med_err:.3f} median_num_components={med_comp:.1f}"
    if np.isfinite(succ_005):
        summary += f" success@0.05={succ_005:.3f}"
    fig.suptitle(f"{title}\n{summary}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot summary stats for eval.py --dump JSONL outputs.")
    p.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to eval dump jsonl (e.g., outputs/eval_100k_t5base_sample.jsonl).",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (png). If omitted, opens a window.",
    )
    p.add_argument(
        "--taus",
        type=str,
        default="0.01,0.02,0.05",
        help="Comma-separated τ thresholds to annotate (default: 0.01,0.02,0.05).",
    )
    p.add_argument(
        "--tau-bar",
        type=float,
        default=0.05,
        help="τ used for the filter_type bar chart (default: 0.05).",
    )
    p.add_argument("--title", type=str, help="Optional custom title (defaults to filename).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.jsonl)
    taus = _parse_csv_floats(args.taus)
    if not taus:
        taus = [0.05]
    tau_bar = float(args.tau_bar)
    if tau_bar not in taus:
        taus = sorted(set([*taus, tau_bar]))

    metrics = compute_metrics(rows, taus=taus, tau_bar=tau_bar)

    title = args.title or args.jsonl.name
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    plot_summary(rows=rows, metrics=metrics, title=title, output=args.output)


if __name__ == "__main__":
    main()

