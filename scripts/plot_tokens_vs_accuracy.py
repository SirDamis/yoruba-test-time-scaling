#!/usr/bin/env python3
"""Accuracy vs token cost for nested E2 pools (the TTC compute/accuracy frontier).

For every (dataset, group) it plots, against the total tokens spent per example
at each sample budget ``k``:

- ``pass@k``  - unbiased generation ceiling
- ``maj@k``   - random-subset majority vote (self-consistency)

One line per model, points labelled with ``k``. Use this to answer "does more
compute buy accuracy, and at what token cost?" per model, and to compare models
at matched budgets (read off the curves at the same x).

Reads an existing ``pool_estimates.json`` (from ``aggregate_ttc_metrics.py``) or
recomputes from ``runs/*/candidates.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.estimators import nested_pool_estimates
from ttcs_yoruba.io_utils import read_json, read_jsonl, write_csv
from ttcs_yoruba.metrics import dedupe_candidate_rows, find_run_dirs

X_FIELDS = {
    "total": ("total_tokens_per_example_at_k", "tokens / example (prompt + completion)"),
    "prompt": ("prompt_tokens_per_example_at_k", "prompt tokens / example"),
    "completion": ("completion_tokens_per_example_at_k", "completion tokens / example"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pass@k / maj@k accuracy against token cost for E2 pools."
    )
    parser.add_argument(
        "--pool-estimates",
        type=str,
        default=None,
        help="Read an existing pool_estimates.json instead of recomputing.",
    )
    parser.add_argument("--runs-dir", type=str, default="runs", help="Run directory root.")
    parser.add_argument("--run-id", type=str, default=None, help="Single run directory name.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ttc_scaling",
        help="Directory for the plot and CSV.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument(
        "--metric",
        choices=["pass", "maj", "both"],
        default="both",
        help="Which accuracy series to plot (default: both).",
    )
    parser.add_argument(
        "--x",
        choices=sorted(X_FIELDS),
        default="total",
        help="Token axis: total (default), prompt, or completion.",
    )
    parser.add_argument("--log-x", action="store_true", help="Log-scale the token axis.")
    parser.add_argument(
        "--ks",
        type=str,
        default=None,
        help="Comma-separated k values for recompute (default: powers of two up to pool size).",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=100,
        help="Random k-subsets per example for maj@k (default: 100).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for subset sampling.")
    return parser.parse_args()


def load_estimates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pool_estimates:
        payload = read_json(Path(args.pool_estimates))
        return list(payload.get("estimates", payload))
    runs_dir = Path(args.runs_dir)
    run_dirs = [runs_dir / args.run_id] if args.run_id else find_run_dirs(runs_dir, require_manifest=False)
    run_dirs = [path for path in run_dirs if path.exists()]
    candidates: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / "candidates.jsonl"
        if path.exists():
            candidates.extend(read_jsonl(path))
    candidates = dedupe_candidate_rows(candidates)
    ks = [int(x) for x in args.ks.split(",") if x.strip()] if args.ks else None
    return nested_pool_estimates(
        candidates, ks=ks, num_subsets=args.num_subsets, seed=args.seed
    )


def frontier_rows(
    estimates: list[dict[str, Any]], *, x_field: str
) -> list[dict[str, Any]]:
    """Flatten estimates into accuracy-vs-tokens rows, adding a per-series step cost."""
    rows: list[dict[str, Any]] = []
    for r in estimates:
        tokens = float(r.get(x_field) or 0.0)
        rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "nested_group_id": r["nested_group_id"],
                "pool_n": int(r["pool_n"]),
                "k": int(r["k"]),
                "num_examples": int(r["num_examples"]),
                "tokens_per_example": tokens,
                "pass_at_k": float(r["pass_at_k"]),
                "maj_at_k_mean": float(r["maj_at_k_mean"]),
            }
        )
    return rows


def _plot_panel(ax: Any, rows: list[dict[str, Any]], metric: str, x_label: str, log_x: bool) -> None:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    for model, points in sorted(by_model.items()):
        points = sorted(points, key=lambda p: (p["tokens_per_example"], p["k"]))
        xs = [p["tokens_per_example"] for p in points]
        if metric in ("pass", "both"):
            ax.plot(xs, [p["pass_at_k"] for p in points], marker="o", label=f"{model} pass@k")
        if metric in ("maj", "both"):
            ax.plot(
                xs,
                [p["maj_at_k_mean"] for p in points],
                marker="s",
                linestyle="--",
                label=f"{model} maj@k",
            )
        for p in points:
            ax.annotate(
                f"k={p['k']}",
                (p["tokens_per_example"], p["pass_at_k"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
            )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates = load_estimates(args)
    if args.dataset:
        estimates = [r for r in estimates if r.get("dataset") == args.dataset]
    if not estimates:
        print("No nested pool estimates found.", file=sys.stderr)
        sys.exit(1)

    x_field, x_label = X_FIELDS[args.x]
    rows = frontier_rows(estimates, x_field=x_field)
    csv_path = output_dir / "accuracy_vs_tokens_pool.csv"
    write_csv(
        csv_path,
        rows,
        [
            "dataset",
            "model",
            "nested_group_id",
            "pool_n",
            "k",
            "num_examples",
            "tokens_per_example",
            "pass_at_k",
            "maj_at_k_mean",
        ],
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            f"matplotlib not installed; wrote {csv_path} only "
            "(`uv pip install matplotlib` to plot).",
            file=sys.stderr,
        )
        return

    by_panel: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_panel[(row["dataset"], row["nested_group_id"])].append(row)

    panels = sorted(by_panel)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4.2 * len(panels)), squeeze=False)
    for ax, key in zip(axes[:, 0], panels):
        _plot_panel(ax, by_panel[key], args.metric, x_label, args.log_x)
        ax.set_title(f"{key[0]} · {key[1]}  ({args.metric} vs {args.x} tokens)")
    fig.tight_layout()
    out_path = output_dir / "accuracy_vs_tokens_pool.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_path}")
    for key, pts in sorted(by_panel.items()):
        for p in sorted(pts, key=lambda r: r["k"]):
            print(
                f"  {p['dataset']:12s} {p['model']:12s} {p['nested_group_id']:18s} "
                f"k={p['k']:<3d} tok/ex={p['tokens_per_example']:9.1f} "
                f"pass={p['pass_at_k']:.1%} maj={p['maj_at_k_mean']:.1%}"
            )


if __name__ == "__main__":
    main()
