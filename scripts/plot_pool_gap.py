#!/usr/bin/env python3
"""Plot the generation-vs-selection gap for nested E2 pools.

For every (dataset, model, nested group) it plots, against the number of samples
``k``:

- ``pass@k``  - unbiased generation ceiling (any of k samples correct)
- ``maj@k``   - random-subset majority-vote accuracy (self-consistency)
- the shaded gap between them

A widening ``pass@k - maj@k`` gap means more sampling surfaces correct answers
that the selector fails to pick: a selection bottleneck. A flat/small gap means
generation is the constraint.

Reads an existing ``pool_estimates.json`` (from ``aggregate_ttc_metrics.py``) or
recomputes the estimates from ``runs/*/candidates.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.estimators import nested_pool_estimates
from ttcs_yoruba.io_utils import read_json, read_jsonl, write_csv
from ttcs_yoruba.metrics import dedupe_candidate_rows, find_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pass@k vs maj@k (generation vs selection gap) for E2 pools."
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
        help="Directory for the plots and gap CSV.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
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


def gap_rows(estimates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten estimates to rows carrying the pass@k - maj@k gap."""
    rows: list[dict[str, Any]] = []
    for row in estimates:
        gap = float(row["pass_at_k"]) - float(row["maj_at_k_mean"])
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "nested_group_id": row["nested_group_id"],
                "pool_n": int(row["pool_n"]),
                "k": int(row["k"]),
                "num_examples": int(row["num_examples"]),
                "pass_at_k": float(row["pass_at_k"]),
                "maj_at_k_mean": float(row["maj_at_k_mean"]),
                "gap": gap,
            }
        )
    return rows


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _plot_group(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    pass_vals = [r["pass_at_k"] for r in rows]
    maj_vals = [r["maj_at_k_mean"] for r in rows]

    ax.plot(ks, pass_vals, marker="o", color="#1f77b4", label="pass@k (generation ceiling)")
    ax.plot(ks, maj_vals, marker="s", linestyle="--", color="#d62728", label="maj@k (self-consistency)")
    ax.fill_between(ks, maj_vals, pass_vals, color="#ff7f0e", alpha=0.2, label="selection gap")

    max_k = ks[-1]
    gap = pass_vals[-1] - maj_vals[-1]
    ax.annotate(
        f"gap@k={max_k}: {gap:.0%}",
        xy=(max_k, (pass_vals[-1] + maj_vals[-1]) / 2),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color="#b35900",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(min(ks) * 0.9, max(ks) * 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("k (samples)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
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

    rows = gap_rows(estimates)
    csv_path = output_dir / "pass_maj_gap.csv"
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
            "pass_at_k",
            "maj_at_k_mean",
            "gap",
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

    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["dataset"], row["model"], row["nested_group_id"])].append(row)

    groups = sorted(by_group)
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 4.2 * len(groups)), squeeze=False)
    for ax, key in zip(axes[:, 0], groups):
        dataset, model, group = key
        group_rows = by_group[key]
        pool_n = group_rows[0]["pool_n"]
        n_examples = group_rows[0]["num_examples"]
        _plot_group(ax, group_rows, f"{dataset} · {model} · {group}\n(pool N={pool_n}, examples={n_examples})")
    fig.tight_layout()
    combined_path = output_dir / "pass_vs_maj_gap.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    for key in groups:
        dataset, model, group = key
        single = plt.subplots(figsize=(8, 5))
        fig1, ax1 = single
        _plot_group(ax1, by_group[key], f"{dataset} · {model} · {group}")
        fig1.tight_layout()
        out_path = output_dir / f"pass_vs_maj_gap_{_safe(dataset)}_{_safe(model)}_{_safe(group)}.png"
        fig1.savefig(out_path, dpi=150)
        plt.close(fig1)

    print(f"Wrote {csv_path}")
    print(f"Wrote {combined_path}")
    print("Gap (pass@k - maj@k) at each k:")
    for key, group_rows in sorted(by_group.items()):
        for row in sorted(group_rows, key=lambda r: r["k"]):
            print(
                f"  {row['dataset']:12s} {row['model']:12s} {row['nested_group_id']:20s} "
                f"k={row['k']:<3d} pass={row['pass_at_k']:.1%} maj={row['maj_at_k_mean']:.1%} "
                f"gap={row['gap']:.1%}"
            )


if __name__ == "__main__":
    main()
