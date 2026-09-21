#!/usr/bin/env python3
"""Aggregate E2 TTC metrics and optionally plot scaling curves.

Reads ``runs/*/candidates.jsonl`` + ``selections.jsonl`` and writes:

- JSON metrics table
- CSV metrics table
- Pool estimates (unbiased pass@k + random-subset maj@k + diversity) for nested runs
- Accuracy vs N plot(s)
- Accuracy vs total tokens plot(s)

The prefix-based ``select@N`` in the main table is order-sensitive for majority
vote; the pool estimates use each example's full max-N pool with random k-subsets
(``estimators.nested_pool_estimates``) and are the statistically stronger read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.estimators import nested_pool_estimates
from ttcs_yoruba.io_utils import read_jsonl, write_csv, write_json
from ttcs_yoruba.metrics import (
    aggregate_runs,
    dedupe_candidate_rows,
    find_run_dirs,
    plot_accuracy_vs_n,
    plot_accuracy_vs_tokens,
    write_metrics_csv,
)

TRUNCATION_WARN_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TTC scaling metrics (E2).")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional single run directory name. Default: all runs with candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ttc_scaling",
        help="Directory for metrics tables and plots.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset filter for plots (metrics table still includes all).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge identical conditions across runs.",
    )
    parser.add_argument(
        "--no-pool-estimates",
        action="store_true",
        help="Skip the unbiased pass@k / random-subset maj@k pool tables.",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=20,
        help="Random k-subsets per example for pool maj@k (default: 20).",
    )
    parser.add_argument(
        "--pool-seed",
        type=int,
        default=0,
        help="RNG seed for random-subset maj@k sampling (default: 0).",
    )
    return parser.parse_args()


POOL_ESTIMATE_FIELDS = [
    "dataset",
    "model",
    "nested_group_id",
    "pool_n",
    "k",
    "num_examples",
    "pass_at_k",
    "maj_at_k_mean",
    "maj_at_k_between_example_std",
    "maj_at_k_subset_std_mean",
    "distinct_answers_mean",
    "degenerate_pool_rate",
    "mean_completion_tokens_per_sample",
    "completion_tokens_per_example_at_k",
    "completion_tokens_per_correct",
]


def print_pool_table(rows: list[dict]) -> None:
    """Print the pool-based pass@k / maj@k estimates for nested E2 runs."""
    if not rows:
        return
    headers = [
        "Dataset",
        "Model",
        "Group",
        "PoolN",
        "k",
        "n",
        "pass@k",
        "maj@k",
        "maj_sd",
        "distinct",
        "degen%",
    ]
    widths = [12, 12, 20, 5, 4, 5, 8, 8, 7, 9, 7]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for r in rows:
        print(
            "  ".join(
                [
                    str(r["dataset"])[: widths[0]].ljust(widths[0]),
                    str(r["model"])[: widths[1]].ljust(widths[1]),
                    str(r["nested_group_id"])[: widths[2]].ljust(widths[2]),
                    str(r["pool_n"]).rjust(widths[3]),
                    str(r["k"]).rjust(widths[4]),
                    str(r["num_examples"]).rjust(widths[5]),
                    f"{r['pass_at_k']:.1%}".rjust(widths[6]),
                    f"{r['maj_at_k_mean']:.1%}".rjust(widths[7]),
                    f"{r['maj_at_k_subset_std_mean']:.3f}".rjust(widths[8]),
                    f"{r['distinct_answers_mean']:.2f}".rjust(widths[9]),
                    f"{r['degenerate_pool_rate']:.1%}".rjust(widths[10]),
                ]
            )
        )


def write_pool_estimates(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    num_subsets: int,
    seed: int,
) -> list[dict]:
    """Write pool_estimates.json/csv from every nested pool across the runs."""
    candidates: list[dict] = []
    for run_dir in run_dirs:
        path = run_dir / "candidates.jsonl"
        if path.exists():
            candidates.extend(read_jsonl(path))
    candidates = dedupe_candidate_rows(candidates)
    rows = nested_pool_estimates(candidates, num_subsets=num_subsets, seed=seed)
    if not rows:
        return []
    payload = {
        "runs": [str(path) for path in run_dirs],
        "num_subsets": num_subsets,
        "seed": seed,
        "estimates": rows,
    }
    write_json(output_dir / "pool_estimates.json", payload)
    write_csv(output_dir / "pool_estimates.csv", rows, POOL_ESTIMATE_FIELDS)
    print(f"  Pool estimates JSON: {output_dir / 'pool_estimates.json'}")
    print(f"  Pool estimates CSV:  {output_dir / 'pool_estimates.csv'}")
    print_pool_table(rows)
    return rows


def print_table(metrics) -> None:
    headers = [
        "Dataset",
        "Model",
        "Method",
        "N",
        "Acc",
        "pass@N",
        "Trunc%",
        "Empty%",
        "Tokens",
        "Tok/ex",
        "PTok/ex",
        "Lat/ex(s)",
    ]
    widths = [12, 16, 22, 4, 7, 7, 7, 7, 8, 8, 8, 9]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + header_line)
    print("-" * len(header_line))
    for m in metrics:
        trunc = m.truncation_rate
        trunc_s = "—" if trunc is None else f"{trunc:.1%}"
        empty = m.empty_extraction_rate
        empty_s = "—" if empty is None else f"{empty:.1%}"
        row = [
            m.dataset[: widths[0]].ljust(widths[0]),
            m.model[: widths[1]].ljust(widths[1]),
            m.method[: widths[2]].ljust(widths[2]),
            str(m.n).rjust(widths[3]),
            f"{m.accuracy:.1%}".rjust(widths[4]),
            f"{m.pass_at_n_rate:.1%}".rjust(widths[5]),
            trunc_s.rjust(widths[6]),
            empty_s.rjust(widths[7]),
            str(m.total_tokens).rjust(widths[8]),
            f"{m.mean_tokens_per_example:.1f}".rjust(widths[9]),
            f"{m.mean_prompt_tokens_per_example:.1f}".rjust(widths[10]),
            f"{m.mean_latency_s_per_example:.2f}".rjust(widths[11]),
        ]
        print(sep.join(row))


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_id:
        run_dirs = [runs_dir / args.run_id]
    else:
        run_dirs = find_run_dirs(runs_dir)

    run_dirs = [path for path in run_dirs if path.exists()]
    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    metrics = aggregate_runs(run_dirs, merge=not args.no_merge)
    if args.dataset:
        plot_metrics = [m for m in metrics if m.dataset == args.dataset]
    else:
        plot_metrics = metrics

    payload = {
        "runs": [str(path) for path in run_dirs],
        "num_conditions": len(metrics),
        "conditions": [m.to_dict() for m in metrics],
    }
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    write_json(json_path, payload)
    write_metrics_csv(csv_path, metrics)

    print(f"Aggregated {len(metrics)} conditions from {len(run_dirs)} run(s).")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print_table(metrics)

    truncated = [
        m for m in metrics if (m.truncation_rate or 0.0) > TRUNCATION_WARN_THRESHOLD
    ]
    if truncated:
        print(
            f"\nWARNING: truncation_rate > {TRUNCATION_WARN_THRESHOLD:.0%} — "
            "pass@N/select@N are biased downward for:"
        )
        for m in truncated:
            print(
                f"  {m.dataset} {m.model} {m.method} n={m.n}: "
                f"{m.truncation_rate:.1%} ({m.truncated_candidates}/{m.total_candidates})"
            )

    if not args.no_pool_estimates:
        write_pool_estimates(
            run_dirs,
            output_dir,
            num_subsets=args.num_subsets,
            seed=args.pool_seed,
        )

    if args.no_plots:
        return

    datasets = sorted({m.dataset for m in plot_metrics if m.dataset})
    # Overall plots (all datasets as separate series labels) + per-dataset plots.
    acc_n = plot_accuracy_vs_n(plot_metrics, output_dir / "accuracy_vs_n.png")
    acc_tok = plot_accuracy_vs_tokens(plot_metrics, output_dir / "accuracy_vs_tokens.png")
    if acc_n is None or acc_tok is None:
        print(
            "\nPlots skipped: install matplotlib to enable figures "
            "(`uv pip install matplotlib`).",
            file=sys.stderr,
        )
    else:
        print(f"  Plot: {acc_n}")
        print(f"  Plot: {acc_tok}")

    if len(datasets) > 1 and args.dataset is None:
        for dataset in datasets:
            subset = [m for m in metrics if m.dataset == dataset]
            p1 = plot_accuracy_vs_n(
                subset, output_dir / f"accuracy_vs_n_{dataset}.png", dataset=dataset
            )
            p2 = plot_accuracy_vs_tokens(
                subset, output_dir / f"accuracy_vs_tokens_{dataset}.png", dataset=dataset
            )
            if p1:
                print(f"  Plot: {p1}")
            if p2:
                print(f"  Plot: {p2}")


if __name__ == "__main__":
    main()
