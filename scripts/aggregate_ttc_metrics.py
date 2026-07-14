#!/usr/bin/env python3
"""Aggregate E2 TTC metrics and optionally plot scaling curves.

Reads ``runs/*/candidates.jsonl`` + ``selections.jsonl`` and writes:

- JSON metrics table
- CSV metrics table
- Accuracy vs N plot(s)
- Accuracy vs total tokens plot(s)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.io_utils import write_json
from ttcs_yoruba.metrics import (
    aggregate_runs,
    find_run_dirs,
    plot_accuracy_vs_n,
    plot_accuracy_vs_tokens,
    write_metrics_csv,
)


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
    return parser.parse_args()


def print_table(metrics) -> None:
    headers = [
        "Dataset",
        "Model",
        "Method",
        "N",
        "Acc",
        "pass@N",
        "Tokens",
        "Tok/ex",
        "Lat/ex(s)",
    ]
    widths = [12, 16, 22, 4, 7, 7, 8, 8, 9]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + header_line)
    print("-" * len(header_line))
    for m in metrics:
        row = [
            m.dataset[: widths[0]].ljust(widths[0]),
            m.model[: widths[1]].ljust(widths[1]),
            m.method[: widths[2]].ljust(widths[2]),
            str(m.n).rjust(widths[3]),
            f"{m.accuracy:.1%}".rjust(widths[4]),
            f"{m.pass_at_n_rate:.1%}".rjust(widths[5]),
            str(m.total_tokens).rjust(widths[6]),
            f"{m.mean_tokens_per_example:.1f}".rjust(widths[7]),
            f"{m.mean_latency_s_per_example:.2f}".rjust(widths[8]),
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
