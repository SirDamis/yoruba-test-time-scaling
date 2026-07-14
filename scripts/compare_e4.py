#!/usr/bin/env python3
"""E4: compare small model + TTC against larger model greedy.

Sources metrics either from:
  - aggregated E2 JSON (``results/ttc_scaling/metrics.json``), or
  - raw run directories under ``runs/``

Writes paper tables and efficiency plots under ``results/e4_comparison/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.e4_compare import (
    E4ComparisonConfig,
    build_e4_comparison,
    load_metrics_from_json,
    load_metrics_from_runs,
    print_e4_table,
    save_e4_report,
)
from ttcs_yoruba.io_utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 small+TTC vs large greedy comparison.")
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="Path to aggregated metrics JSON from aggregate_ttc_metrics.py.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Run directory root if not using --metrics-json.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional single run id when loading from --runs-dir.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/e4_comparison.json",
        help="E4 comparison config (small/large model names).",
    )
    parser.add_argument(
        "--small-model",
        type=str,
        default=None,
        help="Override small model name (default from config: qwen3-4b).",
    )
    parser.add_argument(
        "--large-model",
        type=str,
        default=None,
        help="Override large model name (default from config: qwen3-32b).",
    )
    parser.add_argument(
        "--large-n",
        type=int,
        default=None,
        help="Greedy N for large model baseline (default 1).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset filter.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/e4_comparison",
        help="Output directory for E4 tables and plots.",
    )
    parser.add_argument(
        "--small-method",
        type=str,
        default=None,
        help="Method filter for small model (exact, prefix, or glob*). Default from config.",
    )
    parser.add_argument(
        "--large-method",
        type=str,
        default=None,
        help="Method filter for large model baseline. Default from config.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def load_e4_config(path: Path | None) -> E4ComparisonConfig:
    if path is None or not path.exists():
        return E4ComparisonConfig()
    payload = read_json(path)
    include = payload.get("include_models") or []
    datasets = payload.get("datasets")
    return E4ComparisonConfig(
        small_model=str(payload.get("small_model", "qwen3-4b")),
        large_model=str(payload.get("large_model", "qwen3-32b")),
        large_n=int(payload.get("large_n", 1)),
        include_models=tuple(str(x) for x in include),
        datasets=None if datasets is None else tuple(str(x) for x in datasets),
        small_method=payload.get("small_method", "english_cot_ttc"),
        large_method=payload.get("large_method", "english_cot_ttc"),
        ladder_method=payload.get("ladder_method"),
    )


def main() -> None:
    args = parse_args()
    cfg = load_e4_config(Path(args.config) if args.config else None)
    datasets = cfg.datasets
    if args.datasets:
        datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    cfg = E4ComparisonConfig(
        small_model=args.small_model or cfg.small_model,
        large_model=args.large_model or cfg.large_model,
        large_n=args.large_n if args.large_n is not None else cfg.large_n,
        include_models=cfg.include_models,
        datasets=datasets,
        small_method=args.small_method if args.small_method is not None else cfg.small_method,
        large_method=args.large_method if args.large_method is not None else cfg.large_method,
        ladder_method=cfg.ladder_method,
    )

    if args.metrics_json:
        metrics = load_metrics_from_json(Path(args.metrics_json))
        source = args.metrics_json
    else:
        metrics = load_metrics_from_runs(Path(args.runs_dir), run_id=args.run_id)
        source = f"{args.runs_dir}" + (f"/{args.run_id}" if args.run_id else "/*")

    if not metrics:
        print(f"No metrics found from {source}", file=sys.stderr)
        sys.exit(1)

    report = build_e4_comparison(metrics, config=cfg)
    report["source"] = source

    output_dir = Path(args.output_dir)
    paths = save_e4_report(report, output_dir, make_plots=not args.no_plots)

    print(f"Loaded {len(metrics)} conditions from {source}")
    print(f"Small={cfg.small_model}  Large={cfg.large_model}@N={cfg.large_n}")
    print_e4_table(report)
    print("\nWrote:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
