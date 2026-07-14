#!/usr/bin/env python3
"""Offline E3 re-selection: first / majority vote on saved candidates.

Does not regenerate model traces. Reads ``runs/<id>/candidates.jsonl`` and writes:

- ``selections_<strategy>.jsonl`` per strategy
- ``e3_report.json`` with pass@N vs select@N tables

LLM-as-judge / external verifiers are deferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.metrics import find_run_dirs
from ttcs_yoruba.reselection import print_e3_report_table, reselect_run_dir
from ttcs_yoruba.selection import SUPPORTED_SELECTIONS


def parse_strategies(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise ValueError("At least one strategy is required")
    unknown = [s for s in strategies if s not in SUPPORTED_SELECTIONS]
    if unknown:
        raise ValueError(
            f"Unknown strategies: {unknown}. Supported: {sorted(SUPPORTED_SELECTIONS)}"
        )
    return strategies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline re-selection for E3 (generation vs selection)."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory containing run subdirectories.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Single run directory name. Default: all runs with candidates.jsonl.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="first,majority_vote",
        help="Comma-separated strategies: first, majority_vote.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/e3_reselection",
        help="Root output directory for reselection artifacts.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Optional cap on number of run directories processed.",
    )
    args = parser.parse_args()

    strategies = parse_strategies(args.strategies)
    runs_dir = Path(args.runs_dir)
    output_root = Path(args.output_dir)

    if args.run_id:
        run_dirs = [runs_dir / args.run_id]
    else:
        run_dirs = find_run_dirs(runs_dir)

    if args.limit_runs is not None:
        run_dirs = run_dirs[: args.limit_runs]

    run_dirs = [path for path in run_dirs if path.exists()]
    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    all_reports = []
    for run_dir in run_dirs:
        out_dir = output_root / run_dir.name
        print(f"Re-selecting {run_dir.name} -> {out_dir}")
        report = reselect_run_dir(
            run_dir,
            strategies=strategies,
            output_dir=out_dir,
        )
        print(f"  groups: {report['num_groups']}  conditions: {report['num_conditions']}")
        print(f"  report: {report['report_path']}")
        print_e3_report_table(report["conditions"], strategies)
        all_reports.append(
            {
                "run_id": report["run_id"],
                "source_run_dir": report["source_run_dir"],
                "report_path": report["report_path"],
                "num_conditions": report["num_conditions"],
                "conditions": report["conditions"],
            }
        )

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {"strategies": strategies, "runs": all_reports},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
