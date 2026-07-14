from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.io_utils import write_json
from ttcs_yoruba.metrics import aggregate_run_dir, find_run_dirs


def print_summary_table(all_conditions: list[dict]) -> None:
    headers = ["Run", "Dataset", "Model", "Method", "N", "pass@N", "select@N", "Gap"]
    col_widths = [18, 12, 14, 20, 4, 8, 8, 8]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    for r in all_conditions:
        row = [
            str(r.get("run_ids", [""])[0])[: col_widths[0]].ljust(col_widths[0]),
            r["dataset"][: col_widths[1]].ljust(col_widths[1]),
            r["model"][: col_widths[2]].ljust(col_widths[2]),
            r["method"][: col_widths[3]].ljust(col_widths[3]),
            str(r["n"]).rjust(col_widths[4]),
            f"{r['pass_at_n_rate']:.1%}".rjust(col_widths[5]),
            f"{r['accuracy']:.1%}".rjust(col_widths[6]),
            f"{r['gap']:.1%}".rjust(col_widths[7]),
        ]
        print(sep.join(row))
    print("-" * len(header_line))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selection pipeline: pass@N and select@N (per condition)."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run directory name to evaluate (e.g. 00000). Default: all runs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory containing run subdirectories.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/pass_vs_select",
        help="Directory to write per-run evaluation results (default: results/pass_vs_select).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for a single aggregated JSON of all conditions.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [runs_dir / args.run_id] if args.run_id else find_run_dirs(runs_dir)

    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    all_conditions: list[dict] = []
    for rd in run_dirs:
        if not rd.exists():
            print(f"  Skipping missing run dir: {rd}", file=sys.stderr)
            continue
        print(f"Evaluating {rd.name} ...")
        conditions = aggregate_run_dir(rd)
        if not conditions:
            print(f"  Skipping {rd.name}: no evaluable conditions", file=sys.stderr)
            continue

        payload = {
            "run_dir": str(rd),
            "num_conditions": len(conditions),
            "conditions": [c.to_dict() for c in conditions],
        }
        # Backward-compatible summary if single condition.
        if len(conditions) == 1:
            only = conditions[0].to_dict()
            payload.update(
                {
                    "run_id": only["run_ids"][0] if only["run_ids"] else rd.name,
                    "model": only["model"],
                    "dataset": only["dataset"],
                    "method": only["method"],
                    "n": only["n"],
                    "total_examples": only["total_examples"],
                    "total_candidates": only["total_candidates"],
                    "pass_at_n": only["pass_at_n_correct"],
                    "pass_at_n_rate": only["pass_at_n_rate"],
                    "select_at_n": only["select_correct"],
                    "select_at_n_rate": only["accuracy"],
                    "gap": only["gap"],
                }
            )

        result_path = results_dir / f"{rd.name}.json"
        write_json(result_path, payload)
        for cond in conditions:
            print(
                f"  {cond.dataset} | {cond.model} | {cond.method} | N={cond.n}: "
                f"pass@N={cond.pass_at_n_rate:.1%} select@N={cond.accuracy:.1%} gap={cond.gap:.1%}"
            )
            all_conditions.append(cond.to_dict())
        print(f"  -> {result_path}")

    if args.output:
        write_json(Path(args.output), {"conditions": all_conditions})
        print(f"Wrote aggregated output: {args.output}")

    if len(all_conditions) > 1:
        print_summary_table(all_conditions)


if __name__ == "__main__":
    main()
