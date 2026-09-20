#!/usr/bin/env python3
"""Pool-based TTC estimates: unbiased pass@k, random-subset maj@k, diversity.

Reads candidate traces from one or more E2 runs (nested pools) and reports, for
each nested group and each k:

- pass@k   : full-pool unbiased estimator (not a fixed prefix)
- maj@k    : majority-vote accuracy averaged over random k-subsets
- distinct : mean distinct non-empty answers in the pool (diversity diagnostic)
- degen%   : fraction of examples whose pool has <=1 distinct answer

Writes JSON and prints a table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.estimators import nested_pool_estimates
from ttcs_yoruba.io_utils import read_jsonl, write_json
from ttcs_yoruba.metrics import dedupe_candidate_rows, find_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool-based TTC estimates (E2/E3).")
    parser.add_argument("--runs-dir", default="runs", help="Run directory root.")
    parser.add_argument("--run-id", default=None, help="Single run directory name.")
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: results/pool_estimates/<run-id>.json).",
    )
    parser.add_argument("--num-subsets", type=int, default=20, help="Random k-subsets per example.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for subset sampling.")
    parser.add_argument(
        "--ks",
        default=None,
        help="Comma-separated k values (default: powers of two up to pool size).",
    )
    return parser.parse_args()


def load_candidates(run_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for run_dir in run_dirs:
        path = run_dir / "candidates.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    return dedupe_candidate_rows(rows)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No nested pools found.")
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
        "gen/samp",
        "gen/ex",
        "gen/corr",
        "distinct",
        "degen%",
    ]
    widths = [12, 12, 20, 5, 4, 5, 8, 8, 7, 8, 7, 8, 9, 7]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for r in rows:
        tpc = r.get("completion_tokens_per_correct")
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
                    f"{r['mean_completion_tokens_per_sample']:.1f}".rjust(widths[9]),
                    f"{r['completion_tokens_per_example_at_k']:.1f}".rjust(widths[10]),
                    ("—" if tpc is None else f"{tpc:.1f}").rjust(widths[11]),
                    f"{r['distinct_answers_mean']:.2f}".rjust(widths[12]),
                    f"{r['degenerate_pool_rate']:.1%}".rjust(widths[13]),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if args.run_id:
        run_dirs = [runs_dir / args.run_id]
    else:
        run_dirs = find_run_dirs(runs_dir)
    run_dirs = [path for path in run_dirs if path.exists()]
    if not run_dirs:
        print(f"No run directories found under {runs_dir}", file=sys.stderr)
        sys.exit(1)

    candidates = load_candidates(run_dirs)
    ks = None
    if args.ks:
        ks = [int(x) for x in args.ks.split(",") if x.strip()]

    rows = nested_pool_estimates(
        candidates, ks=ks, num_subsets=args.num_subsets, seed=args.seed
    )
    print_table(rows)

    label = args.run_id or "all_runs"
    output_path = Path(args.output) if args.output else Path("results/pool_estimates") / f"{label}.json"
    write_json(
        output_path,
        {
            "runs": [str(path) for path in run_dirs],
            "num_subsets": args.num_subsets,
            "seed": args.seed,
            "estimates": rows,
        },
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
