from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def compute_accuracy(run_dir: Path) -> None:
    selections_path = run_dir / "selections.jsonl"
    candidates_path = run_dir / "candidates.jsonl"
    if not selections_path.exists():
        print(f"No selections.jsonl in {run_dir}", file=sys.stderr)
        return

    per_method: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(selections_path) as f:
        for line in f:
            row = json.loads(line)
            method = row.get("method", "unknown")
            per_method[method]["total"] += 1
            if row.get("is_correct", False):
                per_method[method]["correct"] += 1

    truncated: dict[str, int] = defaultdict(int)
    total_candidates: dict[str, int] = defaultdict(int)
    if candidates_path.exists():
        with open(candidates_path) as f:
            for line in f:
                row = json.loads(line)
                method = row.get("method", "unknown")
                total_candidates[method] += 1
                fr = (
                    row.get("metadata", {})
                    .get("backend_metadata", {})
                    .get("finish_reason", "")
                )
                if fr == "length":
                    truncated[method] += 1

    with open(selections_path) as f:
        first = json.loads(f.readline())
        model = first.get("model", "")
        dataset = first.get("dataset", "")

    print(f"\n{run_dir.name}")
    print(f"  Model: {model} | Dataset: {dataset}")
    print(f"  {'Method':<22s} {'Accuracy':>10s}  {'Correct':>8s}  {'Truncated':>10s}")
    print(f"  {'-'*22} {'-'*10}  {'-'*8}  {'-'*10}")

    for method in sorted(per_method):
        stats = per_method[method]
        acc = stats["correct"] / stats["total"] * 100
        t = truncated.get(method, 0)
        tc = total_candidates.get(method, 0)
        trunc_str = f"{t}/{tc}" if tc else "N/A"
        print(f"  {method:<22s} {acc:>9.1f}%  {stats['correct']:>4d}/{stats['total']:<4d}  {trunc_str:>10s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracy per reasoning strategy for E1 experiment runs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the experiment run directory (e.g. runs/yoruba-test-time-scaling_runs_e1_afrimgsm_gemma3-4b_vllm).",
    )
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parents[1] / "runs" / args.run_dir
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    compute_accuracy(run_dir)


if __name__ == "__main__":
    main()
