from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_truncated(run_dirs: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_files: dict[str, list] = {
        "english_cot": [],
        "yoruba_cot": [],
        "translate_pivot": [],
    }

    for run_dir in run_dirs:
        candidates_path = run_dir / "candidates.jsonl"
        if not candidates_path.exists():
            print(f"No candidates.jsonl in {run_dir}", file=sys.stderr)
            continue

        count = 0
        with open(candidates_path) as f:
            for line in f:
                row = json.loads(line)
                fr = (
                    row.get("metadata", {})
                    .get("backend_metadata", {})
                    .get("finish_reason", "")
                )
                if fr == "length":
                    method = row.get("method", "")
                    if method in strategy_files:
                        strategy_files[method].append(row)
                        count += 1

        print(f"  {run_dir.name}: {count} truncated rows")

    for method, rows in strategy_files.items():
        if not rows:
            continue
        out_path = output_dir / f"{method}_truncated.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  -> {out_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save truncated candidates (finish_reason=length) to per-strategy files."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        action="append",
        required=True,
        help="Run directory name inside runs/ (can be specified multiple times).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/_truncated",
        help="Output directory (default: runs/_truncated).",
    )
    args = parser.parse_args()

    run_dirs = [PROJECT_ROOT / "runs" / r for r in args.run_dir]
    missing = [rd for rd in run_dirs if not rd.exists()]
    if missing:
        print(f"Missing run dirs: {missing}", file=sys.stderr)
        sys.exit(1)

    extract_truncated(run_dirs, PROJECT_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
