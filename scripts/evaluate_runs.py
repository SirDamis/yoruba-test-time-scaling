from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.extraction import is_exact_match
from ttcs_yoruba.io_utils import read_json, read_jsonl, write_json


def evaluate_run(run_dir: Path) -> dict:
    candidates_path = run_dir / "candidates.jsonl"
    selections_path = run_dir / "selections.jsonl"
    manifest_path = run_dir / "manifest.json"

    if not candidates_path.exists():
        print(f"  Skipping {run_dir.name}: no candidates.jsonl", file=sys.stderr)
        return {}

    manifest = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)

    manifest_datasets = manifest.get("datasets", [])
    manifest_methods = manifest.get("methods", [])
    manifest_models = manifest.get("models", [])

    candidates = read_jsonl(candidates_path)

    by_example: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_example[c["example_id"]].append(c)

    n = 0
    example_results = {}
    for example_id, cands in by_example.items():
        gold_answer = None
        answer_type = None
        per_sample = []
        for c in cands:
            meta = c.get("metadata", {})
            answer_type = meta.get("answer_type", answer_type)
            gold_answer = meta.get("gold_answer", gold_answer)
            pred = c.get("extracted_answer", "")
            per_sample.append(is_exact_match(pred, gold_answer, answer_type))

        n = max(n, len(cands))
        example_results[example_id] = {
            "num_samples": len(cands),
            "pass_at_n": any(per_sample),
            "per_sample_correct": per_sample,
        }

    selections = read_jsonl(selections_path) if selections_path.exists() else []
    selection_results = {}
    for s in selections:
        eid = s["example_id"]
        selection_results[eid] = {
            "selected_answer": s.get("selected_answer", ""),
            "gold_answer": s.get("gold_answer", ""),
            "is_correct": s.get("is_correct", False),
        }

    total = len(by_example)
    pass_at_n_count = sum(1 for r in example_results.values() if r["pass_at_n"])
    select_at_n_count = sum(1 for r in selection_results.values() if r["is_correct"])

    run_info = {
        "run_id": manifest.get("run_id", run_dir.name),
        "model": manifest_models[0]["name"] if manifest_models else "",
        "dataset": manifest_datasets[0] if manifest_datasets else "",
        "method": manifest_methods[0]["name"] if manifest_methods else "",
        "seed": manifest.get("seed", ""),
        "n": manifest_methods[0]["n"] if manifest_methods else n,
        "total_examples": total,
        "total_candidates": len(candidates),
        "pass_at_n": pass_at_n_count,
        "pass_at_n_rate": pass_at_n_count / total if total else 0.0,
        "select_at_n": select_at_n_count,
        "select_at_n_rate": select_at_n_count / total if total else 0.0,
        "gap": (pass_at_n_count - select_at_n_count) / total if total else 0.0,
    }

    return run_info


def find_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "candidates.jsonl").exists()
    )


def print_summary_table(all_results: dict[str, dict]) -> None:
    headers = ["Run", "Dataset", "Method", "N", "pass@N", "select@N", "Gap"]
    col_widths = [20, 12, 22, 4, 8, 8, 8]
    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    for run_name, r in all_results.items():
        row = [
            run_name.ljust(col_widths[0]),
            r["dataset"].ljust(col_widths[1]),
            r["method"].ljust(col_widths[2]),
            str(r["n"]).rjust(col_widths[3]),
            f"{r['pass_at_n_rate']:.1%}".rjust(col_widths[4]),
            f"{r['select_at_n_rate']:.1%}".rjust(col_widths[5]),
            f"{r['gap']:.1%}".rjust(col_widths[6]),
        ]
        print(sep.join(row))
    print("-" * len(header_line))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selection pipeline: pass@N and select@N."
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Specific run directory name to evaluate (e.g. 00000). Default: all runs.",
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Base directory containing run subdirectories.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/pass_vs_select",
        help="Directory to write per-run evaluation results (default: results/pass_vs_select).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [runs_dir / args.run_id] if args.run_id else find_run_dirs(runs_dir)

    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    all_results: dict[str, dict] = {}
    for rd in run_dirs:
        print(f"Evaluating {rd.name} ...")
        result = evaluate_run(rd)
        if result:
            all_results[rd.name] = result
            result_path = results_dir / f"{rd.name}.json"
            write_json(result_path, result)
            print(
                f"  pass@N:  {result['pass_at_n']:>4}/{result['total_examples']}  "
                f"({result['pass_at_n_rate']:.1%})"
            )
            print(
                f"  select@N: {result['select_at_n']:>4}/{result['total_examples']}  "
                f"({result['select_at_n_rate']:.1%})"
            )
            print(f"  gap:     {result['gap']:.1%}")
            print(f"  -> {result_path}")

    if len(all_results) > 1:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
