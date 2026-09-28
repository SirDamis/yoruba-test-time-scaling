from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

THINKPRM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = THINKPRM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(THINKPRM_ROOT.parent / "prm" / "src"))
sys.path.insert(0, str(THINKPRM_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from ttcs_yoruba.io_utils import read_jsonl, write_json, write_jsonl
from ttcs_yoruba.metrics import dedupe_candidate_rows
from ttcs_yoruba.reselection import (
    build_selection_record,
    group_candidates,
    pass_at_n_for_group,
    print_e3_report_table,
    summarize_reselection,
)
from ttcs_yoruba.selection import select_candidate

from thinkprm_yoruba.config import ScoreConfig, load_json, resolve_path
from thinkprm_yoruba.model import ThinkPrmScorer
from thinkprm_yoruba.select import attach_thinkprm_scores


def iter_candidate_files(runs_dir: str, run_ids: list[str] | None) -> list[tuple[str, Path]]:
    root = resolve_path(runs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Runs directory not found: {root}")
    if run_ids:
        directories = [root / run_id for run_id in run_ids]
    else:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    files: list[tuple[str, Path]] = []
    for directory in directories:
        candidates = directory / "candidates.jsonl"
        if candidates.exists():
            files.append((directory.name, candidates))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score E2 candidate pools with a ThinkPRM verifier and select Best-of-N."
    )
    parser.add_argument("--config", default="thinkprm/configs/score_candidates.json")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--aggregation", default=None, help="prefix (default), last, max, mean, or min."
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-verifications", type=int, default=None, help="Parallel verification chains.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument(
        "--per-example",
        action="store_true",
        help="Print one line per verifier selection (N, example, pick, gold, correct).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the running per-N selection summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a live progress line every N groups (default 1; 0 disables).",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Restrict scoring to these run ids (repeatable).",
    )
    args = parser.parse_args()

    config = ScoreConfig.from_dict(load_json(args.config))
    overrides = {}
    for key, value in {
        "runs_dir": args.runs_dir,
        "output_dir": args.output_dir,
        "adapter_path": args.adapter_path,
        "aggregation": args.aggregation,
        "max_steps": args.max_steps,
        "n_verifications": args.n_verifications,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "limit_groups": args.limit_groups,
    }.items():
        if value is not None:
            overrides[key] = value
    if overrides:
        config = replace(config, **overrides)
    if args.run_id:
        config = replace(config, run_ids=args.run_id)

    files = iter_candidate_files(config.runs_dir, config.run_ids)
    if not files:
        raise SystemExit(f"No candidates.jsonl found under {config.runs_dir}")

    progress_every = max(0, int(args.progress_every))
    total_groups = 0
    if not args.no_progress and progress_every:
        total_groups = sum(
            len(group_candidates(dedupe_candidate_rows(read_jsonl(path))))
            for _, path in files
        )
        if config.limit_groups is not None:
            total_groups = min(total_groups, config.limit_groups)
        print(
            f"Scoring {total_groups} groups across {len(files)} run(s); loading model...",
            flush=True,
        )

    scorer = ThinkPrmScorer(config)
    scorer.load()
    print(
        f"Loaded {config.model}"
        + (f" + adapter {config.adapter_path}" if config.adapter_path else "")
        + f" (aggregation={config.aggregation}, n_verifications={config.n_verifications})",
        flush=True,
    )

    output_dir = resolve_path(config.output_dir)
    groups: dict[tuple[str, str, str, int, str], list[dict]] = {}
    selections: list[dict] = []
    all_scored: list[dict] = []

    condition_stats: dict[tuple[str, str, str, int], dict[str, int]] = {}
    current_condition: tuple[str, str, str, int] | None = None

    def flush(condition: tuple[str, str, str, int]) -> None:
        stats = condition_stats.get(condition)
        if not stats or stats["examples"] == 0:
            return
        examples = stats["examples"]
        pass_rate = stats["pass"] / examples
        select_rate = stats["select"] / examples
        _, _, method, n = condition
        print(
            f"[{method}] N={n}  examples={examples}  "
            f"pass@N={pass_rate:.1%}  thinkprm@N={select_rate:.1%}  gap={select_rate - pass_rate:+.1%}",
            flush=True,
        )

    if not args.no_progress:
        print("ThinkPRM verifier selection (per N, vs ground truth):", flush=True)

    processed = 0
    start_time = time.time()
    for run_id, path in files:
        rows = dedupe_candidate_rows(read_jsonl(path))
        run_groups = group_candidates(rows)
        for key in sorted(run_groups):
            if config.limit_groups is not None and processed >= config.limit_groups:
                break
            candidates = run_groups[key]
            scored = attach_thinkprm_scores(
                candidates,
                scorer,
                max_steps=config.max_steps,
                batch_size=config.batch_size,
            )
            groups[key] = scored
            all_scored.extend(scored)
            passed = pass_at_n_for_group(scored)
            result = select_candidate(scored, "thinkprm")
            record = build_selection_record(
                candidates=scored,
                strategy="thinkprm",
                selected_sample_index=result.selected_sample_index,
                selected_answer=result.selected_answer,
                vote_counts=result.vote_counts,
                metadata=result.metadata,
                run_id=run_id,
            )
            selections.append(record)

            condition = (
                str(record["dataset"]),
                str(record["model"]),
                str(record["method"]),
                int(record["n"]),
            )
            if condition != current_condition:
                if not args.no_progress and current_condition is not None:
                    flush(current_condition)
                current_condition = condition
                if not args.no_progress and progress_every:
                    print(f">> {condition[2]} (n={condition[3]})", flush=True)
            stats = condition_stats.setdefault(
                condition, {"examples": 0, "pass": 0, "select": 0}
            )
            stats["examples"] += 1
            stats["pass"] += int(bool(passed))
            stats["select"] += int(bool(record["is_correct"]))

            if args.per_example:
                prm_score = (record.get("selection_metadata") or {}).get("prm_score")
                score_str = (
                    f"{float(prm_score):.3f}"
                    if isinstance(prm_score, (int, float))
                    else "n/a"
                )
                print(
                    f"  N={condition[3]}  {record['example_id']}  thinkprm={score_str}  "
                    f"selected={record['selected_answer']!r}  gold={record['gold_answer']!r}  "
                    f"correct={'yes' if record['is_correct'] else 'no'}",
                    flush=True,
                )
            processed += 1
            if (
                not args.no_progress
                and progress_every
                and (processed % progress_every == 0 or processed == total_groups)
            ):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (total_groups - processed) / rate if rate > 0 else 0.0
                accuracy = stats["select"] / stats["examples"] if stats["examples"] else 0.0
                print(
                    f"  [{processed}/{total_groups}] {condition[2]} n={condition[3]} "
                    f"acc={accuracy:.1%} | {rate:.2f} grp/s | elapsed {elapsed / 60:.1f}m "
                    f"| eta {eta / 60:.1f}m",
                    flush=True,
                )
        if config.limit_groups is not None and processed >= config.limit_groups:
            break

    if not args.no_progress and current_condition is not None:
        flush(current_condition)

    write_jsonl(output_dir / "scores.jsonl", all_scored)
    write_jsonl(output_dir / "selections_thinkprm.jsonl", selections)

    report_rows = summarize_reselection(groups, {"thinkprm": selections})
    report = {
        "model": config.model,
        "adapter_path": config.adapter_path,
        "aggregation": config.aggregation,
        "n_verifications": config.n_verifications,
        "num_groups": len(groups),
        "num_conditions": len(report_rows),
        "conditions": report_rows,
        "paths": {
            "scores": str(output_dir / "scores.jsonl"),
            "selections": str(output_dir / "selections_thinkprm.jsonl"),
        },
    }
    write_json(output_dir / "summary.json", report)
    print_e3_report_table(report_rows, ["thinkprm"])
    print(json.dumps(report["paths"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
