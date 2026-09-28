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
from thinkprm_yoruba.select import attach_thinkprm_scores, question_of, steps_of


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


def _parse_n_values(value: str | None) -> set[int] | None:
    if not value or str(value).strip().lower() in {"all", "*"}:
        return None
    return {int(part) for part in str(value).split(",") if part.strip()}


def _pool_id(row: dict, key: tuple) -> str:
    """Identity shared by the N conditions of a nested pool (else the method)."""
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("nested_group_id"):
        return str(metadata["nested_group_id"])
    return str(key[2])


def _print_verification_flow(
    record: dict,
    rows: list[dict],
    *,
    max_chars: int,
    show_all: bool,
) -> None:
    """Print the verifier's step-by-step reasoning and judgments for a group."""
    selected_index = int(record.get("selected_sample_index", -1))
    chosen = rows if show_all else [
        row for row in rows if int(row.get("sample_index", -2)) == selected_index
    ]
    for row in chosen:
        sample_index = int(row.get("sample_index", -1))
        is_selected = sample_index == selected_index
        raw_prm = row.get("prm_score")
        prm_str = f"{float(raw_prm):.3f}" if isinstance(raw_prm, (int, float)) else "n/a"
        prefix = row.get("thinkprm_prefix_score")
        prefix_str = f"{float(prefix):.3f}" if isinstance(prefix, (int, float)) else "n/a"
        header = (
            f"\n=== {record.get('example_id')} | N={record.get('n')} | sample={sample_index}"
            + (" (selected)" if is_selected else "")
            + f" | prm={prm_str} | prefix_P(Yes)={prefix_str}"
            + f" | answer={row.get('extracted_answer')!r} gold={record.get('gold_answer')!r}"
        )
        if is_selected:
            header += " | correct=yes" if record.get("is_correct") else " | correct=no"
        print(header + " ===", flush=True)

        steps = steps_of(row)
        print(f"Problem: {question_of(row)}", flush=True)
        print(f"Solution ({len(steps)} steps):", flush=True)
        for index, step in enumerate(steps, 1):
            print(f"  {index}. {step}", flush=True)

        labels = list(row.get("thinkprm_step_labels") or [])
        scores = list(row.get("thinkprm_step_scores") or [])
        if labels:
            judgments = ", ".join(
                f"{index}:{int(label)}({float(score):.2f})"
                for index, (label, score) in enumerate(zip(labels, scores), 1)
            )
            print(f"Step judgments (1=correct, 0=incorrect): {judgments}", flush=True)

        for chain_index, output in enumerate(row.get("thinkprm_outputs") or []):
            text = str(output)
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + f" …[{len(text) - max_chars} chars truncated]"
            print(f"Verification chain [{chain_index}]:\n{text}", flush=True)


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
        "--n-values",
        default=None,
        help=(
            "Comma-separated N conditions to select/report, e.g. 2,4,8,16,32,64 "
            "('all' disables the filter). Overrides the config's n_values "
            "(default 2,4,8,16,32,64; N=1 is usually the greedy baseline)."
        ),
    )
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
        help="Print a live progress line every N processed groups (default 1; 0 disables).",
    )
    parser.add_argument(
        "--show-verification",
        action="store_true",
        help=(
            "Print the verifier's step-by-step reasoning and per-step judgments "
            "(off by default; off is recommended for full runs)."
        ),
    )
    parser.add_argument(
        "--show-all-candidates",
        action="store_true",
        help="With --show-verification, print every candidate (default: only the selected one).",
    )
    parser.add_argument(
        "--show-verification-chars",
        type=int,
        default=2000,
        help="Truncate each verification chain to N chars with --show-verification (0 = no limit).",
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

    scorer = ThinkPrmScorer(config)
    scorer.load()
    print(
        f"Loaded {config.model}"
        + (f" + adapter {config.adapter_path}" if config.adapter_path else "")
        + f" (aggregation={config.aggregation}, n_verifications={config.n_verifications})",
        flush=True,
    )

    if args.n_values is not None:
        n_filter = _parse_n_values(args.n_values)
    elif config.n_values:
        n_filter = set(config.n_values)
    else:
        n_filter = None

    raw_groups: dict[tuple[str, str, str, int, str], list[dict]] = {}
    for _, path in files:
        rows = dedupe_candidate_rows(read_jsonl(path))
        for key, candidates in group_candidates(rows).items():
            if n_filter is None or key[3] in n_filter:
                raw_groups[key] = candidates

    # Plan the unique traces to score. Nested pools share candidates across N,
    # so each distinct (example, pool, sample_index) is scored exactly once.
    planned: set[tuple] = set()
    for key, candidates in raw_groups.items():
        pool = _pool_id(candidates[0], key)
        for candidate in candidates:
            planned.add((key[0], key[1], key[4], pool, int(candidate.get("sample_index", 0))))
    total_unique = len(planned)
    total_groups = len(raw_groups)
    examples = sorted({(key[0], key[1], key[4]) for key in raw_groups})
    total_examples = len(examples)

    progress_every = max(0, int(args.progress_every))
    if not args.no_progress and progress_every:
        print(
            f"Scoring {total_unique} unique candidates | {total_groups} conditions | "
            f"{total_examples} examples"
            + (f" | N ∈ {sorted(n_filter)}" if n_filter is not None else ""),
            flush=True,
        )

    output_dir = resolve_path(config.output_dir)
    cache: dict[tuple, dict] = {}
    groups: dict[tuple[str, str, str, int, str], list[dict]] = {}
    selections: list[dict] = []
    condition_stats: dict[tuple[str, str, str, int], dict[str, int]] = {}
    scored_count = 0
    processed_groups = 0
    start_time = time.time()

    def flush(condition: tuple[str, str, str, int]) -> None:
        stats = condition_stats.get(condition)
        if not stats or stats["examples"] == 0:
            return
        examples_count = stats["examples"]
        pass_rate = stats["pass"] / examples_count
        select_rate = stats["select"] / examples_count
        _, _, method, n = condition
        print(
            f"[{method}] N={n}  examples={examples_count}  "
            f"pass@N={pass_rate:.1%}  thinkprm@N={select_rate:.1%}  "
            f"gap={select_rate - pass_rate:+.1%}",
            flush=True,
        )

    for example_index, example in enumerate(examples, start=1):
        example_keys = sorted(
            (key for key in raw_groups if (key[0], key[1], key[4]) == example),
            key=lambda key: (key[3], str(key[2])),
        )
        for key in example_keys:
            if config.limit_groups is not None and processed_groups >= config.limit_groups:
                break
            candidates = raw_groups.pop(key)
            pool = _pool_id(candidates[0], key)

            def cache_key(row: dict) -> tuple:
                return (key[0], key[1], key[4], pool, int(row.get("sample_index", 0)))

            to_score = [candidate for candidate in candidates if cache_key(candidate) not in cache]
            if to_score:
                for row in attach_thinkprm_scores(
                    to_score,
                    scorer,
                    max_steps=config.max_steps,
                    batch_size=config.batch_size,
                ):
                    cache[cache_key(row)] = row
                scored_count += len(to_score)

            scored = []
            for candidate in candidates:
                row = dict(cache[cache_key(candidate)])
                # Cached rows keep the method/n of the first condition that scored
                # them; re-label them for the N condition being reported.
                row["method"] = key[2]
                row["n"] = key[3]
                scored.append(row)
            groups[key] = scored

            passed = pass_at_n_for_group(scored)
            result = select_candidate(scored, "thinkprm")
            record = build_selection_record(
                candidates=scored,
                strategy="thinkprm",
                selected_sample_index=result.selected_sample_index,
                selected_answer=result.selected_answer,
                vote_counts=result.vote_counts,
                metadata=result.metadata,
                run_id=str(candidates[0].get("run_id", "")),
            )
            selections.append(record)

            condition = (
                str(record["dataset"]),
                str(record["model"]),
                str(record["method"]),
                int(record["n"]),
            )
            stats = condition_stats.setdefault(
                condition, {"examples": 0, "pass": 0, "select": 0}
            )
            stats["examples"] += 1
            stats["pass"] += int(bool(passed))
            stats["select"] += int(bool(record["is_correct"]))

            if args.show_verification:
                _print_verification_flow(
                    record,
                    scored,
                    max_chars=args.show_verification_chars,
                    show_all=args.show_all_candidates,
                )

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

            processed_groups += 1
            if (
                not args.no_progress
                and progress_every
                and (processed_groups % progress_every == 0 or processed_groups == total_groups)
            ):
                elapsed = time.time() - start_time
                rate = scored_count / elapsed if elapsed > 0 else 0.0
                eta = (total_unique - scored_count) / rate if rate > 0 else 0.0
                accuracy = stats["select"] / stats["examples"] if stats["examples"] else 0.0
                pick_score = (record.get("selection_metadata") or {}).get("prm_score")
                pick_score_str = (
                    f"{float(pick_score):.2f}"
                    if isinstance(pick_score, (int, float))
                    else "n/a"
                )
                outcome = "OK" if record["is_correct"] else "MISS"
                print(
                    f"  [ex {example_index}/{total_examples}] {example[2]} n={condition[3]} "
                    f"(+{len(to_score)} new, {scored_count}/{total_unique} scored) "
                    f"| sel={outcome} pick={record['selected_answer']!r} "
                    f"gold={record['gold_answer']!r} prm={pick_score_str} "
                    f"| N acc={accuracy:.1%} | {rate:.2f} cand/s | elapsed {elapsed / 60:.1f}m "
                    f"| eta {eta / 60:.1f}m",
                    flush=True,
                )
        if config.limit_groups is not None and processed_groups >= config.limit_groups:
            break

    if not args.no_progress:
        for condition in sorted(condition_stats, key=lambda key: (key[3], str(key[2]))):
            flush(condition)

    write_jsonl(output_dir / "scores.jsonl", list(cache.values()))
    write_jsonl(output_dir / "selections_thinkprm.jsonl", selections)

    report_rows = summarize_reselection(groups, {"thinkprm": selections})
    report = {
        "model": config.model,
        "adapter_path": config.adapter_path,
        "aggregation": config.aggregation,
        "n_verifications": config.n_verifications,
        "n_values": sorted(n_filter) if n_filter is not None else None,
        "num_groups": len(groups),
        "num_unique_candidates": scored_count,
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
