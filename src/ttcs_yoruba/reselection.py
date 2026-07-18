"""Offline re-selection over saved candidate traces (E3)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .extraction import is_exact_match
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .metrics import candidate_dataset, dedupe_candidate_rows
from .selection import select_candidate


def group_candidates(
    candidates: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int, str], list[dict[str, Any]]]:
    """Group candidates by (dataset, model, method, n, example_id)."""
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = (
            candidate_dataset(row),
            str(row.get("model", "")),
            str(row.get("method", "")),
            int(row.get("n") or 1),
            str(row.get("example_id", "")),
        )
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("sample_index", 0)))
    return groups


def build_selection_record(
    *,
    candidates: list[dict[str, Any]],
    strategy: str,
    selected_sample_index: int,
    selected_answer: str,
    vote_counts: dict[str, int],
    metadata: dict[str, Any] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot build selection record without candidates")

    head = candidates[0]
    meta = head.get("metadata") or {}
    gold = str(meta.get("gold_answer", head.get("gold_answer", "")))
    answer_type = str(meta.get("answer_type", head.get("answer_type", "text")))
    return {
        "run_id": run_id or str(head.get("run_id", "")),
        "dataset": candidate_dataset(head),
        "example_id": str(head.get("example_id", "")),
        "task": str(head.get("task", "")),
        "source_dataset": str(head.get("source_dataset", "")),
        "model": str(head.get("model", "")),
        "model_size_label": str(head.get("model_size_label", "")),
        "method": str(head.get("method", "")),
        "prompt_style": str(head.get("prompt_style", "")),
        "selection": strategy,
        "n": int(head.get("n") or 1),
        "reasoning_language": str(head.get("reasoning_language", "")),
        "selected_sample_index": selected_sample_index,
        "selected_answer": selected_answer,
        "gold_answer": gold,
        "answer_type": answer_type,
        "is_correct": is_exact_match(selected_answer, gold, answer_type),
        "candidate_count": len(candidates),
        "vote_counts": vote_counts,
        "selection_metadata": metadata or {},
    }


def pass_at_n_for_group(candidates: list[dict[str, Any]]) -> bool:
    if not candidates:
        return False
    meta = candidates[0].get("metadata") or {}
    gold = str(meta.get("gold_answer", ""))
    answer_type = str(meta.get("answer_type", "text"))
    for row in candidates:
        pred = str(row.get("extracted_answer", ""))
        if is_exact_match(pred, gold, answer_type):
            return True
    return False


def reselect_groups(
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]],
    *,
    strategies: list[str],
    run_id: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Apply each strategy to every candidate group.

    Returns mapping strategy -> list of selection records.
    Local strategies only: ``first``, ``majority_vote``.
    """
    outputs: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in strategies}

    for key in sorted(groups):
        candidates = groups[key]
        head = candidates[0] if candidates else {}
        meta = head.get("metadata") or {}
        answer_type = str(meta.get("answer_type", head.get("answer_type", "text")))
        for strategy in strategies:
            result = select_candidate(candidates, strategy, answer_type=answer_type)
            record = build_selection_record(
                candidates=candidates,
                strategy=strategy,
                selected_sample_index=result.selected_sample_index,
                selected_answer=result.selected_answer,
                vote_counts=result.vote_counts,
                metadata=result.metadata,
                run_id=run_id,
            )
            outputs[strategy].append(record)

    return outputs


def summarize_reselection(
    groups: dict[tuple[str, str, str, int, str], list[dict[str, Any]]],
    selections_by_strategy: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build E3 report rows: pass@N vs select@N per strategy, by condition."""
    # Index selections by condition + example.
    by_strategy_condition: dict[str, dict[tuple[str, str, str, int], dict[str, dict[str, Any]]]] = {
        strategy: defaultdict(dict) for strategy in selections_by_strategy
    }
    for strategy, rows in selections_by_strategy.items():
        for row in rows:
            cond = (
                str(row.get("dataset", "")),
                str(row.get("model", "")),
                str(row.get("method", "")),
                int(row.get("n") or 1),
            )
            by_strategy_condition[strategy][cond][str(row["example_id"])] = row

    # pass@N from candidate groups.
    pass_by_condition: dict[tuple[str, str, str, int], dict[str, bool]] = defaultdict(dict)
    meta_by_condition: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for (dataset, model, method, n, example_id), cands in groups.items():
        cond = (dataset, model, method, n)
        pass_by_condition[cond][example_id] = pass_at_n_for_group(cands)
        if cond not in meta_by_condition and cands:
            head = cands[0]
            meta_by_condition[cond] = {
                "prompt_style": str(head.get("prompt_style", "")),
                "reasoning_language": str(head.get("reasoning_language", "")),
                "model_size_label": str(head.get("model_size_label", "")),
            }

    all_conditions = sorted(pass_by_condition)
    report: list[dict[str, Any]] = []
    for cond in all_conditions:
        dataset, model, method, n = cond
        pass_map = pass_by_condition[cond]
        total = len(pass_map)
        pass_correct = sum(1 for ok in pass_map.values() if ok)
        pass_rate = pass_correct / total if total else 0.0
        meta = meta_by_condition.get(cond, {})

        strategy_stats: dict[str, Any] = {}
        for strategy, cond_map in by_strategy_condition.items():
            rows = cond_map.get(cond, {})
            # Align to pass@N example set when possible.
            if rows:
                select_correct = sum(1 for row in rows.values() if row.get("is_correct"))
                select_total = len(rows)
            else:
                select_correct = 0
                select_total = 0
            select_rate = select_correct / select_total if select_total else 0.0
            strategy_stats[strategy] = {
                "select_correct": select_correct,
                "total_examples": select_total,
                "select_at_n_rate": select_rate,
                "accuracy": select_rate,
                "gap": pass_rate - select_rate,
            }

        report.append(
            {
                "dataset": dataset,
                "model": model,
                "model_size_label": meta.get("model_size_label", ""),
                "method": method,
                "n": n,
                "prompt_style": meta.get("prompt_style", ""),
                "reasoning_language": meta.get("reasoning_language", ""),
                "total_examples": total,
                "pass_at_n_correct": pass_correct,
                "pass_at_n_rate": pass_rate,
                "selections": strategy_stats,
            }
        )
    return report


def reselect_run_dir(
    run_dir: Path,
    *,
    strategies: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Re-select candidates for one run directory and write artifacts + report."""
    candidates_path = run_dir / "candidates.jsonl"
    if not candidates_path.exists():
        raise FileNotFoundError(f"No candidates.jsonl in {run_dir}")

    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    run_id = str(manifest.get("run_id", run_dir.name))

    candidates = dedupe_candidate_rows(read_jsonl(candidates_path))
    groups = group_candidates(candidates)

    selections_by_strategy = reselect_groups(
        groups,
        strategies=strategies,
        run_id=run_id,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for strategy, rows in selections_by_strategy.items():
        path = output_dir / f"selections_{strategy}.jsonl"
        write_jsonl(path, rows)
        written[strategy] = str(path)

    report_rows = summarize_reselection(groups, selections_by_strategy)
    report = {
        "source_run_dir": str(run_dir),
        "run_id": run_id,
        "strategies": strategies,
        "num_groups": len(groups),
        "num_conditions": len(report_rows),
        "conditions": report_rows,
        "selection_paths": written,
    }
    report_path = output_dir / "e3_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def print_e3_report_table(report_rows: list[dict[str, Any]], strategies: Iterable[str]) -> None:
    strategies = list(strategies)
    headers = ["Dataset", "Model", "Method", "N", "pass@N"] + [f"sel:{s}" for s in strategies] + ["gap:maj"]
    # Prefer majority gap if present, else first strategy gap.
    print("\n" + "  ".join(headers))
    print("-" * 100)
    for row in report_rows:
        parts = [
            str(row.get("dataset", ""))[:12],
            str(row.get("model", ""))[:14],
            str(row.get("method", ""))[:18],
            str(row.get("n", "")),
            f"{row.get('pass_at_n_rate', 0.0):.1%}",
        ]
        selections = row.get("selections") or {}
        for strategy in strategies:
            stats = selections.get(strategy) or {}
            parts.append(f"{stats.get('select_at_n_rate', 0.0):.1%}")
        maj = selections.get("majority_vote") or next(iter(selections.values()), {})
        parts.append(f"{maj.get('gap', 0.0):.1%}")
        print("  ".join(parts))
