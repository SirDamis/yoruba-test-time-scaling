"""Build ThinkPRM training data by synthesising verification chains.

Following the ThinkPRM recipe, a strong reasoning teacher critiques each step of
a PRM800K solution. A sampled chain is kept only when its per-step
``\\boxed{correct}``/``\\boxed{incorrect}`` judgements agree with the gold
process labels (process-based filtering) and stay within the length budget.
English is not special here: point ``prm_data`` at the Yoruba build output to
collect Yoruba verification chains instead.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from prm_yoruba.steps import QWEN_STEP_TAG, split_tagged_process
from ttcs_yoruba.io_utils import read_jsonl, write_json, write_jsonl

from .config import CollectConfig, resolve_path
from .parse import extract_boxed_decisions, truncate_after_last_decision
from .prompts import COLLECTION_INSTRUCTION, build_verification_user_message
from .steps import render_numbered_solution, split_into_steps


def _normalise_label(value: Any) -> int | None:
    if value in ("+", 1, True):
        return 1
    if value in ("-", 0, False):
        return 0
    return None


def _row_steps(row: dict[str, Any]) -> list[str]:
    steps = row.get("steps")
    if isinstance(steps, list) and steps:
        return [str(step) for step in steps]
    if row.get("process"):
        return split_tagged_process(str(row["process"]), QWEN_STEP_TAG)
    if row.get("response"):
        return split_into_steps(str(row["response"]))
    return []


def load_prm_rows(
    paths: list[str],
    *,
    max_examples: int | None = None,
    seed: int = 1234,
) -> list[dict[str, Any]]:
    """Load step-labelled rows (``prm/data/processed/*.jsonl``)."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        for raw in read_jsonl(resolve_path(path)):
            steps = _row_steps(raw)
            labels = [_normalise_label(value) for value in raw.get("label", [])]
            if not steps or not labels or any(label is None for label in labels):
                continue
            n = min(len(steps), len(labels))
            rows.append(
                {
                    "id": str(raw.get("id", f"row:{len(rows)}")),
                    "question": str(raw.get("question", "")),
                    "steps": [str(step) for step in steps[:n]],
                    "labels": [int(label) for label in labels[:n]],
                    "language": raw.get("language"),
                    "task": str(raw.get("task", "math")),
                    "source": str(raw.get("source", "prm800k")),
                }
            )
    if max_examples is not None and len(rows) > max_examples:
        rows = random.Random(seed).sample(rows, int(max_examples))
    return rows


def labels_until_first_error(labels: list[int]) -> list[int]:
    """PRM800K prefix labels: everything up to and including the first error."""
    for index, label in enumerate(labels):
        if label == 0:
            return list(labels[: index + 1])
    return list(labels)


def _approx_tokens(text: str) -> int:
    """Cheap whitespace token estimate (keeps data building CPU/transformers-free)."""
    return len(str(text or "").split())


def build_training_example(
    row: dict[str, Any],
    text: str,
    *,
    max_cot_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Validate one teacher verification chain against the gold step labels.

    Returns a training example, or ``None`` when the chain is malformed or a step
    judgement disagrees with the gold process labels.
    """
    decisions = extract_boxed_decisions(text)
    if not decisions:
        return None
    decisions_binary = [1 if decision == "correct" else 0 for decision in decisions]

    gold = [int(label) for label in row["labels"]]
    labels_until_error = labels_until_first_error(gold)
    if len(decisions_binary) not in (len(labels_until_error), len(gold)):
        return None
    if decisions_binary[: len(labels_until_error)] != labels_until_error:
        return None

    cot = truncate_after_last_decision(text)
    if max_cot_tokens is not None and _approx_tokens(text) > max_cot_tokens:
        return None

    labels = list(labels_until_error) + [0] * (len(gold) - len(labels_until_error))
    return {
        "id": f"{row['id']}:cot{len(decisions_binary)}",
        "problem": row["question"],
        "solution": render_numbered_solution(row["steps"]),
        "cot": cot,
        "labels": labels,
        "is_correct": gold[-1] == 1,
        "language": row.get("language"),
        "task": row.get("task", "math"),
        "n_steps": len(gold),
    }


def build_teacher(config: CollectConfig) -> Any:
    from ttcs_yoruba.backends import build_backend
    from ttcs_yoruba.config import InferenceModelConfig

    if not config.teacher:
        raise ValueError(
            "sources[].teacher (or config.teacher) is empty; supply an "
            "OpenAI-compatible reasoning model to synthesise verification chains."
        )
    model_config = InferenceModelConfig.from_dict(config.teacher)
    return build_backend(model_config, default_timeout_s=config.request_timeout_s)


def collect_for_row(
    row: dict[str, Any],
    teacher: Any,
    config: CollectConfig,
) -> list[dict[str, Any]]:
    """Sample teacher verification chains for one row and keep the valid ones."""
    solution = render_numbered_solution(row["steps"])
    message = build_verification_user_message(
        row["question"], solution, config.instruction or COLLECTION_INSTRUCTION
    )
    examples: list[dict[str, Any]] = []
    for _ in range(max(1, config.samples_per_example)):
        output = teacher.generate(
            system_prompt=config.system_prompt,
            user_prompt=message,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        example = build_training_example(
            row, output.response, max_cot_tokens=config.max_cot_tokens
        )
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= max(1, config.max_cots_per_example):
            break
    return examples


def build_dataset(config: CollectConfig) -> list[dict[str, Any]]:
    rows = load_prm_rows(config.prm_data, max_examples=config.max_examples, seed=config.seed)
    if not rows:
        return []
    teacher = build_teacher(config)
    collected: list[dict[str, Any]] = []
    for row in rows:
        collected.extend(collect_for_row(row, teacher, config))
    return collected


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by ``problem`` so all chains for a problem stay together."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["problem"]), []).append(row)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    n_eval = int(round(len(group_ids) * max(0.0, min(1.0, eval_fraction))))
    n_eval = max(1, n_eval) if group_ids and eval_fraction > 0 else 0
    eval_ids = set(group_ids[:n_eval])

    train = [row for row in rows if str(row["problem"]) not in eval_ids]
    evaluate = [row for row in rows if str(row["problem"]) in eval_ids]
    return train, evaluate


def write_built_dataset(
    train: list[dict[str, Any]],
    evaluate: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = resolve_path(output_dir)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "eval.jsonl", evaluate)
    write_jsonl(output / "all.jsonl", all_rows)

    def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_language: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for row in rows:
            by_language[str(row.get("language"))] = by_language.get(str(row.get("language")), 0) + 1
            by_source[str(row.get("source", "prm800k"))] = (
                by_source.get(str(row.get("source", "prm800k")), 0) + 1
            )
        return {
            "total": len(rows),
            "unique_problems": len({str(row.get("problem", "")) for row in rows}),
            "total_step_labels": sum(int(row.get("n_steps", 0)) for row in rows),
            "by_language": by_language,
            "by_source": by_source,
        }

    manifest = {
        "output_dir": str(output),
        "counts": {
            "train": counts(train),
            "eval": counts(evaluate),
            "all": counts(all_rows),
        },
        "paths": {
            "train": str(output / "train.jsonl"),
            "eval": str(output / "eval.jsonl"),
            "all": str(output / "all.jsonl"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


__all__ = [
    "build_dataset",
    "build_teacher",
    "build_training_example",
    "collect_for_row",
    "labels_until_first_error",
    "load_prm_rows",
    "split_rows",
    "write_built_dataset",
]
