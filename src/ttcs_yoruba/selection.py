from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .extraction import normalize_for_match, numeric_match_key


SUPPORTED_SELECTIONS = {"first", "majority_vote"}
# Alias kept for callers that used the old local-only constant.
SUPPORTED_LOCAL_SELECTIONS = SUPPORTED_SELECTIONS


@dataclass(frozen=True)
class SelectionResult:
    selected_sample_index: int
    selected_answer: str
    vote_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


def select_candidate(
    candidates: list[dict[str, object]],
    strategy: str,
    *,
    answer_type: str | None = None,
) -> SelectionResult:
    """Select one candidate from a sampled set.

    Supported strategies: ``first``, ``majority_vote``.

    When ``answer_type == "number"`` (or is present on candidate metadata),
    majority vote pools answers by numeric value so ``11`` and ``11.0`` agree.

    LLM-as-judge / external verifiers are deferred (not implemented).
    """
    if not candidates:
        return SelectionResult(selected_sample_index=-1, selected_answer="", vote_counts={})

    if strategy == "first":
        first = candidates[0]
        return SelectionResult(
            selected_sample_index=int(first["sample_index"]),
            selected_answer=str(first.get("extracted_answer", "")),
            vote_counts={str(first.get("extracted_answer", "")): 1},
        )

    if strategy == "majority_vote":
        return _majority_vote(candidates, answer_type=answer_type)

    raise ValueError(
        f"Unsupported selection strategy: {strategy!r}. "
        f"Expected one of {sorted(SUPPORTED_SELECTIONS)}. "
        f"(LLM-as-judge verifier is deferred.)"
    )


def _resolve_answer_type(
    candidates: list[dict[str, object]],
    answer_type: str | None,
) -> str | None:
    if answer_type:
        return answer_type
    for row in candidates:
        meta = row.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("answer_type"):
            return str(meta["answer_type"])
        if row.get("answer_type"):
            return str(row["answer_type"])
    return None


def _vote_key(answer: str, *, answer_type: str | None, position: int) -> str:
    if not answer:
        return f"__empty_{position}"
    if answer_type == "number":
        return numeric_match_key(answer)
    return normalize_for_match(answer)


def _majority_vote(
    candidates: list[dict[str, object]],
    *,
    answer_type: str | None = None,
) -> SelectionResult:
    resolved_type = _resolve_answer_type(candidates, answer_type)
    vote_counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    display_answers: dict[str, str] = {}
    for position, row in enumerate(candidates):
        answer = str(row.get("extracted_answer", ""))
        key = _vote_key(answer, answer_type=resolved_type, position=position)
        vote_counts[key] = vote_counts.get(key, 0) + 1
        first_seen.setdefault(key, position)
        display_answers.setdefault(key, answer)

    selected_key = min(vote_counts, key=lambda key: (-vote_counts[key], first_seen[key]))
    selected_row = candidates[first_seen[selected_key]]
    return SelectionResult(
        selected_sample_index=int(selected_row["sample_index"]),
        selected_answer=display_answers[selected_key],
        vote_counts=vote_counts,
    )
