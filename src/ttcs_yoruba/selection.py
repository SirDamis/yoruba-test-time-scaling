from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .extraction import normalize_for_match, numeric_match_key


SUPPORTED_SELECTIONS = {"first", "majority_vote", "prm"}
# ``verifier`` is an alias for the process-reward-model strategy.
PRM_ALIASES = {"prm", "verifier"}
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

    Supported strategies: ``first``, ``majority_vote``, ``prm`` (alias
    ``verifier``).

    When ``answer_type == "number"`` (or is present on candidate metadata),
    majority vote pools answers by numeric value so ``11`` and ``11.0`` agree.

    The ``prm`` strategy picks the candidate with the highest precomputed
    ``prm_score`` (a process reward model score attached by the PRM scorer);
    it never loads a model itself, so it stays usable without a GPU.
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

    if strategy in PRM_ALIASES:
        return _prm_select(candidates)

    raise ValueError(
        f"Unsupported selection strategy: {strategy!r}. "
        f"Expected one of {sorted(SUPPORTED_SELECTIONS)}."
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

    # Empty answers (failed extractions) get unique __empty_* keys and must
    # never beat real answers on tie-breaks; prefer non-empty unless all empty.
    non_empty_keys = [key for key in vote_counts if not key.startswith("__empty_")]
    pool = non_empty_keys or list(vote_counts)
    selected_key = min(pool, key=lambda key: (-vote_counts[key], first_seen[key]))
    selected_row = candidates[first_seen[selected_key]]
    return SelectionResult(
        selected_sample_index=int(selected_row["sample_index"]),
        selected_answer=display_answers[selected_key],
        vote_counts=vote_counts,
    )


def _prm_score(row: dict[str, object]) -> float | None:
    value = row.get("prm_score")
    if value is None:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, dict):
            value = metadata.get("prm_score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prm_select(candidates: list[dict[str, object]]) -> SelectionResult:
    scored: list[tuple[float, int, dict[str, object]]] = []
    for position, row in enumerate(candidates):
        score = _prm_score(row)
        if score is not None:
            scored.append((score, position, row))
    if not scored:
        raise ValueError(
            "prm selection requires a precomputed 'prm_score' on candidates "
            "(or candidate metadata). Run prm/scripts/score_candidates.py first."
        )

    # Highest score wins; ties fall back to the earliest sample for determinism.
    score, position, row = max(scored, key=lambda item: (item[0], -item[1]))
    return SelectionResult(
        selected_sample_index=int(row["sample_index"]),
        selected_answer=str(row.get("extracted_answer", "")),
        vote_counts={},
        metadata={
            "prm_score": score,
            "prm_scores": {str(r["sample_index"]): s for s, _, r in scored},
        },
    )
