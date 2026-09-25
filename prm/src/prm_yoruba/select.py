"""Attach PRM scores to candidate rows for selection.

The core selector (:mod:`ttcs_yoruba.selection`) implements the ``prm``
strategy and reads a precomputed ``prm_score`` from each candidate. This module
produces those scores so selection stays GPU-free.
"""

from __future__ import annotations

from typing import Any

from .model import PrmScorer
from .steps import split_into_steps


def question_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("question"):
        return str(metadata["question"])
    return str(row.get("question") or "")


def steps_of(row: dict[str, Any], *, max_steps: int = 256) -> list[str]:
    existing = row.get("steps")
    if isinstance(existing, list) and existing:
        return [str(step) for step in existing][:max_steps]
    return split_into_steps(str(row.get("response", "")), max_steps=max_steps)


def score_row(scorer: PrmScorer, row: dict[str, Any], *, max_steps: int = 256) -> dict[str, Any]:
    """Return the scorer output for one candidate row."""
    return scorer.score(question_of(row), steps_of(row, max_steps=max_steps))


def attach_prm_scores(
    rows: list[dict[str, Any]],
    scorer: PrmScorer,
    *,
    max_steps: int = 256,
) -> list[dict[str, Any]]:
    """Return shallow copies of ``rows`` with ``prm_score`` + ``prm_*`` detail."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        scored = score_row(scorer, row, max_steps=max_steps)
        new_row = dict(row)
        new_row["prm_score"] = float(scored["score"])
        new_row["prm_step_scores"] = scored["step_scores"]
        new_row["prm_last"] = scored["last"]
        new_row["prm_max"] = scored["max"]
        new_row["prm_min"] = scored["min"]
        new_row["prm_mean"] = scored["mean"]
        enriched.append(new_row)
    return enriched
