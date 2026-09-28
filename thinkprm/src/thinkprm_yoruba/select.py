"""Attach ThinkPRM scores to candidate rows for selection.

ThinkPRM scores are written to the core ``prm_score`` field so the existing
GPU-free ``prm`` selector (and its ``thinkprm`` alias) can rank candidates.
ThinkPRM-specific detail (per-step labels, raw verification chains) is kept in
``thinkprm_*`` fields for inspection.
"""

from __future__ import annotations

from typing import Any

from prm_yoruba.select import question_of, steps_of

from .model import ThinkPrmScorer


def attach_thinkprm_scores(
    rows: list[dict[str, Any]],
    scorer: ThinkPrmScorer,
    *,
    max_steps: int = 256,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Return shallow copies of ``rows`` enriched with ThinkPRM scores."""
    enriched: list[dict[str, Any]] = []
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        questions = [question_of(row) for row in chunk]
        steps_batch = [steps_of(row, max_steps=max_steps) for row in chunk]
        results = scorer.score_batch(questions, steps_batch)
        for row, result in zip(chunk, results):
            new_row = dict(row)
            new_row["prm_score"] = float(result["score"])
            new_row["thinkprm_score"] = float(result["score"])
            new_row["thinkprm_prefix_score"] = float(result["prefix_score"])
            new_row["thinkprm_prefix_scores"] = result["prefix_scores"]
            new_row["thinkprm_step_labels"] = result["step_labels"]
            new_row["thinkprm_step_scores"] = result["step_scores"]
            new_row["thinkprm_n"] = result["n_verifications"]
            new_row["thinkprm_outputs"] = result["outputs"]
            new_row["thinkprm_truncated"] = result["truncated"]
            enriched.append(new_row)
    return enriched


__all__ = ["attach_thinkprm_scores", "question_of", "steps_of"]
