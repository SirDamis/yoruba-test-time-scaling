"""Answer grading for PRM data building and scoring.

Delegates to the main pipeline's extractor so math (``number``) and
multiple-choice QA (``choice``) share one definition of correctness. This is the
key difference from the reference, whose grader only handled math/LaTeX.
"""

from __future__ import annotations

from typing import Any

from ttcs_yoruba.extraction import is_exact_match


def answer_type_of(row: dict[str, Any], default: str = "text") -> str:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("answer_type"):
        return str(metadata["answer_type"])
    return str(row.get("answer_type") or default)


def gold_answer_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("gold_answer") not in (None, ""):
        return str(metadata["gold_answer"])
    return str(row.get("gold_answer") or "")


def is_correct(prediction: str, gold_answer: str, answer_type: str) -> bool:
    """True when ``prediction`` matches ``gold_answer`` under project rules."""
    if gold_answer in (None, ""):
        return False
    return is_exact_match(str(prediction), str(gold_answer), answer_type)
