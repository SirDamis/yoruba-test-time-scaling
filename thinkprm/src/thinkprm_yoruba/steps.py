"""Step segmentation shared with the discriminative PRM.

Our candidate traces put one reasoning step per ``\\n`` line (the ``Final
answer:`` line is dropped). That segmentation already lives in
:mod:`prm_yoruba.steps`; ThinkPRM consumes the same traces, so we reuse it and
only add the ``Step {i}: `` numbering that the ThinkPRM template expects.
"""

from __future__ import annotations

import re

from prm_yoruba.steps import clean_step_text, split_into_steps

_STEP_PREFIX_RE = re.compile(r"(?i)^\s*(?:\*\*\s*)?step\s*\d+\s*[:.)-]\s*(?:\*\*)?\s*")


def strip_step_prefix(step: str) -> str:
    """Drop a leading ``Step k:`` / ``**Step k.**`` marker if present."""
    return _STEP_PREFIX_RE.sub("", str(step or "").strip()).strip()


def render_numbered_solution(steps: list[str]) -> str:
    """Render steps as ``Step 1: ...\\nStep 2: ...`` (ThinkPRM's format)."""
    lines = []
    for index, step in enumerate(steps, start=1):
        cleaned = clean_step_text(strip_step_prefix(step))
        if cleaned:
            lines.append(f"Step {index}: {cleaned}")
    return "\n".join(lines)


def parse_numbered_solution(text: str) -> list[str]:
    """Recover step texts from a ``Step {i}:``-numbered solution."""
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"(?im)^\s*Step\s*\d+\s*[:.)-]\s*(.+?)\s*$", str(text or "")
        )
    ]


__all__ = [
    "clean_step_text",
    "parse_numbered_solution",
    "render_numbered_solution",
    "split_into_steps",
    "strip_step_prefix",
]
