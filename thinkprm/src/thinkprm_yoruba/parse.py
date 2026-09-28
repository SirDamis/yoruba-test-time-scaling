"""Parse ThinkPRM verification outputs.

ThinkPRM emits one ``\\boxed{correct}`` / ``\\boxed{incorrect}`` judgement per
solution step, then a final ``Is the solution correct? Yes/No`` decision. These
helpers turn raw generations into per-step labels and a prefix decision without
importing torch, so they are unit-testable on CPU.
"""

from __future__ import annotations

import re

CORRECT_TOKEN = "correct"
INCORRECT_TOKEN = "incorrect"
DECISION_STRING = "Is the solution correct?"

_BOXED_DECISION_RE = re.compile(
    r"\\boxed\{\s*(?:\\text\{\s*)?(correct|incorrect)\s*\}?\s*\}"
)
_DECISION_RE = re.compile(r"is the solution correct\?\s*\**(yes|no)\b", re.IGNORECASE)


def normalize_boxed_decisions(text: str) -> str:
    """Collapse ``\\boxed{\\text{correct}}`` (and variants) to ``\\boxed{correct}``."""
    return re.sub(
        r"\\boxed\{\s*\\text\{\s*(correct|incorrect)\s*\}\s*\}",
        r"\\boxed{\1}",
        str(text or ""),
    )


def extract_boxed_decisions(
    text: str,
    *,
    correct_token: str = CORRECT_TOKEN,
    incorrect_token: str = INCORRECT_TOKEN,
) -> list[str]:
    """Return the ordered ``"correct"``/``"incorrect"`` judgements in ``text``."""
    pattern = re.compile(
        r"\\boxed\{\s*(?:\\text\{\s*)?("
        + re.escape(correct_token)
        + r"|"
        + re.escape(incorrect_token)
        + r")\s*\}?\s*\}"
    )
    return [match.group(1) for match in pattern.finditer(normalize_boxed_decisions(text))]


def extract_step_labels(
    text: str,
    *,
    correct_token: str = CORRECT_TOKEN,
    incorrect_token: str = INCORRECT_TOKEN,
) -> list[int]:
    """Map boxed judgements to ``1`` (correct) / ``0`` (incorrect)."""
    return [
        1 if decision == correct_token else 0
        for decision in extract_boxed_decisions(
            text, correct_token=correct_token, incorrect_token=incorrect_token
        )
    ]


def extract_decision(text: str, *, decision_string: str = DECISION_STRING) -> str | None:
    """Return the final ``"Yes"``/``"No"`` prefix decision, or ``None``."""
    pattern = re.compile(
        re.escape(decision_string) + r"\s*\**\s*(yes|no)\b", re.IGNORECASE
    )
    matches = pattern.findall(str(text or ""))
    if not matches:
        return None
    return "Yes" if matches[-1].lower() == "yes" else "No"


def truncate_after_last_decision(
    text: str,
    *,
    correct_token: str = CORRECT_TOKEN,
    incorrect_token: str = INCORRECT_TOKEN,
) -> str:
    """Keep ``text`` up to and including its last boxed step judgement."""
    normalized = normalize_boxed_decisions(text)
    pattern = re.compile(
        r"\\boxed\{\s*(?:\\text\{\s*)?("
        + re.escape(correct_token)
        + r"|"
        + re.escape(incorrect_token)
        + r")\s*\}?\s*\}"
    )
    last = None
    for match in pattern.finditer(normalized):
        last = match
    return normalized[: last.end()].strip() if last is not None else normalized.strip()


def strip_think(text: str) -> str:
    """Drop an outer ``<think>...</think>`` block, keeping the final decision tail."""
    raw = str(text or "")
    close = raw.rfind("</think>")
    if close != -1:
        tail = raw[close + len("</think>") :].strip()
        if tail:
            return tail
    open_index = raw.find("<think>")
    if open_index != -1:
        inner = raw[open_index + len("<think>") :].replace("</think>", "")
        return inner.strip()
    return raw.strip()


def format_step_scores(labels: list[int]) -> list[float]:
    """Represent ``1``/``0`` step labels as a ``0..1`` score list."""
    return [float(label) for label in labels]


def pad_step_labels(labels: list[int], n_steps: int) -> list[int]:
    """Pad a short label list with ``0`` (incorrect) up to ``n_steps``."""
    labels = list(labels[:n_steps])
    if len(labels) < n_steps:
        labels.extend([0] * (n_steps - len(labels)))
    return labels


__all__ = [
    "CORRECT_TOKEN",
    "DECISION_STRING",
    "INCORRECT_TOKEN",
    "extract_boxed_decisions",
    "extract_decision",
    "extract_step_labels",
    "format_step_scores",
    "normalize_boxed_decisions",
    "pad_step_labels",
    "strip_think",
    "truncate_after_last_decision",
]
