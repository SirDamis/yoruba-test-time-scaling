"""Step segmentation for discriminative process reward models.

Our candidates put **one reasoning step per line** (``\\n``) and separate the
final answer with a blank line (``\\n\\nFinal answer: ...``). The released
PRM800K format instead uses a dedicated step tag, so at scoring time we must
re-segment each trace on ``\\n`` and re-insert that tag ourselves.

This differs from ``prm/reference/eval_prm.py``, which split Qwen generations on
blank lines — that would collapse all our steps into one block.
"""

from __future__ import annotations

import re

# Our CoT prompts finish with a single ``Final answer:`` line (optionally bold).
_FINAL_LINE_RE = re.compile(r"(?i)^\s*\**\s*final\s+answer\s*\**\s*[:\-]")
# "Reasoning:", "**Solution:**", "Explanation: step ..." — everything after the
# last such header is the reasoning body.
_REASONING_HEADER_RE = re.compile(
    r"(?i)^\s*\**\s*(?:reasoning|solution|explanation|analysis|chain\s+of\s+thought|cot)"
    r"\s*\**\s*:\s*(?P<rest>.*)$"
)
# The model sometimes echoes the question/translation before reasoning.
_QUESTION_HEADER_RE = re.compile(
    r"(?i)^\s*\**\s*(?:question(?:\s*\([^)]*\))?|english\s+translation|translation|problem)"
    r"\s*\**\s*:\s*(?P<rest>.*)$"
)
# "Step 3:", "3.", "**Step 3:**" prefixes and "- "/"* "/"• " list markers.
_STEP_PREFIX_RE = re.compile(r"(?i)^\s*(?:\*\*\s*)?(?:step\s*\d+\s*[:.)-]|\d+[.)])\s*(?:\*\*)?\s*")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Qwen2.5-Math-PRM / Math-Shepherd boundary used by the reference code.
QWEN_STEP_TAG = " \n\n\n\n\n"
MISTRAL_STEP_TAG = "ки"
QWEN_GOOD_TOKEN = " +"
QWEN_BAD_TOKEN = " -"
MISTRAL_GOOD_TOKEN = "+"
MISTRAL_BAD_TOKEN = "-"

# Official Qwen2.5-Math-PRM-* interface: one special token after each step and a
# 2-class reward head read at those positions.
QWEN_PRM_STEP_SEP = "<extra_0>"
DEFAULT_PRM_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def strip_final_answer(text: str) -> str:
    """Drop trailing ``Final answer:`` line(s) so they are not scored as steps."""
    lines = [line for line in (text or "").splitlines() if not _FINAL_LINE_RE.match(line)]
    return "\n".join(lines).strip()


def clean_step_text(step: str) -> str:
    """Strip list/step markers and markdown/LaTeX noise the Qwen PRM never saw."""
    text = _LIST_MARKER_RE.sub("", (step or "").strip())
    text = _STEP_PREFIX_RE.sub("", text)
    text = text.replace("**", " ").replace("\\times", "*")
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _reasoning_lines(lines: list[str]) -> list[str]:
    """Keep only the reasoning body: text after the last reasoning header.

    A leading question/translation block is dropped when it precedes the header
    (or, if there is no header, when it is followed by a blank line).
    """
    header_index: int | None = None
    inline_first: str | None = None
    for index, line in enumerate(lines):
        match = _REASONING_HEADER_RE.match(line)
        if match:
            header_index = index
            rest = match.group("rest").strip()
            inline_first = rest or None

    if header_index is not None:
        body = lines[header_index + 1 :]
        if inline_first:
            body = [inline_first, *body]
        return body

    body = list(lines)
    index = 0
    while index < len(body) and not body[index].strip():
        index += 1
    if index < len(body) and _QUESTION_HEADER_RE.match(body[index]):
        question_header = _QUESTION_HEADER_RE.match(body[index])
        inline_question = question_header.group("rest").strip() if question_header else ""
        cursor = index + 1
        while cursor < len(body) and body[cursor].strip():
            cursor += 1
        if cursor < len(body):  # a blank line closed the question block
            return body[cursor:]
        if inline_question:
            return body[index + 1 :]
    return body


def split_into_steps(text: str, *, max_steps: int = 256) -> list[str]:
    """Segment a reasoning trace into ordered steps.

    Uses ``\\n`` as the step delimiter (one step per non-empty line), drops the
    final-answer line and any echoed question/translation, and falls back to
    sentence splitting only for single-line traces.
    """
    raw = text or ""
    if not raw.strip():
        return []

    body = _reasoning_lines(raw.splitlines())
    steps: list[str] = []
    for line in body:
        if _REASONING_HEADER_RE.match(line) or _QUESTION_HEADER_RE.match(line):
            continue
        cleaned = clean_step_text(line)
        if cleaned and not _FINAL_LINE_RE.match(cleaned):
            steps.append(cleaned)

    if len(steps) <= 1:
        fallback_lines = []
        for line in body:
            if _REASONING_HEADER_RE.match(line) or _QUESTION_HEADER_RE.match(line):
                continue
            cleaned = clean_step_text(line)
            if cleaned and not _FINAL_LINE_RE.match(cleaned):
                fallback_lines.append(cleaned)
        joined = " ".join(steps) if steps else " ".join(line for line in fallback_lines if line)
        sentences = [clean_step_text(part) for part in _SENTENCE_SPLIT_RE.split(joined)]
        sentences = [part for part in sentences if part]
        if len(sentences) > len(steps):
            steps = sentences

    return steps[:max_steps] if max_steps else steps


def render_tagged_process(steps: list[str], step_tag: str) -> str:
    """Render steps as ``step<tag>step<tag>...`` (one tag per step)."""
    cleaned = [clean_step_text(step) for step in steps if clean_step_text(step)]
    return step_tag.join(cleaned) + step_tag if cleaned else ""


def split_tagged_process(process: str, step_tag: str) -> list[str]:
    """Recover steps from a ``process`` string built with ``render_tagged_process``."""
    parts = str(process or "").split(step_tag)
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [part.strip() for part in parts if part.strip()]


def render_sep_joined(steps: list[str], sep: str = QWEN_PRM_STEP_SEP) -> str:
    """Render steps as the official PRM assistant content: ``s1<sep>s2<sep>``.

    Positions are preserved (an empty step still gets its separator) so the
    number of ``<extra_0>`` tokens always equals ``len(steps)`` and stays aligned
    with per-step labels during training.
    """
    cleaned = [clean_step_text(step) for step in steps]
    return sep.join(cleaned) + sep if cleaned else ""


def render_scoring_input(question: str, steps: list[str], step_tag: str) -> str:
    """Render the exact string scored at inference (reference ``input_for_prm``)."""
    cleaned = [clean_step_text(step) for step in steps if clean_step_text(step)]
    output = f"{step_tag} ".join(cleaned)
    output += step_tag
    return f"{question} {output}"
