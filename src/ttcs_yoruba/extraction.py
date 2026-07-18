from __future__ import annotations

import math
import re
import string


FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer|idahun\s+ikeyin)\s*[:\-]\s*(.+?)\s*$"
)

# Absolute tolerance for GSM-style integer/simple-decimal equality.
_NUM_EPS = 1e-6


def extract_answer(response: str, answer_type: str, choices: list[str] | None = None) -> str:
    candidate = find_final_answer_text(response)
    if answer_type == "choice":
        return extract_choice_answer(candidate, choices)
    if answer_type == "number":
        return extract_number_answer(candidate)
    return clean_answer_text(candidate)


def find_final_answer_text(response: str) -> str:
    matches = FINAL_ANSWER_RE.findall(response or "")
    if matches:
        return clean_answer_text(matches[-1])
    lines = [line.strip() for line in (response or "").splitlines() if line.strip()]
    return clean_answer_text(lines[-1]) if lines else ""


def extract_choice_answer(candidate: str, choices: list[str] | None = None) -> str:
    valid_labels = [chr(ord("A") + index) for index in range(len(choices or []))]
    if not valid_labels:
        valid_labels = ["A", "B", "C", "D", "E"]

    label_match = re.search(r"\b([A-Z])\b", candidate.upper())
    if label_match and label_match.group(1) in valid_labels:
        return label_match.group(1)

    normalized_candidate = normalize_for_match(candidate)
    for index, choice in enumerate(choices or []):
        label = chr(ord("A") + index)
        normalized_choice = normalize_for_match(strip_choice_label(choice))
        if normalized_candidate == normalized_choice:
            return label
    return clean_answer_text(candidate)


def extract_number_answer(candidate: str) -> str:
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", candidate)
    if match:
        return match.group(0).replace(",", "")
    return clean_answer_text(candidate)


def parse_number(value: str) -> float | None:
    """Extract and parse a number; return None if not parseable."""
    text = extract_number_answer(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def numbers_equal(a: str, b: str, *, eps: float = _NUM_EPS) -> bool:
    """True when *a* and *b* are the same numeric value (e.g. 105 vs 105.0)."""
    na, nb = parse_number(a), parse_number(b)
    if na is None or nb is None:
        return extract_number_answer(a) == extract_number_answer(b)
    return math.isclose(na, nb, rel_tol=0.0, abs_tol=eps)


def numeric_match_key(value: str, *, eps: float = _NUM_EPS) -> str:
    """Stable key so 11 / 11.0 / 11.00 pool together in majority vote."""
    n = parse_number(value)
    if n is None:
        return normalize_for_match(value)
    if math.isclose(n, round(n), abs_tol=eps):
        return f"num:{int(round(n))}"
    return f"num:{round(n, 6)}"


def clean_answer_text(value: str) -> str:
    return (value or "").strip().strip("`*_ \t\r\n")


def strip_choice_label(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0].isalpha() and stripped[1] in {".", ")"}:
        return stripped[2:].strip()
    return stripped


def normalize_for_match(value: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(strip_choice_label(value).lower().translate(table).split())


def is_exact_match(prediction: str, gold_answer: str, answer_type: str) -> bool:
    if not gold_answer:
        return False
    if answer_type == "number":
        return numbers_equal(prediction, gold_answer)
    return normalize_for_match(prediction) == normalize_for_match(gold_answer)
