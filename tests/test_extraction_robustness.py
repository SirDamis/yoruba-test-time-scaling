"""Regression tests for extraction, token accounting, and exemplar fixes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.backends import _completion_token_count
from ttcs_yoruba.extraction import extract_answer, extract_choice_answer, find_final_answer_text
from ttcs_yoruba.prompting import TASK_EXEMPLARS
from ttcs_yoruba.selection import select_candidate


def test_choice_ignores_standalone_pronoun_i() -> None:
    candidate = "I think the answer is B."
    assert extract_choice_answer(candidate, ["x", "y", "z", "w"]) == "B"


def test_choice_keyword_anchored_letter() -> None:
    assert extract_choice_answer("Answer: C", ["a", "b", "c", "d"]) == "C"
    assert (
        extract_choice_answer("Ìdáhùn ni D", ["a", "b", "c", "d"])
        == "D"
    )
    assert (
        extract_choice_answer("The final answer is B", ["a", "b", "c", "d"])
        == "B"
    )


def test_choice_prefers_last_valid_letter() -> None:
    candidate = "A is wrong because of cost, so the answer is D"
    assert extract_choice_answer(candidate, ["a", "b", "c", "d"]) == "D"


def test_choice_plain_letter_and_full_text_fallback() -> None:
    assert extract_choice_answer("B", ["a", "b", "c", "d"]) == "B"
    # No valid letter anywhere -> falls back to cleaned text.
    assert extract_choice_answer("Solar energy", ["Coal", "Oil", "Solar energy"]) == "C"


def test_number_takes_last_number_in_line() -> None:
    assert extract_answer("Reasoning: 5 - 2 = 3.\nFinal answer: 5 - 2 = 3", "number") == "3"
    assert extract_answer("Final answer: 1,200", "number") == "1200"
    assert extract_answer("Final answer: 42", "number") == "42"


def test_completion_token_count_never_uses_total_tokens() -> None:
    # Provider-reported completion tokens win.
    assert (
        _completion_token_count(
            {"completion_tokens": 43, "total_tokens": 794},
            completion_keys=("completion_tokens",),
            content="short",
        )
        == 43
    )
    # Missing/zero completion tokens -> word estimate, NOT prompt-inclusive total.
    assert (
        _completion_token_count(
            {"total_tokens": 9999},
            completion_keys=("completion_tokens",),
            content="one two three four",
        )
        == 4
    )
    assert (
        _completion_token_count(
            {"completion_tokens": 0, "total_tokens": 500},
            completion_keys=("completion_tokens",),
            content="",
        )
        == 1
    )


def test_math_exemplars_do_not_duplicate_afrimgsm_items() -> None:
    """Guard against few-shot exemplars mirroring AfriMGSM test items."""
    banned_snippets = [
        "Kọ̀mpútà mẹ́sàn-án",  # computers 9 + 5/day Mon-Fri (test item)
        "ṣokolétì mejilelọgbọn",  # chocolates 32/42 eat N (test item)
        "Títí ní ìwé mẹ́ta",  # books 3 + 2 packs x 4 = 11 (same shape as test row 1)
    ]
    for ex in TASK_EXEMPLARS["math"]:
        for snippet in banned_snippets:
            assert snippet not in ex.question


def _candidate(sample_index: int, answer: str) -> dict:
    return {
        "sample_index": sample_index,
        "extracted_answer": answer,
    }


def test_majority_vote_never_prefers_empty_answers() -> None:
    candidates = [
        _candidate(0, ""),  # failed extraction, first position
        _candidate(1, "42"),
        _candidate(2, "43"),
    ]
    result = select_candidate(candidates, "majority_vote", answer_type="number")
    assert result.selected_answer == "42"
    assert result.selected_sample_index == 1


def test_majority_vote_all_empty_returns_empty() -> None:
    candidates = [_candidate(i, "") for i in range(3)]
    result = select_candidate(candidates, "majority_vote", answer_type="number")
    assert result.selected_answer == ""


def test_fallback_rejects_long_reasoning_lines() -> None:
    long_line = (
        "so adding the two quantities together and subtracting the remainder "
        "we conclude that the total number of items must be forty two units"
    )
    # Only line is long reasoning prose -> extraction failure, not garbage.
    assert find_final_answer_text(long_line) == ""
    assert find_final_answer_text(f"Step one: {long_line}") == ""


def test_fallback_accepts_short_answer_lines() -> None:
    assert find_final_answer_text("Reasoning: 5 - 2 = 3.\nSo the answer is 3") == "So the answer is 3"
    assert find_final_answer_text("blah\n42") == "42"


if __name__ == "__main__":
    test_choice_ignores_standalone_pronoun_i()
    test_choice_keyword_anchored_letter()
    test_choice_prefers_last_valid_letter()
    test_choice_plain_letter_and_full_text_fallback()
    test_number_takes_last_number_in_line()
    test_completion_token_count_never_uses_total_tokens()
    test_math_exemplars_do_not_duplicate_afrimgsm_items()
    test_majority_vote_never_prefers_empty_answers()
    test_majority_vote_all_empty_returns_empty()
    test_fallback_rejects_long_reasoning_lines()
    test_fallback_accepts_short_answer_lines()
    print("ok")
