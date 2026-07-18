"""Tests for numeric equality (scoring) and numeric majority-vote keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.extraction import (
    is_exact_match,
    normalize_for_match,
    numbers_equal,
    numeric_match_key,
    parse_number,
)
from ttcs_yoruba.selection import select_candidate


def test_numbers_equal_decimal_variants() -> None:
    assert numbers_equal("105", "105")
    assert numbers_equal("105.0", "105")
    assert numbers_equal("105.00", "105")
    assert numbers_equal("11", "11.0")
    assert numbers_equal("1,050", "1050")
    assert numbers_equal("-5.0", "-5")
    assert not numbers_equal("10", "11")
    assert not numbers_equal("105", "106")


def test_is_exact_match_number_uses_numeric_equality() -> None:
    assert is_exact_match("105.0", "105", "number")
    assert is_exact_match("11.00", "11", "number")
    assert is_exact_match("1,050", "1050", "number")
    assert not is_exact_match("10", "11", "number")
    assert not is_exact_match("", "11", "number")
    # Non-number types still use text normalization.
    assert is_exact_match("Hello!", "hello", "text")
    assert not is_exact_match("A", "B", "choice")


def test_numeric_match_key_pools_decimal_variants() -> None:
    assert numeric_match_key("11") == numeric_match_key("11.0")
    assert numeric_match_key("11") == numeric_match_key("11.00")
    assert numeric_match_key("105.0") == numeric_match_key("105")
    assert numeric_match_key("11") != numeric_match_key("12")
    # Guard against the old normalize_for_match bug: "11.0" -> "110"
    assert normalize_for_match("11.0") == "110"
    assert numeric_match_key("11.0") == "num:11"
    assert parse_number("not-a-number") is None


def test_majority_vote_pools_numeric_equivalents() -> None:
    candidates = [
        {"sample_index": 0, "extracted_answer": "11.0"},
        {"sample_index": 1, "extracted_answer": "11"},
        {"sample_index": 2, "extracted_answer": "12"},
    ]
    result = select_candidate(candidates, "majority_vote", answer_type="number")
    assert result.selected_answer in {"11", "11.0"}
    assert result.selected_sample_index in {0, 1}
    # Two votes for the same numeric key.
    assert any(count == 2 for count in result.vote_counts.values())


def test_majority_vote_infers_number_from_metadata() -> None:
    candidates = [
        {
            "sample_index": 0,
            "extracted_answer": "11.0",
            "metadata": {"answer_type": "number"},
        },
        {
            "sample_index": 1,
            "extracted_answer": "11",
            "metadata": {"answer_type": "number"},
        },
        {
            "sample_index": 2,
            "extracted_answer": "12",
            "metadata": {"answer_type": "number"},
        },
    ]
    result = select_candidate(candidates, "majority_vote")
    assert result.selected_answer in {"11", "11.0"}


def test_majority_vote_text_still_uses_normalize() -> None:
    candidates = [
        {"sample_index": 0, "extracted_answer": "Olùkọ́"},
        {"sample_index": 1, "extracted_answer": "oluko"},
        {"sample_index": 2, "extracted_answer": "dókítà"},
    ]
    # Without number type, punctuation/case normalization applies as before.
    result = select_candidate(candidates, "majority_vote", answer_type="text")
    assert result.selected_sample_index in {0, 1}


if __name__ == "__main__":
    test_numbers_equal_decimal_variants()
    test_is_exact_match_number_uses_numeric_equality()
    test_numeric_match_key_pools_decimal_variants()
    test_majority_vote_pools_numeric_equivalents()
    test_majority_vote_infers_number_from_metadata()
    test_majority_vote_text_still_uses_normalize()
    print("ok")
