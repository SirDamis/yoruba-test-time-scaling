"""Tests for pool-based TTC estimators (unbiased pass@k, maj@k, diversity)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.estimators import (
    majority_accuracy_at_k,
    nested_pool_estimates,
    pass_at_k_unbiased,
    pool_diversity,
)


def _cand(sample_index: int, answer: str) -> dict:
    return {
        "sample_index": sample_index,
        "extracted_answer": answer,
        "metadata": {"answer_type": "number", "gold_answer": "5"},
    }


def _nested_cand(example_id: str, sample_index: int, answer: str, gold: str) -> dict:
    return {
        "example_id": example_id,
        "model": "m",
        "method": "ttc_n4",
        "n": 4,
        "sample_index": sample_index,
        "extracted_answer": answer,
        "token_count": 10,
        "prompt_token_count": 20,
        "metadata": {
            "dataset": "d",
            "answer_type": "number",
            "gold_answer": gold,
            "nested": True,
            "nested_group_id": "ttc",
            "nested_pool_n": 4,
        },
    }


def test_pass_at_k_unbiased_values() -> None:
    assert pass_at_k_unbiased(64, 0, 1) == 0.0
    assert pass_at_k_unbiased(64, 1, 1) == pytest.approx(1 / 64)
    assert pass_at_k_unbiased(8, 2, 4) == pytest.approx(1 - 15 / 70)
    assert pass_at_k_unbiased(4, 1, 4) == 1.0
    assert pass_at_k_unbiased(4, 1, 9) == 1.0  # k > n clamps
    assert pass_at_k_unbiased(0, 0, 1) == 0.0


def test_majority_accuracy_full_pool_is_deterministic() -> None:
    pool = [_cand(0, "5"), _cand(1, "5"), _cand(2, "3"), _cand(3, "5")]
    mean, std = majority_accuracy_at_k(
        pool, 4, answer_type="number", num_subsets=10, rng=random.Random(0)
    )
    assert mean == 1.0
    assert std == 0.0


def test_pool_diversity_groups_numeric_variants() -> None:
    pool = [_cand(0, "5"), _cand(1, "5.0"), _cand(2, "3"), _cand(3, "")]
    diversity = pool_diversity(pool, answer_type="number")
    assert diversity["pool_size"] == 4
    assert diversity["distinct_answers"] == 2  # 5/5.0 pool together; 3 is second
    assert diversity["empty_extractions"] == 1
    assert diversity["degenerate"] is False


def test_nested_pool_estimates_synthetic() -> None:
    candidates = []
    for example_id, answers, gold in [
        ("e1", ["5", "5", "3", "5"], "5"),
        ("e2", ["1", "2", "3", "4"], "5"),
    ]:
        for sample_index, answer in enumerate(answers):
            candidates.append(_nested_cand(example_id, sample_index, answer, gold))

    rows = {r["k"]: r for r in nested_pool_estimates(candidates, num_subsets=50, seed=1)}
    assert set(rows) == {1, 2, 4}
    assert rows[1]["pass_at_k"] == pytest.approx(0.375)  # mean(0.75, 0.0)
    assert rows[4]["pass_at_k"] == pytest.approx(0.5)
    assert rows[4]["maj_at_k_mean"] == pytest.approx(0.5)
    assert rows[4]["degenerate_pool_rate"] == 0.0
    assert rows[1]["distinct_answers_mean"] == pytest.approx(3.0)  # mean(2, 4)
    # Token tracking: 10 completion / 20 prompt per sample.
    assert rows[4]["mean_completion_tokens_per_sample"] == pytest.approx(10.0)
    assert rows[4]["completion_tokens_per_example_at_k"] == pytest.approx(40.0)
    assert rows[4]["prompt_tokens_per_example_at_k"] == pytest.approx(80.0)
    assert rows[4]["total_tokens_per_example_at_k"] == pytest.approx(120.0)
    # maj@4 = 0.5 -> tokens per correct = 40 / 0.5 = 80
    assert rows[4]["completion_tokens_per_correct"] == pytest.approx(80.0)


if __name__ == "__main__":
    test_pass_at_k_unbiased_values()
    test_majority_accuracy_full_pool_is_deterministic()
    test_pool_diversity_groups_numeric_variants()
    test_nested_pool_estimates_synthetic()
    print("ok")
