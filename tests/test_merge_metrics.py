"""Tests for safe multi-run metrics merge (no double-counting)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.metrics import ConditionMetrics, merge_condition_metrics


def _m(
    *,
    run_ids: list[str],
    examples: int = 10,
    correct: int = 5,
    tokens: int = 100,
    dataset: str = "afrimgsm",
    model: str = "qwen3-4b",
    method: str = "english_cot_ttc_n4",
    n: int = 4,
) -> ConditionMetrics:
    return ConditionMetrics(
        dataset=dataset,
        model=model,
        model_size_label="4B",
        method=method,
        prompt_style="english_cot",
        reasoning_language="en",
        selection="majority_vote",
        n=n,
        total_examples=examples,
        select_correct=correct,
        pass_at_n_correct=correct,
        total_candidates=examples * n,
        total_tokens=tokens,
        total_latency_s=1.0,
        total_estimated_cost=0.0,
        run_ids=list(run_ids),
    )


def test_disjoint_run_ids_are_summed() -> None:
    a = _m(run_ids=["run_a"], examples=10, correct=4, tokens=100)
    b = _m(run_ids=["run_b"], examples=20, correct=10, tokens=200)
    merged = merge_condition_metrics([a, b])
    assert len(merged) == 1
    m = merged[0]
    assert m.total_examples == 30
    assert m.select_correct == 14
    assert m.total_tokens == 300
    assert set(m.run_ids) == {"run_a", "run_b"}


def test_same_run_id_is_not_double_counted() -> None:
    a = _m(run_ids=["run_a"], examples=10, correct=4, tokens=100)
    dup = _m(run_ids=["run_a"], examples=10, correct=4, tokens=100)
    merged = merge_condition_metrics([a, dup])
    assert len(merged) == 1
    m = merged[0]
    assert m.total_examples == 10
    assert m.select_correct == 4
    assert m.total_tokens == 100
    assert m.run_ids == ["run_a"]


def test_same_run_id_keeps_larger_if_only_that_run() -> None:
    small = _m(run_ids=["run_a"], examples=5, correct=1, tokens=50)
    large = _m(run_ids=["run_a"], examples=50, correct=20, tokens=500)
    merged = merge_condition_metrics([small, large])
    assert merged[0].total_examples == 50
    assert merged[0].select_correct == 20
    assert merged[0].total_tokens == 500


def test_partial_overlap_is_skipped() -> None:
    ab = _m(run_ids=["run_a", "run_b"], examples=30, correct=12, tokens=300)
    a = _m(run_ids=["run_a"], examples=10, correct=4, tokens=100)
    # First item establishes a+b; second shares run_a → skipped (no double-count).
    merged = merge_condition_metrics([ab, a])
    assert merged[0].total_examples == 30
    assert set(merged[0].run_ids) == {"run_a", "run_b"}


def test_different_conditions_not_merged() -> None:
    a = _m(run_ids=["r1"], n=4, method="english_cot_ttc_n4")
    b = _m(run_ids=["r1"], n=8, method="english_cot_ttc_n8")
    merged = merge_condition_metrics([a, b])
    assert len(merged) == 2


if __name__ == "__main__":
    test_disjoint_run_ids_are_summed()
    test_same_run_id_is_not_double_counted()
    test_same_run_id_keeps_larger_if_only_that_run()
    test_partial_overlap_is_skipped()
    test_different_conditions_not_merged()
    print("ok")
