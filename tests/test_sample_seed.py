"""Tests for example-aware sampling seeds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.inference import derive_sample_seed


def test_none_base_seed_returns_none() -> None:
    assert derive_sample_seed(None, "ex1", 0) is None


def test_same_example_sample_is_stable() -> None:
    a = derive_sample_seed(20260708, "afrimgsm_all_000001", 0)
    b = derive_sample_seed(20260708, "afrimgsm_all_000001", 0)
    assert a == b
    assert isinstance(a, int)
    assert 0 <= a < 2**31


def test_different_examples_differ_at_same_sample_index() -> None:
    s1 = derive_sample_seed(20260708, "example_a", 0)
    s2 = derive_sample_seed(20260708, "example_b", 0)
    assert s1 != s2


def test_different_sample_indices_differ() -> None:
    s0 = derive_sample_seed(20260708, "example_a", 0)
    s1 = derive_sample_seed(20260708, "example_a", 1)
    assert s0 != s1


def test_not_equal_to_naive_base_plus_index() -> None:
    """Regression: old scheme was base_seed + sample_index for every example."""
    base = 20260708
    for example_id in ("ex1", "ex2", "ex3"):
        for sample_index in range(4):
            derived = derive_sample_seed(base, example_id, sample_index)
            assert derived != base + sample_index or example_id == "ex1"
    # Stronger: sample 0 must not be identical across examples (old bug).
    seeds = {derive_sample_seed(base, f"ex{i}", 0) for i in range(20)}
    assert len(seeds) == 20


if __name__ == "__main__":
    test_none_base_seed_returns_none()
    test_same_example_sample_is_stable()
    test_different_examples_differ_at_same_sample_index()
    test_different_sample_indices_differ()
    test_not_equal_to_naive_base_plus_index()
    print("ok")
