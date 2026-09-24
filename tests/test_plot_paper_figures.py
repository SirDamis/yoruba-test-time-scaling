"""Tests for the paper-figure helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_paper_figures", ROOT / "scripts" / "plot_paper_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_interval() -> None:
    module = _load_module()
    assert module.wilson(0, 0) == (0.0, 0.0)
    lo, hi = module.wilson(50, 100)
    assert 0.40 < lo < 0.50
    assert 0.50 < hi < 0.60
    assert lo < 0.5 < hi


def test_mcnemar_balanced_is_not_significant() -> None:
    module = _load_module()
    A = {"a": True, "b": False}
    B = {"a": False, "b": True}
    b, c, p = module.mcnemar(A, B)
    assert b == 1 and c == 1
    assert abs(p - 1.0) < 1e-9


def test_mcnemar_one_directional_is_significant() -> None:
    module = _load_module()
    A = {str(i): True for i in range(10)}
    B = {str(i): False for i in range(10)}
    b, c, p = module.mcnemar(A, B)
    assert (b, c) == (10, 0)
    assert p < 0.01


def test_suffix_and_family() -> None:
    module = _load_module()
    assert module.suffix("afrimgsm_yor") == "yor"
    assert module.family("afrimgsm_yor") == "afrimgsm"


def test_sample_diversity_matches_subset_answers() -> None:
    module = _load_module()
    pools = [[{"extracted_answer": "5"}, {"extracted_answer": "5"}, {"extracted_answer": "6"}]]
    distinct, degenerate = module._sample_diversity(pools, 3, num_subsets=10, seed=0)
    assert distinct == 2.0
    assert degenerate == 0.0


if __name__ == "__main__":
    test_wilson_interval()
    test_mcnemar_balanced_is_not_significant()
    test_mcnemar_one_directional_is_significant()
    test_suffix_and_family()
    test_sample_diversity_matches_subset_answers()
    print("ok")
