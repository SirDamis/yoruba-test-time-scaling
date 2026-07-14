"""Tests for E4 small+TTC vs large greedy comparison."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.e4_compare import (
    E4ComparisonConfig,
    build_e4_comparison,
    condition_metrics_from_dict,
    method_matches,
    save_e4_report,
)
from ttcs_yoruba.metrics import ConditionMetrics


def _m(
    *,
    dataset: str,
    model: str,
    n: int,
    accuracy_correct: int,
    total: int = 10,
    tokens: int = 100,
    latency: float = 10.0,
    method: str | None = None,
) -> ConditionMetrics:
    return ConditionMetrics(
        dataset=dataset,
        model=model,
        model_size_label=model.split("-")[-1].upper(),
        method=method or f"english_cot_ttc_n{n}",
        prompt_style="english_cot",
        reasoning_language="en",
        selection="first" if n == 1 else "majority_vote",
        n=n,
        total_examples=total,
        select_correct=accuracy_correct,
        pass_at_n_correct=accuracy_correct,
        total_candidates=total * n,
        total_tokens=tokens * total,
        total_latency_s=latency * total,
        total_estimated_cost=0.0,
        run_ids=["t"],
    )


def test_small_matches_large_at_sufficient_n() -> None:
    metrics = [
        _m(dataset="afrimgsm", model="qwen3-4b", n=1, accuracy_correct=3, tokens=20),
        _m(dataset="afrimgsm", model="qwen3-4b", n=4, accuracy_correct=6, tokens=80),
        _m(dataset="afrimgsm", model="qwen3-4b", n=16, accuracy_correct=8, tokens=300),
        _m(dataset="afrimgsm", model="qwen3-32b", n=1, accuracy_correct=7, tokens=40),
    ]
    report = build_e4_comparison(
        metrics,
        config=E4ComparisonConfig(small_model="qwen3-4b", large_model="qwen3-32b", large_n=1),
    )
    block = report["datasets"][0]
    assert block["large_baseline"]["accuracy"] == 0.7
    assert block["min_n_to_match_large"] == 16
    by_n = {c["small_n"]: c for c in block["comparisons"]}
    assert by_n[1]["matches_or_beats_large"] is False
    assert by_n[4]["matches_or_beats_large"] is False
    assert by_n[16]["matches_or_beats_large"] is True
    assert abs(by_n[16]["accuracy_delta"] - 0.1) < 1e-9
    assert report["headline"]["datasets_matched"][0]["min_n"] == 16


def test_missing_large_baseline() -> None:
    metrics = [
        _m(dataset="afrimgsm", model="qwen3-4b", n=4, accuracy_correct=5),
    ]
    report = build_e4_comparison(
        metrics,
        config=E4ComparisonConfig(small_model="qwen3-4b", large_model="qwen3-32b"),
    )
    assert report["datasets"][0]["large_baseline"] is None
    assert report["headline"]["datasets_missing_large_baseline"] == ["afrimgsm"]


def test_efficiency_fields_present() -> None:
    metrics = [
        _m(dataset="afrimgsm", model="qwen3-4b", n=4, accuracy_correct=5, tokens=100),
        _m(dataset="afrimgsm", model="qwen3-32b", n=1, accuracy_correct=6, tokens=50),
    ]
    report = build_e4_comparison(metrics)
    small = report["paper_table"][0]["small"]
    assert small["tokens_per_correct"] == (100 * 10) / 5
    assert small["mean_tokens_per_example"] == 100.0
    assert small["accuracy_per_1k_tokens"] is not None


def test_save_e4_report_writes_files() -> None:
    metrics = [
        _m(dataset="afrimgsm", model="qwen3-4b", n=1, accuracy_correct=4),
        _m(dataset="afrimgsm", model="qwen3-4b", n=8, accuracy_correct=7),
        _m(dataset="afrimgsm", model="qwen3-32b", n=1, accuracy_correct=6),
    ]
    report = build_e4_comparison(metrics)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = save_e4_report(report, out, make_plots=False)
        assert Path(paths["json"]).exists()
        assert Path(paths["csv"]).exists()
        payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        assert payload["num_datasets"] == 1
        assert payload["headline"]["datasets_matched"][0]["min_n"] == 8


def test_condition_metrics_from_dict_roundtrip() -> None:
    original = _m(dataset="x", model="qwen3-4b", n=4, accuracy_correct=3)
    restored = condition_metrics_from_dict(original.to_dict())
    assert restored.accuracy == original.accuracy
    assert restored.n == 4
    assert restored.model == "qwen3-4b"


def test_method_matches_prefix_and_exact() -> None:
    assert method_matches("english_cot_ttc_n4", "english_cot_ttc")
    assert method_matches("english_cot_ttc_n4", "english_cot_ttc*")
    assert method_matches("english_cot_ttc_n4", "english_cot_ttc_n4")
    assert not method_matches("yoruba_cot_n4", "english_cot_ttc")
    assert method_matches("anything", None)


def test_method_filter_disambiguates_same_model_n() -> None:
    """Two methods at same model+N: filter selects the intended family."""
    metrics = [
        _m(
            dataset="afrimgsm",
            model="qwen3-4b",
            n=4,
            accuracy_correct=2,
            method="yoruba_cot_ttc_n4",
        ),
        _m(
            dataset="afrimgsm",
            model="qwen3-4b",
            n=4,
            accuracy_correct=8,
            method="english_cot_ttc_n4",
        ),
        _m(
            dataset="afrimgsm",
            model="qwen3-32b",
            n=1,
            accuracy_correct=5,
            method="english_cot_ttc_n1",
        ),
    ]
    report = build_e4_comparison(
        metrics,
        config=E4ComparisonConfig(
            small_model="qwen3-4b",
            large_model="qwen3-32b",
            small_method="english_cot_ttc",
            large_method="english_cot_ttc",
        ),
    )
    small = report["paper_table"][0]["small"]
    assert small["method"] == "english_cot_ttc_n4"
    assert small["accuracy"] == 0.8

    # Wrong filter should find nothing for small curve.
    empty = build_e4_comparison(
        metrics,
        config=E4ComparisonConfig(
            small_model="qwen3-4b",
            large_model="qwen3-32b",
            small_method="translate_pivot",
            large_method="english_cot_ttc",
        ),
    )
    assert empty["paper_table"] == []


if __name__ == "__main__":
    test_small_matches_large_at_sufficient_n()
    test_missing_large_baseline()
    test_efficiency_fields_present()
    test_save_e4_report_writes_files()
    test_condition_metrics_from_dict_roundtrip()
    test_method_matches_prefix_and_exact()
    test_method_filter_disambiguates_same_model_n()
    print("ok")
