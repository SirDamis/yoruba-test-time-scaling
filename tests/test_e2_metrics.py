"""Unit tests for E2 TTC metrics aggregation and config expansion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import load_inference_run_config
from ttcs_yoruba.metrics import aggregate_run_dir, write_metrics_csv


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_e2_config_expands_pool_n1_plus_greedy_reference() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling_vllm.json")
    assert [m.name for m in cfg.models] == [
        "qwen3.5-4b",
        "qwen3.5-9b",
        "qwen3-14b",
        "qwen3-32b",
    ]
    methods = {m.name: m for m in cfg.methods}
    assert set(methods) == {
        "translate_pivot_ttc_n1",
        "translate_pivot_ttc_n2",
        "translate_pivot_ttc_n3",
        "translate_pivot_ttc_n4",
        "translate_pivot_ttc_n8",
        "translate_pivot_ttc_n16",
        "translate_pivot_ttc_n32",
        "translate_pivot_ttc_n64",
        "translate_pivot_greedy",
    }
    # N=1 is the first draw of the same stochastic pool (not a separate greedy decode).
    n1 = methods["translate_pivot_ttc_n1"]
    assert n1.n == 1
    assert n1.temperature == 0.7
    assert n1.top_p == 0.95
    assert n1.selection == "majority_vote"
    assert n1.prompt_style == "translate_pivot"
    assert n1.reasoning_language == "en_pivot"
    assert n1.nested_group_id == "translate_pivot_ttc"
    # Greedy N=1 is kept as a separate, labelled no-TTC reference outside the group.
    greedy = methods["translate_pivot_greedy"]
    assert greedy.n == 1
    assert greedy.temperature == 0.0
    assert greedy.top_p is None
    assert greedy.selection == "first"
    assert greedy.prompt_style == "translate_pivot"
    assert greedy.nested_group_id is None
    n4 = methods["translate_pivot_ttc_n4"]
    assert n4.n == 4
    assert n4.temperature == 0.7
    assert n4.top_p == 0.95
    assert n4.selection == "majority_vote"


def test_resolve_method_selection_specs_and_skip_greedy() -> None:
    from ttcs_yoruba.inference import is_greedy_reference, resolve_method_selection

    cfg = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling_vllm.json")
    methods = list(cfg.methods)

    # No filter -> None (all methods).
    assert resolve_method_selection(methods, None) is None

    # Prompt-style spec selects the nested slices and the greedy reference.
    by_style = resolve_method_selection(methods, {"translate_pivot"})
    assert "translate_pivot_ttc_n64" in by_style
    assert "translate_pivot_greedy" in by_style

    # Family prefix selects the slices only (not the greedy reference).
    by_family = resolve_method_selection(methods, {"translate_pivot_ttc"})
    assert "translate_pivot_ttc_n4" in by_family
    assert "translate_pivot_greedy" not in by_family

    # --skip-greedy drops the reference even without an explicit --methods.
    no_greedy = resolve_method_selection(methods, None, skip_greedy=True)
    assert "translate_pivot_greedy" not in no_greedy
    assert "translate_pivot_ttc_n1" in no_greedy

    assert is_greedy_reference(next(m for m in methods if m.name == "translate_pivot_greedy"))

    try:
        resolve_method_selection(methods, {"does_not_exist"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown --methods spec")


def test_aggregate_run_dir_pass_select_and_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_e2"
    run_dir.mkdir()
    candidates = []
    for sample_index, answer in enumerate(["11", "10", "11", "12"]):
        candidates.append(
            {
                "example_id": "ex1",
                "model": "qwen3-4b",
                "model_size_label": "4B",
                "method": "english_cot_ttc_n4",
                "prompt_style": "english_cot",
                "reasoning_language": "en",
                "selection": "majority_vote",
                "n": 4,
                "sample_index": sample_index,
                "extracted_answer": answer,
                "token_count": 10 + sample_index,
                "prompt_token_count": 20,
                "latency_s": 0.5,
                "estimated_cost": 0.0,
                "metadata": {
                    "dataset": "afrimgsm",
                    "gold_answer": "11",
                    "answer_type": "number",
                },
            }
        )
    # Second example: no correct candidates.
    for sample_index in range(4):
        candidates.append(
            {
                "example_id": "ex2",
                "model": "qwen3-4b",
                "model_size_label": "4B",
                "method": "english_cot_ttc_n4",
                "prompt_style": "english_cot",
                "reasoning_language": "en",
                "selection": "majority_vote",
                "n": 4,
                "sample_index": sample_index,
                "extracted_answer": "0",
                "token_count": 5,
                "prompt_token_count": 20,
                "latency_s": 0.25,
                "estimated_cost": 0.0,
                "metadata": {
                    "dataset": "afrimgsm",
                    "gold_answer": "11",
                    "answer_type": "number",
                },
            }
        )
    _write_jsonl(run_dir / "candidates.jsonl", candidates)
    selections = [
        {
            "example_id": "ex1",
            "dataset": "afrimgsm",
            "model": "qwen3-4b",
            "model_size_label": "4B",
            "method": "english_cot_ttc_n4",
            "prompt_style": "english_cot",
            "reasoning_language": "en",
            "selection": "majority_vote",
            "n": 4,
            "selected_answer": "11",
            "is_correct": True,
        },
        {
            "example_id": "ex2",
            "dataset": "afrimgsm",
            "model": "qwen3-4b",
            "model_size_label": "4B",
            "method": "english_cot_ttc_n4",
            "prompt_style": "english_cot",
            "reasoning_language": "en",
            "selection": "majority_vote",
            "n": 4,
            "selected_answer": "0",
            "is_correct": False,
        },
    ]
    _write_jsonl(run_dir / "selections.jsonl", selections)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_e2"}), encoding="utf-8"
    )

    metrics = aggregate_run_dir(run_dir)
    assert len(metrics) == 1
    m = metrics[0]
    assert m.total_examples == 2
    assert m.pass_at_n_correct == 1  # only ex1 has a correct sample
    assert m.select_correct == 1
    assert m.accuracy == 0.5
    assert m.pass_at_n_rate == 0.5
    assert m.total_tokens == (10 + 11 + 12 + 13) + (5 * 4)
    assert m.total_prompt_tokens == 20 * 8
    assert m.mean_prompt_tokens_per_example == 80
    assert m.mean_tokens_per_example == m.total_tokens / 2

    csv_path = tmp_path / "metrics.csv"
    write_metrics_csv(csv_path, metrics)
    assert csv_path.exists()
    assert "accuracy" in csv_path.read_text(encoding="utf-8")


def test_aggregate_run_dir_counts_empty_extractions(tmp_path: Path) -> None:
    """Empty extracted answers are tallied separately from wrong answers."""
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()
    candidates = [
        {
            "example_id": "ex1",
            "model": "qwen3-4b",
            "model_size_label": "4B",
            "method": "english_cot_ttc_n2",
            "prompt_style": "english_cot",
            "reasoning_language": "en",
            "selection": "majority_vote",
            "n": 2,
            "sample_index": sample_index,
            "extracted_answer": "" if sample_index == 0 else "11",
            "token_count": 5,
            "latency_s": 0.1,
            "estimated_cost": 0.0,
            "metadata": {"dataset": "afrimgsm", "gold_answer": "11", "answer_type": "number"},
        }
        for sample_index in range(2)
    ]
    _write_jsonl(run_dir / "candidates.jsonl", candidates)
    _write_jsonl(run_dir / "selections.jsonl", [])
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_empty"}), encoding="utf-8")

    m = aggregate_run_dir(run_dir)[0]
    assert m.total_candidates == 2
    assert m.empty_extraction_candidates == 1
    assert m.empty_extraction_rate == 0.5
    assert m.to_dict()["empty_extraction_candidates"] == 1


def test_bon_uses_e1_prompt_styles() -> None:
    from ttcs_yoruba.examples import InferenceExample
    from ttcs_yoruba.prompting import render_prompt

    ex = InferenceExample(
        id="x",
        task="math",
        question="Ibeere?",
        choices=None,
        gold_answer="1",
        answer_type="number",
        source_dataset="afrimgsm",
    )
    yo = render_prompt(ex, "yoruba_cot")
    en = render_prompt(ex, "english_cot")
    pivot = render_prompt(ex, "translate_pivot")
    assert "Yorùbá" in yo.system or "Yoruba" in yo.user or "Yorùbá" in yo.user
    assert "English" in en.system or "English" in en.user
    assert "translate" in pivot.user.lower() or "Translate" in pivot.user


if __name__ == "__main__":
    from pathlib import Path as P
    import tempfile

    test_e2_config_expands_pool_n1_plus_greedy_reference()
    test_resolve_method_selection_specs_and_skip_greedy()
    with tempfile.TemporaryDirectory() as td:
        test_aggregate_run_dir_pass_select_and_tokens(P(td))
    with tempfile.TemporaryDirectory() as td:
        test_aggregate_run_dir_counts_empty_extractions(P(td))
    test_bon_uses_e1_prompt_styles()
    print("ok")
