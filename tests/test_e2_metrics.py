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


def test_e2_config_expands_n_sweep_with_greedy_n1(tmp_path: Path | None = None) -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling.json")
    assert [m.name for m in cfg.models] == ["qwen3-4b", "qwen3-14b", "qwen3-32b"]
    methods = {m.name: m for m in cfg.methods}
    assert set(methods) == {
        "english_cot_ttc_n1",
        "english_cot_ttc_n4",
        "english_cot_ttc_n8",
        "english_cot_ttc_n16",
        "english_cot_ttc_n32",
        "english_cot_ttc_n64",
    }
    n1 = methods["english_cot_ttc_n1"]
    assert n1.n == 1
    assert n1.temperature == 0.0
    assert n1.selection == "first"
    assert n1.prompt_style == "english_cot"
    n4 = methods["english_cot_ttc_n4"]
    assert n4.n == 4
    assert n4.temperature == 0.7
    assert n4.selection == "majority_vote"


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
    assert m.mean_tokens_per_example == m.total_tokens / 2

    csv_path = tmp_path / "metrics.csv"
    write_metrics_csv(csv_path, metrics)
    assert csv_path.exists()
    assert "accuracy" in csv_path.read_text(encoding="utf-8")


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

    test_e2_config_expands_n_sweep_with_greedy_n1()
    with tempfile.TemporaryDirectory() as td:
        test_aggregate_run_dir_pass_select_and_tokens(P(td))
    test_bon_uses_e1_prompt_styles()
    print("ok")
