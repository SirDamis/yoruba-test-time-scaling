"""Tests for E3 selection strategies and offline re-selection."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.reselection import (
    group_candidates,
    reselect_groups,
    summarize_reselection,
)
from ttcs_yoruba.selection import SUPPORTED_SELECTIONS, select_candidate


def test_supported_selections_are_local_only() -> None:
    assert SUPPORTED_SELECTIONS == {"first", "majority_vote"}
    try:
        select_candidate([{"sample_index": 0, "extracted_answer": "1"}], "llm_judge")
        raise AssertionError("llm_judge should be rejected")
    except ValueError as exc:
        assert "llm_judge" in str(exc) or "Unsupported" in str(exc)


def test_majority_vote_picks_most_common() -> None:
    candidates = [
        {"sample_index": 0, "extracted_answer": "10"},
        {"sample_index": 1, "extracted_answer": "11"},
        {"sample_index": 2, "extracted_answer": "11"},
        {"sample_index": 3, "extracted_answer": "12"},
    ]
    result = select_candidate(candidates, "majority_vote")
    assert result.selected_answer == "11"
    assert result.selected_sample_index == 1


def test_first_selection() -> None:
    candidates = [
        {"sample_index": 0, "extracted_answer": "10"},
        {"sample_index": 1, "extracted_answer": "11"},
    ]
    result = select_candidate(candidates, "first")
    assert result.selected_sample_index == 0
    assert result.selected_answer == "10"


def test_offline_reselection_report() -> None:
    candidates = []
    for sample_index, answer in enumerate(["11", "10", "11", "12"]):
        candidates.append(
            {
                "run_id": "r1",
                "example_id": "ex1",
                "task": "math",
                "source_dataset": "afrimgsm",
                "model": "qwen3-4b",
                "model_size_label": "4B",
                "method": "english_cot_ttc_n4",
                "prompt_style": "english_cot",
                "reasoning_language": "en",
                "sample_index": sample_index,
                "n": 4,
                "response": f"Final answer: {answer}",
                "extracted_answer": answer,
                "metadata": {
                    "dataset": "afrimgsm",
                    "question": "Melo?",
                    "gold_answer": "11",
                    "answer_type": "number",
                    "choices": None,
                },
            }
        )
    # Second example: no correct candidates.
    for sample_index in range(4):
        candidates.append(
            {
                "run_id": "r1",
                "example_id": "ex2",
                "task": "math",
                "source_dataset": "afrimgsm",
                "model": "qwen3-4b",
                "model_size_label": "4B",
                "method": "english_cot_ttc_n4",
                "prompt_style": "english_cot",
                "reasoning_language": "en",
                "sample_index": sample_index,
                "n": 4,
                "response": "Final answer: 0",
                "extracted_answer": "0",
                "metadata": {
                    "dataset": "afrimgsm",
                    "question": "Melo?",
                    "gold_answer": "11",
                    "answer_type": "number",
                    "choices": None,
                },
            }
        )

    groups = group_candidates(candidates)
    assert len(groups) == 2
    selections = reselect_groups(groups, strategies=["first", "majority_vote"], run_id="r1")
    assert len(selections["majority_vote"]) == 2
    report = summarize_reselection(groups, selections)
    assert len(report) == 1
    row = report[0]
    assert row["pass_at_n_rate"] == 0.5  # only ex1 has a correct sample
    assert row["selections"]["majority_vote"]["select_at_n_rate"] == 0.5
    # first picks sample 0 for both: ex1 -> 11 correct, ex2 -> 0 wrong => 0.5
    assert row["selections"]["first"]["select_at_n_rate"] == 0.5


def test_reselect_run_dir_writes_artifacts() -> None:
    from ttcs_yoruba.reselection import reselect_run_dir

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "run_x"
        out_dir = root / "out"
        run_dir.mkdir()
        rows = [
            {
                "run_id": "run_x",
                "example_id": "ex1",
                "task": "math",
                "source_dataset": "afrimgsm",
                "model": "qwen3-4b",
                "model_size_label": "4B",
                "method": "english_cot_ttc_n2",
                "prompt_style": "english_cot",
                "reasoning_language": "en",
                "sample_index": 0,
                "n": 2,
                "response": "Final answer: 11",
                "extracted_answer": "11",
                "prompt": "System:\ns\n\nUser:\nQuestion:\nMelo?\n",
                "metadata": {
                    "dataset": "afrimgsm",
                    "question": "Melo?",
                    "gold_answer": "11",
                    "answer_type": "number",
                },
            },
            {
                "run_id": "run_x",
                "example_id": "ex1",
                "task": "math",
                "source_dataset": "afrimgsm",
                "model": "qwen3-4b",
                "model_size_label": "4B",
                "method": "english_cot_ttc_n2",
                "prompt_style": "english_cot",
                "reasoning_language": "en",
                "sample_index": 1,
                "n": 2,
                "response": "Final answer: 10",
                "extracted_answer": "10",
                "prompt": "System:\ns\n\nUser:\nQuestion:\nMelo?\n",
                "metadata": {
                    "dataset": "afrimgsm",
                    "question": "Melo?",
                    "gold_answer": "11",
                    "answer_type": "number",
                },
            },
        ]
        (run_dir / "candidates.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_x"}), encoding="utf-8")

        report = reselect_run_dir(
            run_dir,
            strategies=["first", "majority_vote"],
            output_dir=out_dir,
        )
        assert (out_dir / "selections_majority_vote.jsonl").exists()
        assert (out_dir / "selections_first.jsonl").exists()
        assert (out_dir / "e3_report.json").exists()
        assert report["conditions"][0]["pass_at_n_rate"] == 1.0


if __name__ == "__main__":
    test_supported_selections_are_local_only()
    test_majority_vote_picks_most_common()
    test_first_selection()
    test_offline_reselection_report()
    test_reselect_run_dir_writes_artifacts()
    print("ok")
