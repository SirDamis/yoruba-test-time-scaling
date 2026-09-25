from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PRM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PRM_ROOT / "src"))

from ttcs_yoruba.selection import select_candidate

from prm_yoruba.config import BuildConfig
from prm_yoruba.data import build_dataset, validate_prm800k_row, write_built_dataset
from prm_yoruba.grading import is_correct
from prm_yoruba.model import aggregate_scores, resolve_prm_tokens
from prm_yoruba.steps import (
    QWEN_STEP_TAG,
    render_scoring_input,
    render_tagged_process,
    split_into_steps,
)


class _FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str) -> list[int]:
        return list(self.mapping[text])


def _prm800k_config(tmp_path: Path, path: Path, languages: list[str] | None = None) -> BuildConfig:
    source = {
        "type": "prm800k",
        "enabled": True,
        "languages": languages or ["yor"],
        "paths": {"yor": [str(path)]},
    }
    return BuildConfig.from_dict(
        {"output_dir": str(tmp_path / "out"), "seed": 0, "sources": [source]}
    )


def _write_prm800k(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_split_into_steps_drops_final_answer():
    steps = split_into_steps("First.\n\nSecond.\n\nFinal answer: 7")
    assert steps == ["First.", "Second."]
    tagged = render_tagged_process(["a", "b"], QWEN_STEP_TAG)
    assert tagged.count(QWEN_STEP_TAG) == 2


def test_split_into_steps_uses_newline_delimiter():
    text = "Question:\nKí ni ìdáhùn?\n\nReasoning:\nStep one.\nStep two.\n\nFinal answer: 4"
    assert split_into_steps(text) == ["Step one.", "Step two."]


def test_split_into_steps_strips_bullets_and_step_labels():
    text = "Reasoning:\n- Àkọ́kọ́, a ṣe é.\n- Èkejì, a ṣe é."
    assert split_into_steps(text) == ["Àkọ́kọ́, a ṣe é.", "Èkejì, a ṣe é."]
    assert split_into_steps("Reasoning:\n**Step 1:** Find x.\nStep 2: Use x.") == [
        "Find x.",
        "Use x.",
    ]


def test_split_into_steps_bold_final_answer():
    text = "Reasoning:\n- A.\n- B.\n\n**Final answer:** 4"
    assert split_into_steps(text) == ["A.", "B."]


def test_split_into_steps_drops_echoed_question_without_reasoning_header():
    text = "Question: A farmer has 2 bags.\n\nFirst step.\nSecond step.\n\nFinal answer: 1"
    assert split_into_steps(text) == ["First step.", "Second step."]


def test_split_into_steps_direct_answer_has_no_steps():
    assert split_into_steps("Final answer: 7") == []


def test_split_into_steps_single_line_sentence_fallback():
    text = "Reasoning:\nFirst, do a. Then do b.\n\nFinal answer: 2"
    assert split_into_steps(text) == ["First, do a.", "Then do b."]


def test_build_prm800k_rows_from_local_file(tmp_path):
    path = _write_prm800k(
        tmp_path / "prm800k_yor.json",
        [
            {"question": "Q1", "process": f"s1{QWEN_STEP_TAG}s2{QWEN_STEP_TAG}", "label": ["+", "-"]},
            {"question": "Q2", "process": "no step tag here", "label": ["+", "-"]},
        ],
    )
    rows = build_dataset(_prm800k_config(tmp_path, path))
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "prm800k"
    assert row["language"] == "yor"
    assert row["problem_source"] == "supplied"
    assert row["label"] == ["+", "-"]
    assert row["process"].count(QWEN_STEP_TAG) == row["n_steps"] == 2


def test_validate_prm800k_adds_missing_trailing_tag():
    valid = validate_prm800k_row(
        {"question": "Q", "process": f"a{QWEN_STEP_TAG}b", "label": ["+", "-"]},
        step_tag=QWEN_STEP_TAG,
    )
    assert valid is not None
    assert valid["process"].count(QWEN_STEP_TAG) == 2
    assert valid["process"].endswith(QWEN_STEP_TAG)


def test_write_built_dataset_counts(tmp_path):
    path = _write_prm800k(
        tmp_path / "prm800k_yor.json",
        [{"question": "Q1", "process": f"s1{QWEN_STEP_TAG}", "label": ["+"]}],
    )
    rows = build_dataset(_prm800k_config(tmp_path, path))
    manifest = write_built_dataset(rows, [], rows, tmp_path / "out")
    counts = manifest["counts"]["all"]
    assert counts["total"] == 1
    assert counts["by_source"] == {"prm800k": 1}
    assert counts["by_language"] == {"yor": 1}
    assert "by_outcome" not in counts
    assert (tmp_path / "out" / "train.jsonl").exists()


def test_prm800k_unreleased_language_requires_path(tmp_path):
    source = {"type": "prm800k", "enabled": True, "languages": ["yor"]}
    config = BuildConfig.from_dict(
        {"output_dir": str(tmp_path / "out"), "seed": 0, "sources": [source]}
    )
    with pytest.raises(ValueError, match="not in the released dataset"):
        build_dataset(config)


def test_prm800k_missing_file_raises(tmp_path):
    source = {
        "type": "prm800k",
        "enabled": True,
        "languages": ["yor"],
        "paths": {"yor": [str(tmp_path / "does_not_exist.json")]},
    }
    config = BuildConfig.from_dict(
        {"output_dir": str(tmp_path / "out"), "sources": [source]}
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        build_dataset(config)


def test_runs_source_is_rejected(tmp_path):
    source = {"type": "runs", "enabled": True, "runs_dir": "runs"}
    config = BuildConfig.from_dict(
        {"output_dir": str(tmp_path / "out"), "sources": [source]}
    )
    with pytest.raises(ValueError, match="never on outcome-derived labels"):
        build_dataset(config)


def test_resolve_prm_tokens_qwen_and_mistral():
    qwen = _FakeTokenizer({" + -": [100, 101], " \n\n\n\n\n": [200, 201]})
    tokens = resolve_prm_tokens(qwen, "Qwen/Qwen2.5-Math-PRM-7B")
    assert tokens.candidate_tokens == [100, 101]
    assert tokens.step_tag_id == 201

    mistral = _FakeTokenizer({"+ -": [7, 8], "ки": [9]})
    tokens = resolve_prm_tokens(mistral, "mistral-7b-sft")
    assert tokens.candidate_tokens == [7, 8]
    assert tokens.step_tag_id == 9


def test_resolve_prm_tokens_drops_leading_token():
    tokenizer = _FakeTokenizer({" + -": [0, 100, 101], " \n\n\n\n\n": [201]})
    tokens = resolve_prm_tokens(tokenizer, "qwen3.5-9b")
    assert tokens.candidate_tokens == [100, 101]
    assert tokens.step_tag_id == 201


def test_aggregate_scores():
    assert aggregate_scores([0.1, 0.9, 0.5], "last") == pytest.approx(0.5)
    assert aggregate_scores([0.1, 0.9, 0.5], "max") == pytest.approx(0.9)
    assert aggregate_scores([0.1, 0.9, 0.5], "min") == pytest.approx(0.1)
    assert aggregate_scores([0.1, 0.9, 0.5], "mean") == pytest.approx(0.5)
    assert aggregate_scores([], "max") == 0.0
    with pytest.raises(ValueError):
        aggregate_scores([0.5], "nope")


def test_render_scoring_input():
    text = render_scoring_input("Q", ["a", "b"], QWEN_STEP_TAG)
    assert text == f"Q a{QWEN_STEP_TAG} b{QWEN_STEP_TAG}"
    assert text.count(QWEN_STEP_TAG) == 2


def test_grading_number_and_choice():
    assert is_correct("3", "3", "number")
    assert is_correct("3.0", "3", "number")
    assert not is_correct("4", "3", "number")
    assert is_correct("B", "B", "choice")
    assert not is_correct("A", "B", "choice")


def test_select_candidate_prm_strategy():
    candidates = [
        {"sample_index": 0, "extracted_answer": "3", "prm_score": 0.1},
        {"sample_index": 1, "extracted_answer": "4", "prm_score": 0.9},
    ]
    result = select_candidate(candidates, "prm")
    assert result.selected_sample_index == 1
    assert result.selected_answer == "4"
    assert result.metadata["prm_score"] == pytest.approx(0.9)


def test_select_candidate_prm_requires_scores():
    candidates = [{"sample_index": 0, "extracted_answer": "3"}]
    with pytest.raises(ValueError, match="prm_score"):
        select_candidate(candidates, "prm")


def test_select_candidate_defaults_unchanged():
    candidates = [
        {"sample_index": 0, "extracted_answer": "3"},
        {"sample_index": 1, "extracted_answer": "3"},
        {"sample_index": 2, "extracted_answer": "4"},
    ]
    assert select_candidate(candidates, "majority_vote").selected_answer == "3"
    assert select_candidate(candidates, "first").selected_sample_index == 0
