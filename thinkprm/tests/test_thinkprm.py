from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

THINKPRM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = THINKPRM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "prm" / "src"))
sys.path.insert(0, str(THINKPRM_ROOT / "src"))

from ttcs_yoruba.selection import select_candidate

from thinkprm_yoruba.config import CollectConfig, ScoreConfig, TrainConfig
from thinkprm_yoruba.data import (
    build_training_example,
    labels_until_first_error,
    load_prm_rows,
    split_rows,
    write_built_dataset,
)
from thinkprm_yoruba.model import ThinkPrmScorer, aggregate_step_scores
from thinkprm_yoruba.parse import (
    extract_boxed_decisions,
    extract_decision,
    extract_step_labels,
    pad_step_labels,
    strip_think,
    truncate_after_last_decision,
)
from thinkprm_yoruba.prompts import build_assistant_target, build_verification_user_message
from thinkprm_yoruba.select import attach_thinkprm_scores
from thinkprm_yoruba.steps import (
    parse_numbered_solution,
    render_numbered_solution,
    strip_step_prefix,
)


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text)))


class _ThinkTokenizer:
    """Mimics the ThinkPRM base: generation prompt opens a <think> block."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:<think>\n"
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [151648] if text == "<think>" else [ord(char) for char in text]

    def decode(self, ids) -> str:
        if isinstance(ids, int):
            return "<think>" if ids == 151648 else chr(ids)
        if list(ids) == [151648]:
            return "<think>"
        return "".join(chr(int(i)) for i in ids)


class _FakeScorer:
    def score_batch(self, questions, steps_batch):
        results = []
        for index, steps in enumerate(steps_batch):
            results.append(
                {
                    "score": 0.1 * index,
                    "prefix_score": 0.1 * index,
                    "prefix_scores": [0.1 * index],
                    "step_labels": [1] * len(steps),
                    "step_scores": [1.0] * len(steps),
                    "n_verifications": 1,
                    "outputs": ["verification"],
                    "n_steps": len(steps),
                    "truncated": False,
                }
            )
        return results


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_render_numbered_solution_and_parse_roundtrip():
    steps = ["Find x.", "Use x."]
    rendered = render_numbered_solution(steps)
    assert rendered == "Step 1: Find x.\nStep 2: Use x."
    assert parse_numbered_solution(rendered) == steps


def test_strip_step_prefix():
    assert strip_step_prefix("Step 2: Do the thing.") == "Do the thing."
    assert strip_step_prefix("**Step 3.** Done.") == "Done."
    assert strip_step_prefix("No prefix") == "No prefix"


def test_extract_boxed_decisions_handles_text_wrapper():
    text = (
        "Step 1 is fine \\boxed{correct}. "
        "Step 2 is wrong \\boxed{\\text{incorrect}}. "
        "Step 3 ok \\boxed{correct}."
    )
    assert extract_boxed_decisions(text) == ["correct", "incorrect", "correct"]
    assert extract_step_labels(text) == [1, 0, 1]


def test_extract_decision_variants():
    assert extract_decision("... \nIs the solution correct? Yes") == "Yes"
    assert extract_decision("Is the solution correct? **No**") == "No"
    assert extract_decision("Is the solution correct?") is None
    assert (
        extract_decision("Is the solution correct? No\nIs the solution correct? Yes")
        == "Yes"
    )


def test_truncate_after_last_decision_and_strip_think():
    text = "critique \\boxed{correct}\nmore \\boxed{incorrect}\ntrailing text"
    assert truncate_after_last_decision(text) == "critique \\boxed{correct}\nmore \\boxed{incorrect}"
    assert strip_think("<think>reasoning</think>final judgement") == "final judgement"
    assert strip_think("<think>only reasoning</think>") == "only reasoning"


def test_pad_step_labels():
    assert pad_step_labels([1, 0], 4) == [1, 0, 0, 0]
    assert pad_step_labels([1, 0, 1, 1], 2) == [1, 0]
    assert pad_step_labels([], 3) == [0, 0, 0]


def test_build_verification_user_message_contains_parts():
    message = build_verification_user_message("What is 2+2?", "Step 1: add.", "Verify it.")
    assert "[Math Problem]" in message
    assert "What is 2+2?" in message
    assert "Step 1: add." in message
    assert message.rstrip().endswith("Verify it.")


def test_build_verification_user_message_survives_boxed_instruction():
    from thinkprm_yoruba.prompts import COLLECTION_INSTRUCTION

    message = build_verification_user_message("Q", "S", COLLECTION_INSTRUCTION)
    assert "\\boxed{correct}" in message


def test_build_assistant_target_is_prefix_continuation():
    # Prefix already opened <think>, so the target closes it before the decision.
    target = build_assistant_target("step critique", "No", opens_think=True)
    assert target == "step critique\n</think>\nIs the solution correct? No"

    # An existing think wrapper is preserved (leading <think> dropped only).
    wrapped = build_assistant_target("<think>thinking</think>critique", "Yes", opens_think=True)
    assert wrapped == "thinking</think>critique\nIs the solution correct? Yes"

    # Without a think-opening prefix, neither tag is added.
    plain = build_assistant_target("critique", "Yes", opens_think=False)
    assert plain == "critique\nIs the solution correct? Yes"


def test_aggregate_step_scores():
    scores = [0.1, 0.9, 0.5]
    assert aggregate_step_scores(scores, "last") == pytest.approx(0.5)
    assert aggregate_step_scores(scores, "max") == pytest.approx(0.9)
    assert aggregate_step_scores(scores, "min") == pytest.approx(0.1)
    assert aggregate_step_scores(scores, "mean") == pytest.approx(0.5)
    assert aggregate_step_scores([], "max") == 0.0
    with pytest.raises(ValueError):
        aggregate_step_scores([0.5], "nope")


def test_labels_until_first_error():
    assert labels_until_first_error([1, 1, 0, 1]) == [1, 1, 0]
    assert labels_until_first_error([1, 1, 1]) == [1, 1, 1]
    assert labels_until_first_error([0, 1]) == [0]


def test_build_training_example_accepts_matching_chain():
    row = {"id": "r1", "question": "Q", "steps": ["a", "b", "c"], "labels": [1, 1, 0],
           "language": "yor", "task": "math"}
    text = "\\boxed{correct} \\boxed{correct} \\boxed{incorrect} extra"
    example = build_training_example(row, text)
    assert example is not None
    assert example["labels"] == [1, 1, 0]
    assert example["is_correct"] is False
    assert example["solution"].startswith("Step 1: a")
    assert example["cot"].endswith("\\boxed{incorrect}")


def test_build_training_example_rejects_mismatched_or_short_chain():
    row = {"id": "r1", "question": "Q", "steps": ["a", "b", "c"], "labels": [1, 1, 0]}
    assert build_training_example(row, "\\boxed{correct} \\boxed{incorrect} \\boxed{incorrect}") is None
    assert build_training_example(row, "\\boxed{correct} \\boxed{correct}") is None
    assert build_training_example(row, "no decisions here") is None


def test_build_training_example_respects_length_budget():
    row = {"id": "r1", "question": "Q", "steps": ["a", "b"], "labels": [1, 1]}
    text = "\\boxed{correct} \\boxed{correct} " + " ".join(["word"] * 50)
    assert build_training_example(row, text, max_cot_tokens=10) is None
    assert build_training_example(row, text, max_cot_tokens=1000) is not None


def test_load_prm_rows_prefers_steps_and_process(tmp_path):
    path = _write_jsonl(
        tmp_path / "all.jsonl",
        [
            {"id": "a", "question": "Q1", "steps": ["s1", "s2"], "label": ["+", "-"],
             "language": "yor", "task": "math"},
            {"id": "b", "question": "Q2", "process": "s1 \n\n\n\n\n s2 \n\n\n\n\n",
             "label": ["+", "+"], "language": "en"},
            {"id": "c", "question": "bad", "steps": ["x"], "label": ["?"]},
        ],
    )
    rows = load_prm_rows([str(path)])
    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["labels"] == [1, 0]
    assert rows[1]["steps"] == ["s1", "s2"]


def test_split_rows_groups_by_problem():
    rows = [
        {"problem": "p1", "id": "1"},
        {"problem": "p1", "id": "2"},
        {"problem": "p2", "id": "3"},
        {"problem": "p3", "id": "4"},
    ]
    train, evaluate = split_rows(rows, eval_fraction=0.5, seed=0)
    train_problems = {row["problem"] for row in train}
    eval_problems = {row["problem"] for row in evaluate}
    assert train_problems.isdisjoint(eval_problems)
    assert train_problems | eval_problems == {"p1", "p2", "p3"}


def test_write_built_dataset_counts(tmp_path):
    rows = [
        {"problem": "p1", "id": "1", "language": "yor", "n_steps": 2},
        {"problem": "p2", "id": "2", "language": "yor", "n_steps": 3},
    ]
    manifest = write_built_dataset(rows, [], rows, tmp_path / "out")
    counts = manifest["counts"]["all"]
    assert counts["total"] == 2
    assert counts["unique_problems"] == 2
    assert counts["total_step_labels"] == 5
    assert counts["by_language"] == {"yor": 2}
    assert (tmp_path / "out" / "train.jsonl").exists()


def test_attach_thinkprm_scores_and_selection_alias():
    rows = [
        {"sample_index": 0, "extracted_answer": "3", "response": "Reasoning:\nA.\n\nFinal answer: 3",
         "metadata": {"question": "Q"}},
        {"sample_index": 1, "extracted_answer": "4", "response": "Reasoning:\nB.\n\nFinal answer: 4",
         "metadata": {"question": "Q"}},
    ]
    enriched = attach_thinkprm_scores(rows, _FakeScorer(), max_steps=8, batch_size=2)
    assert enriched[0]["prm_score"] == pytest.approx(0.0)
    assert enriched[1]["prm_score"] == pytest.approx(0.1)
    assert "thinkprm_step_labels" in enriched[1]
    result = select_candidate(enriched, "thinkprm")
    assert result.selected_sample_index == 1
    assert result.selected_answer == "4"


def test_scoreconfig_defaults():
    config = ScoreConfig.from_dict({"name": "x", "model": "launch/ThinkPRM-1.5B"})
    assert config.aggregation == "prefix"
    assert config.n_verifications == 1
    assert config.trust_remote_code is False


def test_collectconfig_defaults():
    config = CollectConfig.from_dict({"prm_data": ["a.jsonl"]})
    assert config.prm_data == ["a.jsonl"]
    assert config.samples_per_example == 4


class _StubTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = "\n".join(m["content"] for m in messages)
        if add_generation_prompt:
            text += "\n<think>\n"
        return text

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(str(text))))}


def test_fit_prompt_returns_kept_steps_and_truncation_flag():
    scorer = ThinkPrmScorer(ScoreConfig.from_dict({"name": "x", "model": "m"}))
    scorer._tokenizer = _StubTokenizer()
    prompt, kept, truncated = scorer._fit_prompt("Q", ["a", "b", "c"])
    assert kept == ["a", "b", "c"]
    assert truncated is False
    assert "<think>" in prompt

    tiny = ThinkPrmScorer(
        ScoreConfig.from_dict({"name": "x", "model": "m", "max_input_tokens": 1})
    )
    tiny._tokenizer = _StubTokenizer()
    _, kept_steps, was_truncated = tiny._fit_prompt("Q", ["a", "b", "c"])
    assert was_truncated is True
    assert len(kept_steps) < 3


def test_score_batch_with_stubbed_generation():
    scorer = ThinkPrmScorer(ScoreConfig.from_dict({"name": "x", "model": "m"}))
    scorer._tokenizer = _StubTokenizer()
    scorer._model = object()  # skip load()
    scorer._yes_id = 1
    scorer._no_id = 2
    scorer._generate = lambda prompts: [  # type: ignore[method-assign]
        ["Step 1 ok \\boxed{correct}\n</think>"] for _ in prompts
    ]
    scorer._decision_probabilities = lambda contexts: [  # type: ignore[method-assign]
        0.75 for _ in contexts
    ]
    results = scorer.score_batch(["Q"], [["a", "b"]])
    assert results[0]["prefix_score"] == pytest.approx(0.75)
    assert results[0]["score"] == pytest.approx(0.75)
    assert results[0]["step_labels"] == [1, 0]
    assert results[0]["n_steps"] == 2


def test_score_batch_empty_steps_returns_zero():
    scorer = ThinkPrmScorer(ScoreConfig.from_dict({"name": "x", "model": "m"}))
    scorer._tokenizer = _StubTokenizer()
    scorer._model = object()
    scorer._yes_id = 1
    scorer._no_id = 2
    results = scorer.score_batch(["Q"], [[]])
    assert results[0]["score"] == 0.0
    assert results[0]["step_labels"] == []


def test_trainconfig_requires_fields():
    config = TrainConfig.from_dict(
        {"run_name": "r", "model": "launch/ThinkPRM-1.5B", "train_data": "t.jsonl"}
    )
    assert config.use_lora is True
    assert config.add_think_token is True


def test_build_training_text_preserves_critique_for_think_template():
    from thinkprm_yoruba.train import build_training_text

    config = TrainConfig.from_dict(
        {"run_name": "r", "model": "launch/ThinkPRM-1.5B", "train_data": "t.jsonl"}
    )
    row = {
        "problem": "Q?",
        "solution": "Step 1: a",
        "cot": "Step 1 is fine \\boxed{correct}",
        "is_correct": False,
        "labels": [0],
    }
    text = build_training_text(_ThinkTokenizer(), row, config)
    # The prefix opens <think>; the critique must survive the target (regression:
    # the base chat template drops assistant content before </think>).
    assert text.count("<think>") == 1
    assert "Step 1 is fine \\boxed{correct}" in text
    assert text.endswith("</think>\nIs the solution correct? No")
