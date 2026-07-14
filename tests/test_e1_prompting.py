"""Unit tests for E1 reasoning-language prompts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import load_inference_run_config
from ttcs_yoruba.examples import InferenceExample
from ttcs_yoruba.extraction import extract_answer
from ttcs_yoruba.prompting import render_prompt


def _math_example() -> InferenceExample:
    return InferenceExample(
        id="math_001",
        task="math",
        question="Bàbá ra àpótí 2. Ọ̀kọ̀ọ̀kan ní ẹṣọ 3. Melo ni lápapọ̀?",
        choices=None,
        gold_answer="6",
        answer_type="number",
        source_dataset="afrimgsm",
    )


def test_e1_config_has_three_strategies() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language.json")
    assert len(cfg.models) == 3
    assert {m.prompt_style for m in cfg.methods} == {
        "yoruba_cot",
        "english_cot",
        "translate_pivot",
    }
    assert {m.reasoning_language for m in cfg.methods} == {"yo", "en", "en_pivot"}


def test_translate_pivot_instructs_translation() -> None:
    prompt = render_prompt(_math_example(), "translate_pivot")
    assert "English translation" in prompt.user
    assert "Translate the Yoruba question" in prompt.user
    assert "Question (Yoruba)" in prompt.user


def test_yoruba_cot_uses_yoruba_exemplar_reasoning() -> None:
    prompt = render_prompt(_math_example(), "yoruba_cot")
    assert "Start with 10 boxes" not in prompt.user
    assert "Bẹ̀rẹ̀ pẹ̀lú àpótí" in prompt.user or "Àpapọ̀" in prompt.user


def test_english_cot_uses_english_exemplar_reasoning() -> None:
    prompt = render_prompt(_math_example(), "english_cot")
    assert "Start with 10 boxes" in prompt.user
    assert "Bẹ̀rẹ̀ pẹ̀lú àpótí" not in prompt.user


def test_extract_answer_from_translate_pivot_response() -> None:
    response = (
        "English translation: A father bought 2 boxes with 3 oranges each.\n"
        "Reasoning: 2 * 3 = 6.\n"
        "Final answer: 6"
    )
    assert extract_answer(response, "number") == "6"


if __name__ == "__main__":
    test_e1_config_has_three_strategies()
    test_translate_pivot_instructs_translation()
    test_yoruba_cot_uses_yoruba_exemplar_reasoning()
    test_english_cot_uses_english_exemplar_reasoning()
    test_extract_answer_from_translate_pivot_response()
    print("ok")
