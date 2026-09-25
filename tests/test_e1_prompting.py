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


def _translated_math_example() -> InferenceExample:
    return InferenceExample(
        id="math_en_001",
        task="math",
        question="Titi has 3 books. She buys 2 packs with 4 books each. How many books does she have now?",
        choices=None,
        gold_answer="11",
        answer_type="number",
        source_dataset="afrimgsm_translate",
    )


def test_e1_config_has_expected_strategies() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_vllm.json")
    assert len(cfg.models) == 5
    assert {m.prompt_style for m in cfg.methods} == {
        "yoruba_cot",
        "english_cot",
        "translate_pivot",
        "direct",
    }
    assert {m.reasoning_language for m in cfg.methods} == {"yo", "en", "en_pivot", "none"}
    assert all(m.backend == "openai_compatible" for m in cfg.models)


def test_e1_vllm_config_matches_experiment_with_openai_compatible() -> None:
    """E1-vLLM is the experiment matrix, served via a local OpenAI-compatible API."""
    vllm = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_vllm.json")
    assert {m.name for m in vllm.models} == {
        "qwen3.5-4b",
        "qwen3.5-9b",
        "qwen3.5-27b",
        "gemma3-4b",
        "llama3.2-3b",
    }
    assert all(m.backend == "openai_compatible" for m in vllm.models)
    assert all(m.base_url_env == "OPENAI_COMPATIBLE_BASE_URL" for m in vllm.models)
    assert all(m.api_key_env == "OPENAI_COMPATIBLE_API_KEY" for m in vllm.models)
    assert {m.prompt_style for m in vllm.methods} == {
        "yoruba_cot",
        "english_cot",
        "translate_pivot",
        "direct",
    }
    assert {m.reasoning_language for m in vllm.methods} == {"yo", "en", "en_pivot", "none"}
    assert all(m.n == 1 and m.selection == "first" for m in vllm.methods)
    assert vllm.run_id == "e1_reasoning_language_vllm"


def test_translate_pivot_instructs_translation() -> None:
    prompt = render_prompt(_math_example(), "translate_pivot")
    assert "Read the Yoruba problem" in prompt.system
    assert "English translation" in prompt.user
    assert "Translate the Yoruba question" in prompt.user
    assert "Question (Yoruba)" in prompt.user


def test_yoruba_cot_uses_yoruba_exemplar_reasoning() -> None:
    prompt = render_prompt(_math_example(), "yoruba_cot")
    assert "Pínpín dọ́gba" in prompt.user
    assert "First translate the quantities" not in prompt.user
    assert "Kọ́kọ́ túmọ̀ iye" not in prompt.user
    assert "translate" not in prompt.user.lower()


def test_english_cot_uses_english_exemplar_reasoning() -> None:
    prompt = render_prompt(_math_example(), "english_cot")
    assert "Sharing equally means" in prompt.user
    assert "Kọ́kọ́ túmọ̀ iye" not in prompt.user
    assert "translate" not in prompt.user.lower()


def test_math_exemplar_reasoning_avoids_translation_and_varies() -> None:
    """Exemplars must not prime a translate-first strategy (that is translate_pivot only)."""
    from ttcs_yoruba.prompting import TASK_EXEMPLARS

    en_reasonings = [ex.reasoning_en for ex in TASK_EXEMPLARS["math"]]
    yo_reasonings = [ex.reasoning_yo for ex in TASK_EXEMPLARS["math"]]
    for text in en_reasonings:
        assert "translate" not in text.lower()
    for text in yo_reasonings:
        assert "túmọ̀" not in text
    # No single boilerplate opener repeated across the demonstrations.
    assert len({text.split()[0] for text in en_reasonings}) > 1
    assert len({text.split()[0] for text in yo_reasonings}) > 1


def test_english_cot_on_translated_dataset_does_not_call_question_yoruba() -> None:
    prompt = render_prompt(_translated_math_example(), "english_cot")
    assert "Read the Yoruba problem" not in prompt.system
    assert "Question (Yoruba)" not in prompt.user
    assert "Yoruba quantities" not in prompt.user
    assert "Question:\nTiti has 3 books" in prompt.user


def test_direct_prompt_omits_reasoning() -> None:
    prompt = render_prompt(_math_example(), "direct")
    assert "Reasoning:" not in prompt.user
    assert "First translate the quantities" not in prompt.user
    assert "Kọ́kọ́ túmọ̀ iye" not in prompt.user
    assert "Answer directly without showing any reasoning" in prompt.user
    assert "Question (Yoruba)" in prompt.user
    # Yoruba math exemplar shown answer-only.
    assert "Final answer: 3" in prompt.user


def test_direct_prompt_on_translated_dataset_uses_english_exemplars() -> None:
    prompt = render_prompt(_translated_math_example(), "direct")
    assert "Reasoning:" not in prompt.user
    assert "Answer directly without showing any reasoning" in prompt.user
    assert "Question (Yoruba)" not in prompt.user
    # English-input exemplars, shown answer-only.
    assert "Tom has 5 bags" in prompt.user


def test_extract_answer_from_translate_pivot_response() -> None:
    response = (
        "English translation: A father bought 2 boxes with 3 oranges each.\n"
        "Reasoning: 2 * 3 = 6.\n"
        "Final answer: 6"
    )
    assert extract_answer(response, "number") == "6"


if __name__ == "__main__":
    test_e1_config_has_expected_strategies()
    test_e1_vllm_config_matches_experiment_with_openai_compatible()
    test_translate_pivot_instructs_translation()
    test_yoruba_cot_uses_yoruba_exemplar_reasoning()
    test_english_cot_uses_english_exemplar_reasoning()
    test_math_exemplar_reasoning_avoids_translation_and_varies()
    test_english_cot_on_translated_dataset_does_not_call_question_yoruba()
    test_direct_prompt_omits_reasoning()
    test_direct_prompt_on_translated_dataset_uses_english_exemplars()
    test_extract_answer_from_translate_pivot_response()
    print("ok")
