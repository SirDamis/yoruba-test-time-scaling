"""Multi-language support: dataset keys, --language filtering, language-aware prompts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.datasets import (
    ALL_LANGUAGE_CODES,
    available_dataset_keys,
    build_language_spec,
    resolve_dataset_spec,
)
from ttcs_yoruba.examples import InferenceExample
from ttcs_yoruba.prompting import example_language, render_prompt


def _example(source_dataset: str, task: str = "math") -> InferenceExample:
    return InferenceExample(
        id="x_001",
        task=task,
        question="Test question",
        choices=None,
        gold_answer="42",
        answer_type="number",
        source_dataset=source_dataset,
    )


def test_available_keys_cover_all_template_languages() -> None:
    keys = set(available_dataset_keys())
    for base in ("afrimgsm", "afrimmlu"):
        for lang in ALL_LANGUAGE_CODES:
            assert f"{base}_{lang}" in keys


def test_resolve_spec_per_language() -> None:
    spec = resolve_dataset_spec("afrimgsm_hau")
    assert spec.key == "afrimgsm_hau"
    assert spec.config == "hau"
    assert spec.data_files["test"].endswith("/data/hau/test.tsv")

    spec = resolve_dataset_spec("afrimmlu_swa")
    assert spec.config == "swa"
    assert spec.data_files["test"].endswith("/data/swa/test.tsv")


def test_resolve_spec_legacy_and_translate() -> None:
    legacy = resolve_dataset_spec("afrimgsm")
    assert legacy.key == "afrimgsm_yor"
    assert legacy.config == "yor"

    translate = resolve_dataset_spec("afrimgsm_translate")
    assert translate.config == "yor"


def test_example_language_inference() -> None:
    assert example_language(_example("afrimgsm_yor")) == "yor"
    assert example_language(_example("afrimmlu_hau")) == "hau"
    assert example_language(_example("afrimmlu_swa")) == "swa"
    assert example_language(_example("afrimgsm_amh")) == "amh"
    # Legacy names and English-input variants.
    assert example_language(_example("afrimgsm")) == "yor"
    assert example_language(_example("afrimgsm_translate")) == "eng"


def test_prompts_generalize_to_non_yoruba_languages() -> None:
    ex = _example("afrimgsm_hau")

    native = render_prompt(ex, "yoruba_cot")
    assert "Read the Hausa problem" in native.system
    assert "Reason step by step in Hausa" in native.user
    assert "Question (Hausa)" in native.user
    # No Yoruba exemplars exist for Hausa -> zero-shot.
    assert "Examples:" not in native.user

    english = render_prompt(ex, "english_cot")
    assert "Reason step by step in English" in english.user
    # English-question exemplars for non-Yoruba inputs.
    assert "Tom has 5 bags" in english.user

    pivot = render_prompt(ex, "translate_pivot")
    assert "Translate the Hausa question into English" in pivot.user


def test_yoruba_prompts_keep_few_shot_exemplars() -> None:
    ex = _example("afrimgsm_yor")
    native = render_prompt(ex, "yoruba_cot")
    assert "Examples:" in native.user
    assert "Reason step by step in Yoruba" in native.user


if __name__ == "__main__":
    test_available_keys_cover_all_template_languages()
    test_resolve_spec_per_language()
    test_resolve_spec_legacy_and_translate()
    test_example_language_inference()
    test_prompts_generalize_to_non_yoruba_languages()
    test_yoruba_prompts_keep_few_shot_exemplars()
    print("ok")
