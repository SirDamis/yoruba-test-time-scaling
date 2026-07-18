"""vLLM experiment configs mirror HF experiment matrices with openai_compatible backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import load_inference_run_config


def _assert_vllm_mirrors_hf(hf_name: str, vllm_name: str, *, expected_run_id: str) -> None:
    hf = load_inference_run_config(ROOT / "configs" / hf_name)
    vllm = load_inference_run_config(ROOT / "configs" / vllm_name)

    assert vllm.run_id == expected_run_id
    assert {m.name for m in vllm.models} == {m.name for m in hf.models}
    assert {m.model for m in vllm.models} == {m.model for m in hf.models}
    assert all(m.backend == "openai_compatible" for m in vllm.models)
    assert all(m.base_url_env == "OPENAI_COMPATIBLE_BASE_URL" for m in vllm.models)
    assert all(m.api_key_env == "OPENAI_COMPATIBLE_API_KEY" for m in vllm.models)
    assert all(m.backend == "transformers" for m in hf.models)

    # Method templates match (after expand, names/styles/n may differ in shape but
    # source rows should share prompt_style / nested intent).
    assert len(vllm.methods) == len(hf.methods)
    for hm, vm in zip(hf.methods, vllm.methods, strict=True):
        assert hm.name == vm.name
        assert hm.prompt_style == vm.prompt_style
        assert hm.selection == vm.selection
        assert hm.n == vm.n
        assert hm.reasoning_language == vm.reasoning_language
        assert hm.nested_group_id == vm.nested_group_id


def test_e1_vllm_mirrors_hf() -> None:
    _assert_vllm_mirrors_hf(
        "e1_reasoning_language.json",
        "e1_reasoning_language_vllm.json",
        expected_run_id="e1_reasoning_language_vllm",
    )


def test_e2_vllm_mirrors_hf() -> None:
    _assert_vllm_mirrors_hf(
        "e2_ttc_scaling.json",
        "e2_ttc_scaling_vllm.json",
        expected_run_id="e2_ttc_scaling_vllm",
    )
    vllm = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling_vllm.json")
    # Nested expansion: n1 + n4..n64
    assert any(m.n == 1 for m in vllm.methods)
    assert max(m.n for m in vllm.methods) == 64
    assert any(m.nested_group_id for m in vllm.methods)


def test_e2_optional_vllm_mirrors_hf() -> None:
    _assert_vllm_mirrors_hf(
        "e2_ttc_scaling_optional.json",
        "e2_ttc_scaling_optional_vllm.json",
        expected_run_id="e2_ttc_scaling_optional_vllm",
    )


def test_e4_vllm_comparison_config() -> None:
    hf = json.loads((ROOT / "configs" / "e4_comparison.json").read_text(encoding="utf-8"))
    vllm = json.loads((ROOT / "configs" / "e4_comparison_vllm.json").read_text(encoding="utf-8"))
    for key in (
        "small_model",
        "large_model",
        "large_n",
        "include_models",
        "small_method",
        "large_method",
    ):
        assert vllm[key] == hf[key]
    assert "e2_ttc_scaling_vllm" in str(vllm.get("notes", ""))


if __name__ == "__main__":
    test_e1_vllm_mirrors_hf()
    test_e2_vllm_mirrors_hf()
    test_e2_optional_vllm_mirrors_hf()
    test_e4_vllm_comparison_config()
    print("ok")
