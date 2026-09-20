"""Experiment config integrity: every backend is an OpenAI-compatible REST API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import load_inference_run_config

QWEN35_MODELS = {"qwen3.5-4b", "qwen3.5-9b"}


def test_e1_vllm_config() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_vllm.json")
    assert cfg.run_id == "e1_reasoning_language_vllm"
    assert all(m.backend == "openai_compatible" for m in cfg.models)
    assert all(m.base_url_env == "OPENAI_COMPATIBLE_BASE_URL" for m in cfg.models)
    assert all(m.api_key_env == "OPENAI_COMPATIBLE_API_KEY" for m in cfg.models)
    assert QWEN35_MODELS <= {m.name for m in cfg.models}


def test_e2_vllm_config() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling_vllm.json")
    assert cfg.run_id == "e2_ttc_scaling_vllm"
    # Nested expansion: n1 + n4..n64
    assert any(m.n == 1 for m in cfg.methods)
    assert max(m.n for m in cfg.methods) == 64
    assert any(m.nested_group_id for m in cfg.methods)
    assert QWEN35_MODELS <= {m.name for m in cfg.models}


def test_e2_optional_vllm_config() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling_optional_vllm.json")
    assert cfg.run_id == "e2_ttc_scaling_optional_vllm"
    assert all(m.backend == "openai_compatible" for m in cfg.models)


def test_router_configs_use_openai_compatible_backends() -> None:
    for name in (
        "e0_english_baseline_openrouter.json",
        "e1_reasoning_language_openrouter.json",
        "e2_ttc_scaling_openrouter.json",
    ):
        cfg = load_inference_run_config(ROOT / "configs" / name)
        assert all(m.backend == "openai_compatible" for m in cfg.models)
        assert "qwen3.5-9b" in {m.name for m in cfg.models}
    for name in (
        "e0_english_baseline_ramp_router.json",
        "e1_reasoning_language_ramp_router.json",
    ):
        cfg = load_inference_run_config(ROOT / "configs" / name)
        assert all(m.backend == "responses_api" for m in cfg.models)
        assert QWEN35_MODELS <= {m.name for m in cfg.models}


def test_no_transformers_backend_in_any_config() -> None:
    for path in sorted((ROOT / "configs").glob("*.json")):
        if path.name.startswith("e4"):
            continue
        cfg = load_inference_run_config(path)
        assert all(m.backend != "transformers" for m in cfg.models), path.name


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
    test_e1_vllm_config()
    test_e2_vllm_config()
    test_e2_optional_vllm_config()
    test_router_configs_use_openai_compatible_backends()
    test_no_transformers_backend_in_any_config()
    test_e4_vllm_comparison_config()
    print("ok")
