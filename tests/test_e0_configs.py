"""E0 (English baseline) config checks: native eng splits, greedy english_cot, backend parity."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import load_inference_run_config


def _assert_e0_scope(cfg) -> None:
    names = {d.name for d in cfg.datasets}
    assert names == {"afrimgsm_eng", "afrimmlu_eng"}
    for d in cfg.datasets:
        assert str(d.path).endswith(f"{d.name}/test.jsonl")
    # E0 is the English-arm of E1: same method, greedy decode.
    assert len(cfg.methods) == 1
    method = cfg.methods[0]
    assert method.prompt_style == "english_cot"
    assert method.n == 1 and method.temperature == 0.0 and method.selection == "first"


def test_e0_hf_config() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e0_english_baseline.json")
    _assert_e0_scope(cfg)
    assert cfg.run_id == "e0_english_baseline"
    assert all(m.backend == "transformers" for m in cfg.models)


def test_e0_vllm_config_mirrors_hf() -> None:
    hf = load_inference_run_config(ROOT / "configs" / "e0_english_baseline.json")
    vllm = load_inference_run_config(ROOT / "configs" / "e0_english_baseline_vllm.json")
    assert vllm.run_id == "e0_english_baseline_vllm"
    assert {m.name for m in vllm.models} == {m.name for m in hf.models}
    assert all(m.backend == "openai_compatible" for m in vllm.models)
    assert [(m.name, m.prompt_style, m.n, m.temperature) for m in vllm.methods] == [
        (m.name, m.prompt_style, m.n, m.temperature) for m in hf.methods
    ]


def test_e0_openrouter_config() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e0_english_baseline_openrouter.json")
    _assert_e0_scope(cfg)
    assert cfg.run_id == "e0_english_baseline_openrouter"
    assert cfg.max_concurrent == 4
    assert cfg.transient_retry_rounds == 5
    assert {m.name for m in cfg.models} == {
        "qwen3-4b",
        "qwen3-8b",
        "gemma3-4b",
        "llama3.2-3b",
    }
    for model in cfg.models:
        assert model.backend == "openai_compatible"
        assert model.base_url == "https://openrouter.ai/api/v1"
        assert model.api_key_env == "OPENROUTER_API_KEY"
        assert model.backend_kwargs.get("max_retries") == 5
        # Prompted-CoT experiment: native thinking off for every model.
        assert model.backend_kwargs.get("reasoning") == {"enabled": False}
        # Let OpenRouter route around a rate-limited upstream provider.
        assert model.backend_kwargs.get("extra_body") == {
            "provider": {"allow_fallbacks": False}
        }


def test_e0_ramp_router_config() -> None:
    from ttcs_yoruba.inference import effective_max_concurrent

    cfg = load_inference_run_config(ROOT / "configs" / "e0_english_baseline_ramp_router.json")
    _assert_e0_scope(cfg)
    assert cfg.run_id == "e0_english_baseline_ramp_router"
    assert {m.name for m in cfg.models} == {
        "qwen3-4b",
        "gemma3-4b",
        "llama3.2-3b",
        "deepseek-v4-flash",
    }
    for model in cfg.models:
        assert model.backend == "responses_api"
        assert model.base_url == "https://api.router.com/v1"
        assert model.api_key_env == "ROUTER_KEY"
        assert model.backend_kwargs.get("reasoning_effort") == "none"
        assert model.backend_kwargs.get("impersonate") == "chrome"
        assert effective_max_concurrent(cfg, model) == 4


def test_e0_matches_e1_decode_settings() -> None:
    """E0 must share E1's max_tokens so gap estimates aren't confounded by truncation."""
    e1 = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language.json")
    e0 = load_inference_run_config(ROOT / "configs" / "e0_english_baseline.json")
    e1_tokens = {m.max_tokens for m in e1.methods}
    e0_tokens = {m.max_tokens for m in e0.methods}
    assert e1_tokens == e0_tokens


if __name__ == "__main__":
    test_e0_hf_config()
    test_e0_vllm_config_mirrors_hf()
    test_e0_openrouter_config()
    test_e0_ramp_router_config()
    test_e0_matches_e1_decode_settings()
    print("ok")
