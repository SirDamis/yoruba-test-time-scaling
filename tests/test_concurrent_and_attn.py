"""Tests for concurrent batching helpers and attention resolution."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.backends import (
    BackendError,
    OpenAICompatibleChatBackend,
    OpenAIResponsesBackend,
    build_backend,
)
from ttcs_yoruba.config import InferenceModelConfig, InferenceRunConfig, load_inference_run_config
from ttcs_yoruba.inference import (
    _provider_cost,
    effective_max_concurrent,
    run_concurrent_map,
    run_concurrent_map_tolerant,
)


def _openai_compatible_config(**backend_kwargs: object) -> InferenceModelConfig:
    return InferenceModelConfig(
        name="test-api",
        backend="openai_compatible",
        model="test/model",
        size_label="test",
        base_url="https://example.test/api/v1",
        api_key="test-key",
        backend_kwargs=dict(backend_kwargs),
    )


def test_run_concurrent_map_preserves_order() -> None:
    def work(x: int) -> int:
        # Reverse-ish timing so completion order != input order.
        time.sleep(0.01 * (5 - x))
        return x * 10

    out = run_concurrent_map([1, 2, 3, 4], work, max_workers=4)
    assert out == [10, 20, 30, 40]


def test_run_concurrent_map_serial_when_one_worker() -> None:
    out = run_concurrent_map([1, 2, 3], lambda x: x + 1, max_workers=1)
    assert out == [2, 3, 4]


def test_run_concurrent_map_tolerant_keeps_successes_on_partial_failure() -> None:
    def work(x: int) -> int:
        if x == 2:
            raise RuntimeError("fail-2")
        return x * 10

    results, errors = run_concurrent_map_tolerant([1, 2, 3], work, max_workers=3)
    assert results == [10, None, 30]
    assert len(errors) == 1
    assert errors[0][0] == 1
    assert "fail-2" in str(errors[0][1])


def test_effective_max_concurrent_uses_configured_value() -> None:
    from ttcs_yoruba.config import DatasetConfig, InferenceMethodConfig

    cfg = InferenceRunConfig(
        run_id="t",
        output_dir=Path("runs"),
        datasets=[
            DatasetConfig(name="d", path=Path("x.jsonl"), task="math", source_dataset="d")
        ],
        models=[
            InferenceModelConfig(
                name="m", backend="openai_compatible", model="x", size_label="4B"
            )
        ],
        methods=[
            InferenceMethodConfig(
                name="english_cot",
                prompt_style="english_cot",
                selection="first",
                n=1,
            )
        ],
        max_concurrent=16,
    )
    assert effective_max_concurrent(cfg, cfg.models[0]) == 16
    assert effective_max_concurrent(
        InferenceRunConfig(
            run_id="t2",
            output_dir=Path("runs"),
            datasets=[],
            models=[],
            methods=[],
            max_concurrent=0,
        ),
        cfg.models[0],
    ) == 1


def test_openai_compatible_retries_rate_limits() -> None:
    backend = OpenAICompatibleChatBackend(
        _openai_compatible_config(
            max_retries=1,
            retry_initial_backoff_s=0,
            retry_max_backoff_s=0,
        )
    )
    calls = 0

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"choices":[{"message":{"content":"ok"}}],"usage":{"cost":0.01}}'

    def urlopen(*args: object, **kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://example.test",
                429,
                "rate limited",
                {"Retry-After": "0"},
                BytesIO(b"rate limited"),
            )
        return _Response()

    with patch("urllib.request.urlopen", side_effect=urlopen):
        result = backend._post_json({"model": "test/model", "messages": []})
    assert calls == 2
    assert result["usage"]["cost"] == 0.01


def test_provider_cost_accepts_valid_usage_cost_only() -> None:
    assert _provider_cost({"usage": {"cost": 0.0012}}) == 0.0012
    assert _provider_cost({"usage": {"cost": "0.0012"}}) == 0.0012
    assert _provider_cost({"usage": {"cost": -1}}) is None
    assert _provider_cost({"usage": {"cost": True}}) is None
    assert _provider_cost({}) is None


def _assert_experiment_scope(cfg: object) -> None:
    """E1/E2 target the 5 experiment languages + native English baselines, test splits only."""
    datasets = {d.name: d for d in cfg.datasets}  # type: ignore[attr-defined]
    expected = set()
    for base in ("afrimgsm", "afrimmlu"):
        for lang in ("yor", "hau", "ibo", "swa", "amh", "eng"):
            expected.add(f"{base}_{lang}")
    assert set(datasets) == expected
    for name, d in datasets.items():
        assert str(d.path).endswith(f"{name}/test.jsonl"), name


def test_e1_vllm_config_has_concurrency_and_2048() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_vllm.json")
    assert cfg.max_concurrent == 8
    assert all(m.max_tokens == 2048 for m in cfg.methods)
    _assert_experiment_scope(cfg)


def test_e2_configs_target_test_splits_only() -> None:
    for name in ("e2_ttc_scaling_vllm.json", "e2_ttc_scaling_optional_vllm.json"):
        cfg = load_inference_run_config(ROOT / "configs" / name)
        _assert_experiment_scope(cfg)


def test_e1_openrouter_config_has_cost_and_retry_settings() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_openrouter.json")
    assert cfg.max_concurrent == 4
    assert all(m.max_tokens == 2048 for m in cfg.methods)
    assert {m.name for m in cfg.models} == {"qwen3.5-9b", "gemma3-4b", "llama3.2-3b", "deepseek-v4-flash", "deepseek-v4.1-flash"}
    for model in cfg.models:
        assert model.backend == "openai_compatible"
        assert model.base_url == "https://openrouter.ai/api/v1"
        assert model.api_key_env == "OPENROUTER_API_KEY"
        assert model.backend_kwargs.get("max_retries") == 5
        # Prompted-CoT experiment: native thinking off for every model.
        assert model.backend_kwargs.get("reasoning") == {"enabled": False}
        assert model.backend_kwargs.get("extra_body") == {
            "provider": {"allow_fallbacks": False}
        }
    assert cfg.transient_retry_rounds == 5


def test_e1_ramp_router_config_uses_responses_backend() -> None:
    from ttcs_yoruba.inference import effective_max_concurrent

    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_ramp_router.json")
    assert cfg.max_concurrent == 4
    assert all(m.max_tokens == 2048 for m in cfg.methods)
    assert {m.name for m in cfg.models} == {"qwen3.5-4b", "qwen3.5-9b", "gemma3-4b", "llama3.2-3b", "deepseek-v4-flash", "deepseek-v4.1-flash"}
    for model in cfg.models:
        assert model.backend == "responses_api"
        assert model.base_url == "https://api.router.com/v1"
        assert model.api_key_env == "ROUTER_KEY"
        assert model.backend_kwargs.get("reasoning_effort") == "none"
        assert model.backend_kwargs.get("impersonate") == "chrome"
        assert "User-Agent" in model.backend_kwargs.get("headers", {})
        assert effective_max_concurrent(cfg, model) == 4


def _responses_config(**backend_kwargs: object) -> InferenceModelConfig:
    return InferenceModelConfig(
        name="test-responses",
        backend="responses_api",
        model="test/model",
        size_label="test",
        base_url="https://example.test/api/v1",
        api_key="test-key",
        backend_kwargs=dict(backend_kwargs),
    )


def test_responses_payload_and_output_extraction() -> None:
    backend = OpenAIResponsesBackend(_responses_config(send_seed=True, reasoning_effort="none"))

    calls: list[dict[str, object]] = []

    def capture(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {
            "id": "resp_test",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The answer is 42."}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 7, "total_tokens": 17},
        }

    with patch.object(backend, "_post_json", side_effect=capture):
        result = backend.generate(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
            max_tokens=512,
            seed=123,
            top_p=0.9,
        )

    assert calls[0]["model"] == "test/model"
    assert calls[0]["instructions"] == "sys"
    assert calls[0]["input"] == "user"
    assert calls[0]["max_output_tokens"] == 512
    assert calls[0]["seed"] == 123
    assert calls[0]["reasoning"] == {"effort": "none"}
    assert "messages" not in calls[0]
    assert "max_tokens" not in calls[0]
    assert result.response == "The answer is 42."
    assert result.token_count == 7
    assert result.prompt_token_count == 10
    assert result.metadata["status"] == "completed"


def test_responses_extract_text_skips_reasoning_and_string_fallbacks() -> None:
    backend = OpenAIResponsesBackend(_responses_config())
    # reasoning-only output -> no text -> raises
    assert (
        backend._extract_text({"output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "x"}]}]})
        == ""
    )
    # top-level output_text fallback
    assert backend._extract_text({"output_text": "direct text"}) == "direct text"
    # nested message content extraction
    body = {
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "b"}]},
        ]
    }
    assert backend._extract_text(body) == "a\nb"


def test_responses_backend_builds() -> None:
    backend = build_backend(_responses_config(), default_timeout_s=60)
    assert isinstance(backend, OpenAIResponsesBackend)
    # endpoint is /responses, not /chat/completions
    assert backend.endpoint_path == "responses"
    assert backend._endpoint_url() == "https://example.test/api/v1/responses"
    chat = build_backend(_openai_compatible_config(), default_timeout_s=60)
    assert chat._endpoint_url() == "https://example.test/api/v1/chat/completions"


def test_responses_no_reasoning_key_when_omitted() -> None:
    backend = OpenAIResponsesBackend(_responses_config())

    captured: list[dict[str, object]] = []

    def capture(payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        return {"status": "completed", "output": []}

    with patch.object(backend, "_post_json", side_effect=capture):
        try:
            backend.generate(
                system_prompt="",
                user_prompt="hi",
                temperature=None,
                max_tokens=None,
            )
        except BackendError:
            pass
    assert captured
    assert "reasoning" not in captured[0]


def _capture_chat_payload(backend: OpenAICompatibleChatBackend) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    def capture(payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }

    with patch.object(backend, "_post_json", side_effect=capture):
        backend.generate(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
            max_tokens=16,
        )
    return captured


def test_chat_backend_forwards_reasoning_mapping() -> None:
    backend = OpenAICompatibleChatBackend(
        _openai_compatible_config(reasoning={"enabled": False})
    )
    captured = _capture_chat_payload(backend)
    assert captured[0]["reasoning"] == {"enabled": False}


def test_chat_backend_reasoning_bool_shorthand() -> None:
    backend = OpenAICompatibleChatBackend(_openai_compatible_config(reasoning=True))
    captured = _capture_chat_payload(backend)
    assert captured[0]["reasoning"] == {"enabled": True}


def test_chat_backend_no_reasoning_key_when_omitted() -> None:
    backend = OpenAICompatibleChatBackend(_openai_compatible_config())
    captured = _capture_chat_payload(backend)
    assert "reasoning" not in captured[0]


def test_chat_backend_tracks_prompt_tokens() -> None:
    backend = OpenAICompatibleChatBackend(_openai_compatible_config())

    def capture(payload: dict[str, object]) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }

    with patch.object(backend, "_post_json", side_effect=capture):
        result = backend.generate(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
            max_tokens=16,
        )
    assert result.prompt_token_count == 42
    assert result.token_count == 7


def test_chat_backend_prompt_tokens_none_when_absent() -> None:
    backend = OpenAICompatibleChatBackend(_openai_compatible_config())

    def capture(payload: dict[str, object]) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }

    with patch.object(backend, "_post_json", side_effect=capture):
        result = backend.generate(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
            max_tokens=16,
        )
    assert result.prompt_token_count is None


def test_urllib_transport_raises_transport_http_error() -> None:
    backend = OpenAIResponsesBackend(_responses_config())

    def urlopen(*args: object, **kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://example.test",
            403,
            "forbidden",
            {"Retry-After": "2"},
            BytesIO(b"cloudflare 1010"),
        )

    with patch("urllib.request.urlopen", side_effect=urlopen):
        with pytest.raises(BackendError) as excinfo:
            backend._post_json({"model": "x"})
    assert "HTTP 403" in str(excinfo.value)
    assert "cloudflare 1010" in str(excinfo.value)


def test_curl_cffi_transport_used_when_impersonate_set() -> None:
    pytest.importorskip("curl_cffi")
    backend = OpenAIResponsesBackend(_responses_config(impersonate="chrome"))
    captured: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        text = '{"ok": true}'
        headers = {}

    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        captured["content"] = kwargs.get("content")
        captured["impersonate"] = kwargs.get("impersonate")
        return _FakeResponse()

    with patch("curl_cffi.requests.post", side_effect=fake_post) as mock_post:
        result = backend._post_json({"model": "x"})
    assert mock_post.called
    assert captured["impersonate"] == "chrome"
    assert result.get("ok") is True
    assert "_latency_s" in result


def test_curl_cffi_transport_rejects_4xx() -> None:
    pytest.importorskip("curl_cffi")
    from ttcs_yoruba.backends import _TransportHTTPError

    backend = OpenAIResponsesBackend(_responses_config(impersonate="chrome"))

    class _FakeResponse:
        status_code = 403
        text = "cloudflare error code 1010"
        headers = {}

    with patch("curl_cffi.requests.post", return_value=_FakeResponse()):
        try:
            backend._post_json({"model": "x"})
        except BackendError as exc:
            assert "cloudflare error code 1010" in str(exc)
            assert "HTTP 403" in str(exc)
        else:
            raise AssertionError("expected BackendError for 403")


def test_process_wave_retries_transient_failures() -> None:
    from ttcs_yoruba.inference import process_wave_with_retries

    calls = {"n": 0}

    def worker(item: int) -> str:
        calls["n"] += 1
        if item == 2 and calls["n"] <= 2:
            raise BackendError("HTTP 429 rate limited", retryable=True)
        return f"ok-{item}"

    seen: list[str] = []
    process_wave_with_retries(
        [1, 2, 3],
        worker,
        concurrency=1,
        max_rounds=3,
        initial_backoff_s=0.0,
        max_backoff_s=0.0,
        on_success=seen.append,
    )
    assert sorted(seen) == ["ok-1", "ok-2", "ok-3"]
    assert calls["n"] == 4  # item 2 failed once, then one retry


def test_process_wave_raises_non_transient_immediately() -> None:
    from ttcs_yoruba.inference import process_wave_with_retries

    calls = {"n": 0}

    def worker(item: int) -> str:
        calls["n"] += 1
        raise BackendError("HTTP 400 bad request", retryable=False)

    with pytest.raises(BackendError):
        process_wave_with_retries(
            [1],
            worker,
            concurrency=1,
            max_rounds=3,
            initial_backoff_s=0.0,
            max_backoff_s=0.0,
            on_success=lambda result: None,
        )
    assert calls["n"] == 1


def test_process_wave_exhausts_transient_retries() -> None:
    from ttcs_yoruba.inference import process_wave_with_retries

    calls = {"n": 0}

    def worker(item: int) -> str:
        calls["n"] += 1
        raise BackendError("HTTP 503 upstream error", retryable=True)

    with pytest.raises(BackendError):
        process_wave_with_retries(
            [1],
            worker,
            concurrency=1,
            max_rounds=2,
            initial_backoff_s=0.0,
            max_backoff_s=0.0,
            on_success=lambda result: None,
        )
    assert calls["n"] == 3  # initial attempt + 2 retry rounds


if __name__ == "__main__":
    test_run_concurrent_map_preserves_order()
    test_run_concurrent_map_serial_when_one_worker()
    test_run_concurrent_map_tolerant_keeps_successes_on_partial_failure()
    test_effective_max_concurrent_uses_configured_value()
    test_openai_compatible_retries_rate_limits()
    test_provider_cost_accepts_valid_usage_cost_only()
    test_process_wave_retries_transient_failures()
    test_process_wave_raises_non_transient_immediately()
    test_process_wave_exhausts_transient_retries()
    test_e1_vllm_config_has_concurrency_and_2048()
    test_e2_configs_target_test_splits_only()
    test_e1_openrouter_config_has_cost_and_retry_settings()
    test_e1_ramp_router_config_uses_responses_backend()
    test_responses_payload_and_output_extraction()
    test_responses_extract_text_skips_reasoning_and_string_fallbacks()
    test_responses_backend_builds()
    test_responses_no_reasoning_key_when_omitted()
    test_chat_backend_forwards_reasoning_mapping()
    test_chat_backend_reasoning_bool_shorthand()
    test_chat_backend_no_reasoning_key_when_omitted()
    test_chat_backend_tracks_prompt_tokens()
    test_chat_backend_prompt_tokens_none_when_absent()
    test_urllib_transport_raises_transport_http_error()
    test_curl_cffi_transport_used_when_impersonate_set()
    test_curl_cffi_transport_rejects_4xx()
    print("ok")
