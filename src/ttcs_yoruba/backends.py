from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import InferenceModelConfig
from .schema import BackendOutput


class BackendError(RuntimeError):
    """Raised when a model backend cannot complete a generation request."""


@dataclass(frozen=True)
class _TransportHTTPError(Exception):
    """Normalized HTTP error from any transport (urllib or curl_cffi)."""

    code: int
    error_body: str
    retry_after_s: float | None = None


class _TransportNetworkError(RuntimeError):
    """Normalized network-level failure (DNS, TLS, connect, timeout)."""


def resolve_attn_implementation(requested: Any, torch_mod: Any) -> str | None:
    """Pick an attention backend for Hugging Face Transformers.

    - ``auto`` / omitted: FlashAttention-2 on CUDA SM>=8.0 when ``flash_attn`` is
      installed; otherwise PyTorch SDPA (including Turing/T4 and CPU).
    - Explicit values are returned as-is (``flash_attention_2``, ``sdpa``, ``eager``).
    """
    if requested in (None, "", "auto"):
        if not getattr(torch_mod, "cuda", None) or not torch_mod.cuda.is_available():
            return "sdpa"
        try:
            major, _minor = torch_mod.cuda.get_device_capability(0)
        except Exception:
            return "sdpa"
        if major >= 8:
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except ImportError:
                return "sdpa"
        # Turing (T4, SM 7.5): FA2 unsupported; SDPA mem-efficient is fine.
        return "sdpa"
    return str(requested)


class InferenceBackend:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> BackendOutput:
        raise NotImplementedError


@dataclass
class TransformersChatBackend(InferenceBackend):
    config: InferenceModelConfig

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError(
                "The transformers backend requires torch and transformers. "
                "Install cloud dependencies with: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        backend_kwargs = dict(self.config.backend_kwargs)
        hf_token = self._resolve_hf_token(backend_kwargs)

        shared_kwargs = {}
        for key in ("cache_dir", "revision", "trust_remote_code", "local_files_only"):
            if key in backend_kwargs:
                shared_kwargs[key] = backend_kwargs[key]
        if hf_token:
            shared_kwargs["token"] = hf_token

        tokenizer_kwargs = dict(shared_kwargs)
        if "use_fast" in backend_kwargs:
            tokenizer_kwargs["use_fast"] = backend_kwargs["use_fast"]

        model_kwargs = dict(shared_kwargs)
        model_kwargs["device_map"] = backend_kwargs.get("device_map", "auto")
        model_kwargs["torch_dtype"] = self._resolve_torch_dtype(backend_kwargs.get("torch_dtype", "auto"))
        for key in ("low_cpu_mem_usage", "load_in_4bit", "load_in_8bit"):
            if key in backend_kwargs:
                model_kwargs[key] = backend_kwargs[key]

        requested_attn = backend_kwargs.get("attn_implementation", "auto")
        attn_impl = resolve_attn_implementation(requested_attn, torch)
        self.attn_implementation = attn_impl
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, **tokenizer_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
        except Exception as exc:
            # FA2 often fails if flash-attn is missing/mismatched; fall back to SDPA.
            if model_kwargs.get("attn_implementation") == "flash_attention_2":
                model_kwargs["attn_implementation"] = "sdpa"
                self.attn_implementation = "sdpa"
                self.model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
            else:
                raise BackendError(
                    f"Failed to load model {self.config.model!r}: {exc}"
                ) from exc
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> BackendOutput:
        if seed is not None:
            self.torch.manual_seed(seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(seed)

        started = time.monotonic()
        inputs = self._build_inputs(system_prompt, user_prompt)
        input_token_count = int(inputs["input_ids"].shape[-1])
        generation_kwargs = self._generation_kwargs(temperature, max_tokens, top_p=top_p)

        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = outputs[0][input_token_count:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.numel())
        return BackendOutput(
            response=response,
            token_count=completion_tokens,
            latency_s=time.monotonic() - started,
            metadata={
                "input_tokens": input_token_count,
                "completion_tokens": completion_tokens,
                "model_id": self.config.model,
                "backend": "transformers",
                "attn_implementation": self.attn_implementation,
            },
        )

    def _build_inputs(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except (AttributeError, TypeError, ValueError):
            prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
            tokenized = self.tokenizer(prompt, return_tensors="pt")

        if not isinstance(tokenized, Mapping):
            tokenized = {"input_ids": tokenized}
        device = self._input_device()
        return {key: value.to(device) for key, value in tokenized.items()}

    def _input_device(self) -> Any:
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for device in device_map.values():
                if device not in {"cpu", "disk"}:
                    if isinstance(device, int):
                        return self.torch.device(f"cuda:{device}" if self.torch.cuda.is_available() else "cpu")
                    return self.torch.device(device)
        return getattr(self.model, "device", self.torch.device("cpu"))

    def _generation_kwargs(self, temperature: float | None, max_tokens: int | None, *, top_p: float | None = None) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": max_tokens or int(self.config.backend_kwargs.get("max_new_tokens", 1024)),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if temperature is not None and temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p if top_p is not None else float(self.config.backend_kwargs.get("top_p", 0.95))
        else:
            kwargs["do_sample"] = False
        for key in ("top_k", "repetition_penalty"):
            if key in self.config.backend_kwargs:
                kwargs[key] = self.config.backend_kwargs[key]
        return kwargs

    def _resolve_torch_dtype(self, value: Any) -> Any:
        if value in (None, "auto"):
            return "auto"
        if isinstance(value, str):
            return getattr(self.torch, value)
        return value

    def _resolve_hf_token(self, backend_kwargs: dict[str, Any]) -> str | None:
        if backend_kwargs.get("token"):
            return str(backend_kwargs["token"])
        token_env = backend_kwargs.get("token_env", "HF_TOKEN")
        return os.environ.get(str(token_env)) if token_env else None


class _OpenAIHTTPMixin:
    """Shared URL/auth/retry helpers for OpenAI-compatible REST backends.

    Subclasses must expose ``config`` (:class:`InferenceModelConfig`), a
    ``default_timeout_s`` float, and an ``endpoint_path`` property naming the
    REST method path relative to the base URL (e.g. ``chat/completions``).
    """

    @property
    def endpoint_path(self) -> str:
        raise NotImplementedError

    def _endpoint_url(self) -> str:
        base_url = self.config.base_url
        if base_url is None and self.config.base_url_env:
            base_url = os.environ.get(self.config.base_url_env)
        if not base_url:
            raise BackendError(
                f"Model {self.config.name} needs base_url or env var {self.config.base_url_env!r}"
            )
        base_url = base_url.rstrip("/")
        if base_url.endswith(f"/{self.endpoint_path}"):
            return base_url
        return f"{base_url}/{self.endpoint_path}"

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **dict(self.config.backend_kwargs.get("headers", {})),
        }
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = self.config.request_timeout_s or self.default_timeout_s
        impersonate = self.config.backend_kwargs.get("impersonate")
        transport = "curl_cffi" if impersonate else "urllib"

        max_retries = int(self.config.backend_kwargs.get("max_retries", 0))
        if max_retries < 0:
            raise BackendError("backend_kwargs.max_retries must be >= 0")
        initial_backoff_s = float(self.config.backend_kwargs.get("retry_initial_backoff_s", 1.0))
        max_backoff_s = float(self.config.backend_kwargs.get("retry_max_backoff_s", 30.0))
        if initial_backoff_s < 0 or max_backoff_s < 0:
            raise BackendError("retry backoff values must be >= 0")

        started = time.monotonic()
        last_error: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                if transport == "curl_cffi":
                    parsed = self._post_json_curl_cffi(body, headers, timeout, impersonate)
                else:
                    parsed = self._post_json_urllib(body, headers, timeout)
                parsed["_latency_s"] = time.monotonic() - started
                return parsed
            except _TransportHTTPError as exc:
                last_error = exc
                message = f"HTTP {exc.code} from {self.config.name}: {exc.error_body[:1000]}"
                retry_after_s = exc.retry_after_s
                retryable = exc.code == 429 or 500 <= exc.code < 600
            except _TransportNetworkError as exc:
                last_error = exc
                message = f"Request failed for {self.config.name}: {exc}"
                retry_after_s = None
                retryable = True

            if not retryable or attempt == max_retries:
                raise BackendError(message) from last_error
            self._sleep_before_retry(
                attempt,
                initial_backoff_s=initial_backoff_s,
                max_backoff_s=max_backoff_s,
                retry_after_s=retry_after_s,
            )

        raise AssertionError("retry loop must return or raise")

    def _post_json_urllib(
        self,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            self._endpoint_url(),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _TransportHTTPError(
                code=exc.code,
                error_body=exc.read().decode("utf-8", errors="replace"),
                retry_after_s=self._retry_after_seconds(exc.headers),
            ) from exc
        except urllib.error.URLError as exc:
            raise _TransportNetworkError(str(exc)) from exc
        except TimeoutError as exc:
            raise _TransportNetworkError(f"timeout after {timeout}s: {exc}") from exc

    def _post_json_curl_cffi(
        self,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
        impersonate: Any,
    ) -> dict[str, Any]:
        try:
            from curl_cffi import requests as crequests  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendError(
                "backend_kwargs.impersonate requires curl_cffi. "
                "Install with: uv pip install curl_cffi"
            ) from exc

        try:
            response = crequests.post(
                self._endpoint_url(),
                content=body,
                headers=headers,
                timeout=timeout,
                impersonate=str(impersonate),
            )
        except Exception as exc:
            # Network-level failures (DNS, TLS, connect, timeout).
            raise _TransportNetworkError(str(exc)) from exc

        if response.status_code >= 400:
            raise _TransportHTTPError(
                code=response.status_code,
                error_body=str(response.text or "")[:1000],
                retry_after_s=self._retry_after_seconds(
                    getattr(response, "headers", None) or {}
                ),
            )
        try:
            return json.loads(response.text)
        except (TypeError, ValueError) as exc:
            raise _TransportNetworkError(f"non-JSON response: {str(response.text or '')[:200]}") from exc

    @staticmethod
    def _retry_after_seconds(headers: Any) -> float | None:
        if headers is None:
            return None
        try:
            value = headers.get("Retry-After")
        except AttributeError:
            return None
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sleep_before_retry(
        attempt: int,
        *,
        initial_backoff_s: float,
        max_backoff_s: float,
        retry_after_s: float | None,
    ) -> None:
        if retry_after_s is not None:
            delay_s = retry_after_s
        else:
            delay_s = min(max_backoff_s, initial_backoff_s * (2**attempt))
            # Avoid synchronized retries when multiple TTC samples are in flight.
            delay_s *= 0.5 + random.random()
        time.sleep(delay_s)

    def _resolve_api_key(self) -> str | None:
        if self.config.api_key is not None:
            return self.config.api_key
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise BackendError(
                    f"Model {self.config.name} needs API key env var {self.config.api_key_env!r}"
                )
            return api_key
        return None


@dataclass
class OpenAICompatibleChatBackend(_OpenAIHTTPMixin, InferenceBackend):
    config: InferenceModelConfig
    default_timeout_s: float = 120.0

    @property
    def endpoint_path(self) -> str:
        return "chat/completions"

    def _chat_completions_url(self) -> str:
        return self._endpoint_url()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> BackendOutput:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None and self.config.backend_kwargs.get("send_seed", False):
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        payload.update(dict(self.config.backend_kwargs.get("extra_body", {})))

        response = self._post_json(payload)
        choices = response.get("choices") or []
        if not choices:
            raise BackendError(f"Backend returned no choices for model {self.config.name}")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content") or first_choice.get("text") or ""
        usage = response.get("usage") or {}
        token_count = int(
            usage.get("completion_tokens")
            or usage.get("total_tokens")
            or max(1, len(str(content).split()))
        )
        return BackendOutput(
            response=str(content),
            token_count=token_count,
            latency_s=float(response.get("_latency_s", 0.0)),
            metadata={"usage": usage, "finish_reason": first_choice.get("finish_reason")},
        )


@dataclass
class OpenAIResponsesBackend(_OpenAIHTTPMixin, InferenceBackend):
    """OpenAI Responses API (``POST /responses``) backend.

    Used by Ramp Router (``api.router.com/v1``), which only exposes
    ``/responses`` (and an Anthropic-compatible ``/messages`` surface):
    ``/chat/completions`` does not exist there.
    """

    config: InferenceModelConfig
    default_timeout_s: float = 120.0

    @property
    def endpoint_path(self) -> str:
        return "responses"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> BackendOutput:
        payload: dict[str, Any] = {"model": self.config.model}
        if system_prompt:
            payload["instructions"] = system_prompt
        payload["input"] = user_prompt
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None and self.config.backend_kwargs.get("send_seed", False):
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        reasoning_effort = self.config.backend_kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": str(reasoning_effort)}
        payload.update(dict(self.config.backend_kwargs.get("extra_body", {})))

        response = self._post_json(payload)
        content = self._extract_text(response)
        if not content:
            raise BackendError(f"Backend returned no output text for model {self.config.name}")
        usage = response.get("usage") or {}
        token_count = int(
            usage.get("output_tokens")
            or usage.get("total_tokens")
            or max(1, len(str(content).split()))
        )
        return BackendOutput(
            response=str(content),
            token_count=token_count,
            latency_s=float(response.get("_latency_s", 0.0)),
            metadata={"usage": usage, "status": response.get("status")},
        )

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Pull the answer text from an OpenAI Responses object.

        Prefers ``output_text`` content blocks inside assistant ``message``
        items, then falls back to top-level text fields. Reasoning items are
        skipped so we return the final answer, not the thinking dump.
        """
        parts: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    walk(item)
                return
            if not isinstance(node, dict):
                return
            node_type = node.get("type")
            if node_type == "output_text" and isinstance(node.get("text"), str):
                parts.append(node["text"])
            elif node_type == "message" or node.get("role") == "assistant":
                walk(node.get("content"))
            elif node_type == "reasoning":
                return  # Skip thinking; only final output matters
            else:
                for key in ("output", "content", "items"):
                    if key in node:
                        walk(node[key])

        walk(response.get("output", response))
        text = "\n".join(parts).strip()
        if text:
            return text
        for key in ("output_text", "text", "content_text"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


def build_backend(config: InferenceModelConfig, *, default_timeout_s: float) -> InferenceBackend:
    if config.backend == "transformers":
        return TransformersChatBackend(config=config)
    if config.backend == "openai_compatible":
        return OpenAICompatibleChatBackend(config=config, default_timeout_s=default_timeout_s)
    if config.backend in ("responses", "responses_api", "openai_responses"):
        return OpenAIResponsesBackend(config=config, default_timeout_s=default_timeout_s)
    raise ValueError(f"Unsupported backend {config.backend!r} for model {config.name}")
