from __future__ import annotations

import json
import os
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
    ) -> BackendOutput:
        if seed is not None:
            self.torch.manual_seed(seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(seed)

        started = time.monotonic()
        inputs = self._build_inputs(system_prompt, user_prompt)
        input_token_count = int(inputs["input_ids"].shape[-1])
        generation_kwargs = self._generation_kwargs(temperature, max_tokens)

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

    def _generation_kwargs(self, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": max_tokens or int(self.config.backend_kwargs.get("max_new_tokens", 512)),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if temperature is not None and temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(self.config.backend_kwargs.get("top_p", 0.95))
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


@dataclass
class OpenAICompatibleChatBackend(InferenceBackend):
    config: InferenceModelConfig
    default_timeout_s: float = 120.0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None = None,
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

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **dict(self.config.backend_kwargs.get("headers", {})),
        }
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            self._chat_completions_url(),
            data=body,
            headers=headers,
            method="POST",
        )
        timeout = self.config.request_timeout_s or self.default_timeout_s
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise BackendError(f"HTTP {exc.code} from {self.config.name}: {error_body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"Request failed for {self.config.name}: {exc}") from exc
        parsed["_latency_s"] = time.monotonic() - started
        return parsed

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url
        if base_url is None and self.config.base_url_env:
            base_url = os.environ.get(self.config.base_url_env)
        if not base_url:
            raise BackendError(
                f"Model {self.config.name} needs base_url or env var {self.config.base_url_env!r}"
            )
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

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


def build_backend(config: InferenceModelConfig, *, default_timeout_s: float) -> InferenceBackend:
    if config.backend == "transformers":
        return TransformersChatBackend(config=config)
    if config.backend == "openai_compatible":
        return OpenAICompatibleChatBackend(config=config, default_timeout_s=default_timeout_s)
    raise ValueError(f"Unsupported backend {config.backend!r} for model {config.name}")
