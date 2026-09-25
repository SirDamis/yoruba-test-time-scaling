"""Discriminative PRM inference (Qwen2.5-Math-PRM / Math-Shepherd style).

Reuses the exact token contract from the reference implementation: a step tag
marks each boundary and the model's probability of the ``+`` token at those
positions is the per-step score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ScoreConfig, resolve_path
from .steps import (
    MISTRAL_BAD_TOKEN,
    MISTRAL_GOOD_TOKEN,
    MISTRAL_STEP_TAG,
    QWEN_BAD_TOKEN,
    QWEN_GOOD_TOKEN,
    QWEN_STEP_TAG,
    render_scoring_input,
)

AGGREGATIONS = {"last", "max", "mean", "min"}


@dataclass(frozen=True)
class PrmTokens:
    good_token: str
    bad_token: str
    step_tag: str
    good_id: int
    bad_id: int
    step_tag_id: int
    candidate_tokens: list[int]


def resolve_prm_tokens(tokenizer: Any, model_name: str) -> PrmTokens:
    """Derive the good/bad/step-tag ids for a tokenizer.

    Mirrors ``prm/reference/ft.py``: Mistral-SFT uses ``ки`` plus bare ``+``/``-``;
    Qwen/Llama (including Qwen2.5-Math-PRM) use a five-newline step tag with
    leading-space ``+``/``-``. Asserts guard tokenizer drift.
    """
    lowered = (model_name or "").lower()
    if "mistral-7b-sft" in lowered:
        good_token, bad_token, step_tag = MISTRAL_GOOD_TOKEN, MISTRAL_BAD_TOKEN, MISTRAL_STEP_TAG
        candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")
    else:
        good_token, bad_token, step_tag = QWEN_GOOD_TOKEN, QWEN_BAD_TOKEN, QWEN_STEP_TAG
        candidate_tokens = tokenizer.encode(f"{good_token}{bad_token}")
    if len(candidate_tokens) > 2:
        candidate_tokens = candidate_tokens[1:]
    if len(candidate_tokens) != 2:
        raise ValueError(
            f"Expected exactly two candidate tokens for {model_name!r}, got {candidate_tokens}."
        )
    step_ids = tokenizer.encode(step_tag)
    if not step_ids:
        raise ValueError(f"Step tag {step_tag!r} does not tokenize for {model_name!r}.")
    return PrmTokens(
        good_token=good_token,
        bad_token=bad_token,
        step_tag=step_tag,
        good_id=int(candidate_tokens[0]),
        bad_id=int(candidate_tokens[1]),
        step_tag_id=int(step_ids[-1]),
        candidate_tokens=[int(t) for t in candidate_tokens],
    )


def _resolve_torch_dtype(torch_mod: Any, value: Any) -> Any:
    if value in (None, "auto"):
        return "auto"
    if isinstance(value, str):
        return getattr(torch_mod, value)
    return value


def aggregate_scores(step_scores: list[float], aggregation: str) -> float:
    if not step_scores:
        return 0.0
    if aggregation == "last":
        return float(step_scores[-1])
    if aggregation == "max":
        return float(max(step_scores))
    if aggregation == "min":
        return float(min(step_scores))
    if aggregation == "mean":
        return float(sum(step_scores) / len(step_scores))
    raise ValueError(f"Unknown aggregation {aggregation!r}. Expected one of {sorted(AGGREGATIONS)}")


class PrmScorer:
    """Score candidate reasoning traces with a discriminative PRM."""

    def __init__(self, config: ScoreConfig) -> None:
        if config.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {config.aggregation!r}. Expected one of {sorted(AGGREGATIONS)}"
            )
        self.config = config
        self.tokens: PrmTokens | None = None
        self._model = None
        self._tokenizer = None
        self._torch = None

    def load(self) -> "PrmScorer":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "torch_dtype": _resolve_torch_dtype(torch, self.config.torch_dtype),
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
        if self.config.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(
                self._model, str(resolve_path(self.config.adapter_path))
            )
        self._model.eval()
        self._torch = torch
        self.tokens = resolve_prm_tokens(self._tokenizer, self.config.model)
        return self

    def _input_device(self) -> Any:
        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for device in device_map.values():
                if device not in {"cpu", "disk"}:
                    if isinstance(device, int):
                        return self._torch.device(
                            f"cuda:{device}" if self._torch.cuda.is_available() else "cpu"
                        )
                    return self._torch.device(device)
        return getattr(self._model, "device", self._torch.device("cpu"))

    def score(self, question: str, steps: list[str]) -> dict[str, Any]:
        """Return per-step good-token probabilities and their aggregations."""
        if self._model is None:
            self.load()
        assert self.tokens is not None
        if not steps:
            return {
                "step_scores": [],
                "last": 0.0,
                "max": 0.0,
                "min": 0.0,
                "mean": 0.0,
                "n_scored_steps": 0,
                "score": 0.0,
                "truncated": False,
            }

        limit = self.config.max_input_tokens
        if limit is None:
            limit = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)

        text = render_scoring_input(question, steps, self.tokens.step_tag)
        input_ids = self._tokenizer.encode(text)
        truncated = False
        if limit and len(input_ids) > limit:
            truncated = True
            # Drop leading steps (keeping the question and the tail) so the final
            # step survives for ``last`` aggregation; hard-truncate as a last resort.
            kept = list(steps)
            while len(kept) > 1:
                kept = kept[1:]
                input_ids = self._tokenizer.encode(
                    render_scoring_input(question, kept, self.tokens.step_tag)
                )
                if len(input_ids) <= limit:
                    break
            else:
                input_ids = self._tokenizer.encode(
                    render_scoring_input(question, kept, self.tokens.step_tag)
                )[:limit]
        tensor = self._torch.tensor([input_ids], device=self._input_device())
        with self._torch.no_grad():
            logits = self._model(tensor).logits[:, :, self.tokens.candidate_tokens]
            probs = logits.softmax(dim=-1)[:, :, 0]
            step_scores = probs[tensor == self.tokens.step_tag_id]
        scores = [float(value) for value in step_scores.cpu().tolist()]
        return {
            "step_scores": scores,
            "last": aggregate_scores(scores, "last"),
            "max": aggregate_scores(scores, "max"),
            "min": aggregate_scores(scores, "min"),
            "mean": aggregate_scores(scores, "mean"),
            "n_scored_steps": len(scores),
            "score": aggregate_scores(scores, self.config.aggregation),
            "truncated": truncated,
        }
