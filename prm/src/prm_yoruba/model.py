"""PRM inference.

Two interfaces are supported:

* ``qwen2.5-math-prm`` (default): the official ``Qwen/Qwen2.5-Math-PRM-*``
  checkpoints (``Qwen2ForProcessRewardModel``). Steps are joined with the
  ``<extra_0>`` special token and the 2-class reward head is read at those
  positions — see the model card.
* ``token_classifier``: the reference repo's ``+``/``-`` LM-token method used
  with ``Qwen2.5-Math-7B-Instruct`` (kept for compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ScoreConfig, resolve_path
from .steps import (
    MISTRAL_BAD_TOKEN,
    MISTRAL_GOOD_TOKEN,
    MISTRAL_STEP_TAG,
    QWEN_BAD_TOKEN,
    QWEN_GOOD_TOKEN,
    QWEN_PRM_STEP_SEP,
    QWEN_STEP_TAG,
    render_scoring_input,
    render_sep_joined,
)

AGGREGATIONS = {"last", "max", "mean", "min"}
INTERFACES = {"qwen2.5-math-prm", "token_classifier"}


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
    """Derive the good/bad/step-tag ids for a legacy ``token_classifier`` model."""
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


def resolve_sep_id(tokenizer: Any, sep: str = QWEN_PRM_STEP_SEP) -> int:
    """Resolve the ``<extra_0>`` step-separator id (official PRM interface)."""
    ids = tokenizer.encode(sep)
    if not ids:
        raise ValueError(f"Step separator {sep!r} does not tokenize.")
    return int(ids[0])


def ensure_pad_token_id(model: Any, tokenizer: Any) -> None:
    """Backfill ``model.config.pad_token_id`` when the checkpoint config omits it.

    ``Qwen2RMConfig``/``Qwen2Model`` read ``config.pad_token_id``; the released
    ``config.json`` does not define it, so some transformers versions raise
    ``AttributeError``. We also pass ``pad_token_id`` at load time; this is a
    belt-and-braces fix for anything that reads it afterwards.
    """
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id


def load_official_model(
    model_name: str,
    model_kwargs: dict[str, Any],
    tokenizer: Any,
    *,
    trust_remote_code: bool,
) -> Any:
    """Load ``Qwen2ForProcessRewardModel``, working around the missing ``pad_token_id``.

    ``Qwen2RMConfig``/``Qwen2Model`` read ``config.pad_token_id``, but the released
    ``config.json`` omits it. Passing ``pad_token_id`` as a ``from_pretrained``
    kwarg does **not** help here (the remote config ignores it, so it reaches the
    model constructor and raises ``TypeError``); instead we build the remote
    config explicitly, set the attribute, and load with ``config=config``.
    """
    from transformers import AutoConfig, AutoModel

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = pad_id
    # The remote modeling code calls ``DynamicCache.from_legacy_cache`` (removed in
    # newer transformers) whenever ``use_cache`` is on. We never use the KV cache,
    # so disable it both on the config and on the loaded model.
    config.use_cache = False
    model = AutoModel.from_pretrained(model_name, config=config, **model_kwargs)
    ensure_pad_token_id(model, tokenizer)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return model


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
    """Score candidate reasoning traces with a process reward model."""

    def __init__(self, config: ScoreConfig) -> None:
        if config.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {config.aggregation!r}. Expected one of {sorted(AGGREGATIONS)}"
            )
        if config.prm_interface not in INTERFACES:
            raise ValueError(
                f"Unknown prm_interface {config.prm_interface!r}. Expected one of {sorted(INTERFACES)}"
            )
        self.config = config
        self.tokens: PrmTokens | None = None
        self.sep_id: int | None = None
        self._model = None
        self._tokenizer = None
        self._torch = None

    def load(self) -> "PrmScorer":
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
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

        if self.config.prm_interface == "qwen2.5-math-prm":
            model = load_official_model(
                self.config.model,
                model_kwargs,
                tokenizer,
                trust_remote_code=self.config.trust_remote_code,
            )
            self.sep_id = resolve_sep_id(tokenizer, self.config.step_sep_token)
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
            self.tokens = resolve_prm_tokens(tokenizer, self.config.model)

        if self.config.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(resolve_path(self.config.adapter_path)))
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
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

    def _official_text(self, question: str, steps: list[str]) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": render_sep_joined(steps, self.config.step_sep_token)},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _score_official(self, question: str, steps: list[str]) -> dict[str, Any]:
        limit = self.config.max_input_tokens
        if limit is None:
            limit = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)

        input_ids = self._tokenizer.encode(self._official_text(question, steps))
        truncated = False
        if limit and len(input_ids) > limit:
            truncated = True
            kept = list(steps)
            while len(kept) > 1:
                kept = kept[1:]
                input_ids = self._tokenizer.encode(self._official_text(question, kept))
                if len(input_ids) <= limit:
                    break
            else:
                input_ids = input_ids[:limit]

        tensor = self._torch.tensor([input_ids], device=self._input_device())
        with self._torch.no_grad():
            outputs = self._model(tensor, use_cache=False)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            logits = outputs[0]  # [1, seq, 2]
        probabilities = logits.softmax(dim=-1)[0]
        selected = probabilities[tensor[0] == self.sep_id][:, 1]
        scores = [float(value) for value in selected.cpu().tolist()]
        return {
            "step_scores": scores,
            "n_scored_steps": len(scores),
            "truncated": truncated,
        }

    def _score_token_classifier(self, question: str, steps: list[str]) -> dict[str, Any]:
        assert self.tokens is not None
        limit = self.config.max_input_tokens
        if limit is None:
            limit = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)

        input_ids = self._tokenizer.encode(
            render_scoring_input(question, steps, self.tokens.step_tag)
        )
        truncated = False
        if limit and len(input_ids) > limit:
            truncated = True
            kept = list(steps)
            while len(kept) > 1:
                kept = kept[1:]
                input_ids = self._tokenizer.encode(
                    render_scoring_input(question, kept, self.tokens.step_tag)
                )
                if len(input_ids) <= limit:
                    break
            else:
                input_ids = input_ids[:limit]

        tensor = self._torch.tensor([input_ids], device=self._input_device())
        with self._torch.no_grad():
            logits = self._model(tensor).logits[:, :, self.tokens.candidate_tokens]
            probs = logits.softmax(dim=-1)[:, :, 0]
            step_scores = probs[tensor == self.tokens.step_tag_id]
        scores = [float(value) for value in step_scores.cpu().tolist()]
        return {"step_scores": scores, "n_scored_steps": len(scores), "truncated": truncated}

    def score(self, question: str, steps: list[str]) -> dict[str, Any]:
        """Return per-step good-token probabilities and their aggregations."""
        if self._model is None:
            self.load()
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

        if self.config.prm_interface == "qwen2.5-math-prm":
            result = self._score_official(question, steps)
        else:
            result = self._score_token_classifier(question, steps)
        scores = result["step_scores"]
        return {
            "step_scores": scores,
            "last": aggregate_scores(scores, "last"),
            "max": aggregate_scores(scores, "max"),
            "min": aggregate_scores(scores, "min"),
            "mean": aggregate_scores(scores, "mean"),
            "n_scored_steps": result["n_scored_steps"],
            "score": aggregate_scores(scores, self.config.aggregation),
            "truncated": result["truncated"],
        }
