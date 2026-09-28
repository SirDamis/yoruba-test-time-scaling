"""ThinkPRM inference: a generative, long-CoT process reward model.

Unlike a discriminative PRM (which reads a reward head at step separators),
ThinkPRM *generates* a verification chain-of-thought and then answers
``Is the solution correct? Yes/No``. The score for a candidate is the model's
calibrated probability of ``Yes`` given that chain (the "prefix score" of the
paper), optionally averaged over several parallel chains.

The released checkpoints (``launch/ThinkPRM-{1.5B,7B,14B}``) can be used
off-the-shelf or with a LoRA adapter fine-tuned on Yoruba verification chains.
"""

from __future__ import annotations

from typing import Any

from .config import ScoreConfig, resolve_path
from .parse import (
    extract_decision,
    extract_step_labels,
    format_step_scores,
    pad_step_labels,
    truncate_after_last_decision,
)
from .prompts import PREDECISION_STRING, format_verification_prompt
from .steps import render_numbered_solution

AGGREGATIONS = {"prefix", "last", "max", "mean", "min"}


def _resolve_torch_dtype(torch_mod: Any, value: Any) -> Any:
    if value in (None, "auto"):
        return "auto"
    if isinstance(value, str):
        return getattr(torch_mod, value)
    return value


def aggregate_step_scores(step_scores: list[float], aggregation: str) -> float:
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


class ThinkPrmScorer:
    """Score reasoning traces with a ThinkPRM generative verifier."""

    def __init__(self, config: ScoreConfig) -> None:
        if config.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {config.aggregation!r}. Expected one of {sorted(AGGREGATIONS)}"
            )
        if config.n_verifications < 1:
            raise ValueError("n_verifications must be >= 1")
        self.config = config
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def load(self) -> "ThinkPrmScorer":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

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
        model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
        if self.config.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(resolve_path(self.config.adapter_path)))
        model.eval()

        self._resolve_decision_ids(tokenizer)
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        return self

    def _resolve_decision_ids(self, tokenizer: Any) -> None:
        yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
        no_ids = tokenizer.encode(" No", add_special_tokens=False)
        if not yes_ids or not no_ids:
            raise ValueError("Tokenizer cannot encode the ' Yes'/' No' decision tokens.")
        self._yes_id = int(yes_ids[-1])
        self._no_id = int(no_ids[-1])

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

    def _render_prompt(self, question: str, steps: list[str]) -> str:
        solution = render_numbered_solution(steps)
        return format_verification_prompt(
            self._tokenizer, question, solution, self.config.instruction
        )

    def _fit_prompt(self, question: str, steps: list[str]) -> tuple[str, bool]:
        """Render the prompt, dropping leading steps if ``max_input_tokens`` is set."""
        prompt = self._render_prompt(question, steps)
        limit = self.config.max_input_tokens
        if not limit:
            return prompt, False
        input_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(input_ids) <= limit:
            return prompt, False
        kept = list(steps)
        while len(kept) > 1:
            kept = kept[1:]
            prompt = self._render_prompt(question, kept)
            if len(self._tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= limit:
                return prompt, True
        return prompt, True

    def _generate(self, prompts: list[str]) -> list[list[str]]:
        """Generate ``n_verifications`` completions per prompt (list-of-lists)."""
        torch = self._torch
        tokenizer = self._tokenizer
        n = self.config.n_verifications
        do_sample = bool(self.config.temperature and self.config.temperature > 0)
        if not do_sample:
            n = 1

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_width = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p

        with torch.no_grad():
            sequences = self._model.generate(**encoded, **gen_kwargs)
        generated = sequences[:, input_width:]

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        grouped: list[list[str]] = []
        for index in range(len(prompts)):
            grouped.append([str(text) for text in decoded[index * n : (index + 1) * n]])
        return grouped

    def _decision_probabilities(self, contexts: list[str]) -> list[float]:
        """P(Yes) for each ``context`` ending just before the decision token."""
        torch = self._torch
        tokenizer = self._tokenizer
        encoded = tokenizer(
            contexts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits[:, -1, :]
        pair = torch.stack([logits[:, self._yes_id], logits[:, self._no_id]], dim=-1)
        probabilities = torch.softmax(pair / self.config.decision_temperature, dim=-1)
        return [float(value) for value in probabilities[:, 0].cpu().tolist()]

    def score_batch(
        self, questions: list[str], steps_batch: list[list[str]]
    ) -> list[dict[str, Any]]:
        """Score a batch of (question, steps) pairs in one generation pass."""
        if self._model is None:
            self.load()

        prepared: list[tuple[str, list[str], bool]] = [
            self._fit_prompt(question, steps)
            for question, steps in zip(questions, steps_batch)
        ]
        prompts = [prompt for prompt, _, _ in prepared]

        if all(not steps for _, steps, _ in prepared):
            return [self._empty_result(steps) for _, steps, _ in prepared]

        completions = self._generate(prompts)

        results: list[dict[str, Any]] = []
        for (prompt, steps, truncated_input), chain_texts in zip(prepared, completions):
            if not steps:
                results.append(self._empty_result(steps))
                continue

            contexts: list[str] = []
            labels_per_chain: list[list[int]] = []
            any_truncated = bool(truncated_input)
            for text in chain_texts:
                body = truncate_after_last_decision(text)
                decision = extract_decision(text, decision_string=self.config.predecision_string)
                labels = extract_step_labels(body)
                labels_per_chain.append(pad_step_labels(labels, len(steps)))
                contexts.append(self._decision_context(prompt, body))
                if decision is None:
                    any_truncated = True

            probabilities = self._decision_probabilities(contexts)
            prefix_score = sum(probabilities) / len(probabilities) if probabilities else 0.0
            step_labels = labels_per_chain[0] if labels_per_chain else [0] * len(steps)
            step_scores = format_step_scores(step_labels)

            if self.config.aggregation == "prefix":
                score = prefix_score
            else:
                score = aggregate_step_scores(step_scores, self.config.aggregation)

            results.append(
                {
                    "score": float(score),
                    "prefix_score": float(prefix_score),
                    "prefix_scores": [float(value) for value in probabilities],
                    "step_labels": step_labels,
                    "step_scores": step_scores,
                    "n_verifications": len(chain_texts),
                    "outputs": [truncate_after_last_decision(text) for text in chain_texts],
                    "n_steps": len(steps),
                    "truncated": any_truncated,
                }
            )
        return results

    def _decision_context(self, prompt: str, body: str) -> str:
        """Append the pre-decision question so the next token is Yes/No."""
        decorated = f"{prompt}{body}"
        predecision = self.config.predecision_string or PREDECISION_STRING
        if not decorated.rstrip().endswith(predecision.rstrip()):
            decorated = f"{decorated}\n{predecision}"
        return decorated

    def score(self, question: str, steps: list[str]) -> dict[str, Any]:
        return self.score_batch([question], [steps])[0]

    def _empty_result(self, steps: list[str]) -> dict[str, Any]:
        return {
            "score": 0.0,
            "prefix_score": 0.0,
            "prefix_scores": [],
            "step_labels": [0] * len(steps),
            "step_scores": [0.0] * len(steps),
            "n_verifications": 0,
            "outputs": [],
            "n_steps": len(steps),
            "truncated": False,
        }


__all__ = ["AGGREGATIONS", "ThinkPrmScorer", "aggregate_step_scores"]
