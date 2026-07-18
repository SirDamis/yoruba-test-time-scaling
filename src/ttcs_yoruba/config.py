from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import read_json


SUPPORTED_PROMPT_STYLES = {
    "english_cot",
    "yoruba_cot",
    "translate_pivot",
    "best_of_n_cot",
}
SUPPORTED_SELECTIONS = {"first", "majority_vote"}

# Default reasoning-language tags for experiment logging / filtering.
PROMPT_STYLE_REASONING_LANGUAGE = {
    "yoruba_cot": "yo",
    "english_cot": "en",
    "best_of_n_cot": "en",
    "translate_pivot": "en_pivot",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    task: str | None = None
    source_dataset: str | None = None
    limit: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DatasetConfig":
        limit = row.get("limit")
        return cls(
            name=str(row["name"]),
            path=Path(row["path"]),
            task=None if row.get("task") is None else str(row["task"]),
            source_dataset=None if row.get("source_dataset") is None else str(row["source_dataset"]),
            limit=None if limit is None else int(limit),
        )


@dataclass(frozen=True)
class InferenceModelConfig:
    name: str
    backend: str
    model: str
    size_label: str
    base_url_env: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    cost_per_1k_tokens: float = 0.0
    request_timeout_s: float | None = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "InferenceModelConfig":
        return cls(
            name=str(row["name"]),
            backend=str(row.get("backend", "transformers")),
            model=str(row.get("model", row["name"])),
            size_label=str(row.get("size_label", "unknown")),
            base_url_env=None if row.get("base_url_env") is None else str(row["base_url_env"]),
            api_key_env=None if row.get("api_key_env") is None else str(row["api_key_env"]),
            base_url=None if row.get("base_url") is None else str(row["base_url"]),
            api_key=None if row.get("api_key") is None else str(row["api_key"]),
            cost_per_1k_tokens=float(row.get("cost_per_1k_tokens", 0.0)),
            request_timeout_s=None
            if row.get("request_timeout_s") is None
            else float(row["request_timeout_s"]),
            backend_kwargs=dict(row.get("backend_kwargs", {})),
        )


@dataclass(frozen=True)
class InferenceMethodConfig:
    name: str
    prompt_style: str
    selection: str
    n: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_language: str = "unknown"
    # When set, this method is part of a nested N-sweep: sample once at max N, slice for each k.
    nested_group_id: str | None = None
    nested_max_n: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "InferenceMethodConfig":
        prompt_style = str(row["prompt_style"])
        if prompt_style not in SUPPORTED_PROMPT_STYLES:
            raise ValueError(
                f"Unsupported prompt_style {prompt_style!r}. "
                f"Expected one of {sorted(SUPPORTED_PROMPT_STYLES)}"
            )

        selection = str(row.get("selection", "first"))
        if selection not in SUPPORTED_SELECTIONS:
            raise ValueError(
                f"Unsupported selection {selection!r}. "
                f"Expected one of {sorted(SUPPORTED_SELECTIONS)}"
            )

        n = int(row.get("n", 1))
        if n < 1:
            raise ValueError(f"Method {row.get('name')!r} has invalid n={n}")

        reasoning_language = row.get("reasoning_language")
        if reasoning_language is None:
            reasoning_language = PROMPT_STYLE_REASONING_LANGUAGE.get(prompt_style, "en")

        nested_group_id = row.get("nested_group_id")
        nested_max_n = row.get("nested_max_n")
        return cls(
            name=str(row["name"]),
            prompt_style=prompt_style,
            selection=selection,
            n=n,
            temperature=None if row.get("temperature") is None else float(row["temperature"]),
            max_tokens=None if row.get("max_tokens") is None else int(row["max_tokens"]),
            reasoning_language=str(reasoning_language),
            nested_group_id=None if nested_group_id is None else str(nested_group_id),
            nested_max_n=None if nested_max_n is None else int(nested_max_n),
        )


@dataclass(frozen=True)
class InferenceRunConfig:
    run_id: str
    output_dir: Path
    datasets: list[DatasetConfig]
    models: list[InferenceModelConfig]
    methods: list[InferenceMethodConfig]
    seed: int | None = None
    default_request_timeout_s: float = 120.0
    continue_on_error: bool = False
    # Concurrent in-flight generations (useful for local vLLM). HF Transformers
    # is clamped to 1 in the pipeline (model generate is not thread-safe).
    max_concurrent: int = 1

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "InferenceRunConfig":
        datasets = [DatasetConfig.from_dict(item) for item in row.get("datasets", [])]
        models = [InferenceModelConfig.from_dict(item) for item in row.get("models", [])]
        methods = expand_method_configs(row.get("methods", []))
        if not datasets:
            raise ValueError("Inference config must define at least one dataset")
        if not models:
            raise ValueError("Inference config must define at least one model")
        if not methods:
            raise ValueError("Inference config must define at least one method")

        seed = row.get("seed")
        max_concurrent = int(row.get("max_concurrent", 1))
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        return cls(
            run_id=str(row.get("run_id", "yoruba_ttc_cloud")),
            output_dir=Path(row.get("output_dir", "runs")),
            datasets=datasets,
            models=models,
            methods=methods,
            seed=None if seed is None else int(seed),
            default_request_timeout_s=float(row.get("default_request_timeout_s", 120.0)),
            continue_on_error=bool(row.get("continue_on_error", False)),
            max_concurrent=max_concurrent,
        )


def expand_method_configs(rows: list[dict[str, Any]]) -> list[InferenceMethodConfig]:
    """Expand method rows, including TTC ``n_values`` sweeps.

    Optional keys for E2 sweeps:
    - ``temperature_n1``: temperature used when ``n == 1`` (default: keep ``temperature``)
    - ``greedy_n1``: if true, force ``temperature=0`` and ``selection=first`` for ``n == 1``.
      With ``nested_n``, N=1 is a **separate true-greedy decode** (not first-of-pool);
      N>1 still share one stochastic pool of size max(N).
    - ``nested_n`` / ``nested``: if true with multi ``n_values``, sample once at max N and
      evaluate nested prefixes for N>1 (saves compute vs independent per-N runs)
    """
    methods: list[InferenceMethodConfig] = []
    for row in rows:
        n_values = row.get("n_values")
        if n_values is None:
            n_values = [row.get("n", 1)]
        if not isinstance(n_values, list):
            raise ValueError(f"Method {row.get('name')!r} has non-list n_values")

        base_name = str(row["name"])
        multi = len(n_values) > 1
        nested = bool(row.get("nested_n", row.get("nested", False))) and multi
        int_ns = [int(v) for v in n_values]
        nested_max_n = max(int_ns) if nested else None
        nested_group_id = base_name if nested else None

        for n_value in int_ns:
            concrete = dict(row)
            concrete.pop("n_values", None)
            concrete.pop("temperature_n1", None)
            concrete.pop("greedy_n1", None)
            concrete.pop("nested_n", None)
            concrete.pop("nested", None)
            n = int(n_value)
            concrete["n"] = n
            concrete["name"] = f"{base_name}_n{n}" if multi else base_name
            if nested_group_id is not None:
                concrete["nested_group_id"] = nested_group_id
                concrete["nested_max_n"] = nested_max_n

            if n == 1:
                if "temperature_n1" in row:
                    concrete["temperature"] = row["temperature_n1"]
                if bool(row.get("greedy_n1", multi)):
                    # True greedy N=1 baseline (temp 0, selection=first). With nested_n the
                    # pipeline generates this decode separately from the stochastic pool.
                    concrete["temperature"] = (
                        float(row["temperature_n1"]) if "temperature_n1" in row else 0.0
                    )
                    concrete["selection"] = "first"

            methods.append(InferenceMethodConfig.from_dict(concrete))
    return methods


def load_inference_run_config(path: str | Path) -> InferenceRunConfig:
    return InferenceRunConfig.from_dict(read_json(path))
