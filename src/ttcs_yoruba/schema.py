from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ANSWER_TYPES = {"choice", "number", "text", "freeform", "instruction"}
REQUIRED_ITEM_FIELDS = {
    "id",
    "task",
    "language",
    "question",
    "choices",
    "gold_answer",
    "answer_type",
    "source_dataset",
    "requires_yoruba_output",
    "metadata",
}


class ValidationError(ValueError):
    """Raised when project data does not match the normalized schema."""


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    task: str
    language: str
    question: str
    choices: list[str] | None
    gold_answer: str
    answer_type: str
    source_dataset: str
    requires_yoruba_output: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any], row_number: int | None = None) -> "BenchmarkItem":
        location = f" on row {row_number}" if row_number is not None else ""
        missing = REQUIRED_ITEM_FIELDS - set(row)
        if missing:
            raise ValidationError(f"Missing required fields{location}: {sorted(missing)}")

        if row["language"] != "yo":
            raise ValidationError(
                f"Paper 1 is Yoruba-only; expected language='yo'{location}, got {row['language']!r}"
            )
        if row["answer_type"] not in ANSWER_TYPES:
            raise ValidationError(
                f"Invalid answer_type{location}: {row['answer_type']!r}. Expected one of {sorted(ANSWER_TYPES)}"
            )
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValidationError(f"Question must be a non-empty string{location}")
        if not isinstance(row["gold_answer"], str) or not row["gold_answer"].strip():
            raise ValidationError(f"gold_answer must be a non-empty string{location}")
        if row["choices"] is not None:
            if not isinstance(row["choices"], list) or not all(isinstance(choice, str) for choice in row["choices"]):
                raise ValidationError(f"choices must be null or a list of strings{location}")
        if not isinstance(row["metadata"], dict):
            raise ValidationError(f"metadata must be an object{location}")

        return cls(
            id=str(row["id"]),
            task=str(row["task"]),
            language="yo",
            question=row["question"].strip(),
            choices=row["choices"],
            gold_answer=row["gold_answer"].strip(),
            answer_type=row["answer_type"],
            source_dataset=str(row["source_dataset"]),
            requires_yoruba_output=bool(row["requires_yoruba_output"]),
            metadata=dict(row["metadata"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str
    size_label: str = "unknown"
    cost_per_1k_tokens: float = 0.0
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ModelConfig":
        return cls(
            name=str(row["name"]),
            backend=str(row.get("backend", "mock")),
            size_label=str(row.get("size_label", "unknown")),
            cost_per_1k_tokens=float(row.get("cost_per_1k_tokens", 0.0)),
            backend_kwargs=dict(row.get("backend_kwargs", {})),
        )


@dataclass(frozen=True)
class MethodConfig:
    name: str
    prompt_style: str
    selection: str
    n: int
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MethodConfig":
        n = int(row.get("n", 1))
        if n < 1:
            raise ValidationError(f"Method {row.get('name')!r} has invalid n={n}")
        return cls(
            name=str(row["name"]),
            prompt_style=str(row["prompt_style"]),
            selection=str(row.get("selection", "first")),
            n=n,
            temperature=None if "temperature" not in row else float(row["temperature"]),
            max_tokens=None if "max_tokens" not in row else int(row["max_tokens"]),
        )


@dataclass(frozen=True)
class BackendOutput:
    response: str
    token_count: int
    latency_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRecord:
    run_id: str
    example_id: str
    task: str
    source_dataset: str
    model: str
    model_size_label: str
    method: str
    prompt_style: str
    selection: str
    sample_index: int
    n: int
    reasoning_language: str
    prompt: str
    response: str
    extracted_answer: str
    token_count: int
    latency_s: float
    estimated_cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
