from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .schema import ANSWER_TYPES


@dataclass(frozen=True)
class InferenceExample:
    id: str
    task: str
    question: str
    choices: list[str] | None
    gold_answer: str
    answer_type: str
    source_dataset: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        row_number: int,
        path: str | Path,
        dataset_name: str | None = None,
        task: str | None = None,
        source_dataset: str | None = None,
    ) -> "InferenceExample":
        source_path = Path(path)
        inferred_source = source_dataset or row.get("source_dataset") or dataset_name or source_path.parent.name
        answer_type = str(row.get("answer_type", "text"))
        if answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"Invalid answer_type {answer_type!r} on row {row_number} of {source_path}. "
                f"Expected one of {sorted(ANSWER_TYPES)}"
            )

        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing non-empty question on row {row_number} of {source_path}")

        raw_choices = row.get("choices")
        choices: list[str] | None
        if raw_choices is None:
            choices = None
        elif isinstance(raw_choices, list):
            choices = [str(choice) for choice in raw_choices]
        else:
            choices = [str(raw_choices)]

        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata = {
            **metadata,
            "dataset_name": dataset_name,
            "source_path": str(source_path),
            "row_number": row_number,
        }

        example_id = row.get("id")
        if example_id in (None, ""):
            example_id = f"{inferred_source}_{source_path.stem}_{row_number:06d}"

        return cls(
            id=str(example_id),
            task=str(row.get("task") or task or infer_task(answer_type, inferred_source, source_path)),
            question=question.strip(),
            choices=choices,
            gold_answer=str(row.get("gold_answer", "")).strip(),
            answer_type=answer_type,
            source_dataset=str(inferred_source),
            metadata=metadata,
        )


def infer_task(answer_type: str, source_dataset: str, path: Path) -> str:
    path_text = "/".join(path.parts).lower()
    source = source_dataset.lower()
    if answer_type == "number" or "math" in path_text or "gsm" in source:
        return "math"
    if source in {"naijarc", "yorc"} or "reading" in path_text:
        return "reading_comprehension"
    return "qa"


def load_inference_examples(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    task: str | None = None,
    source_dataset: str | None = None,
    limit: int | None = None,
) -> list[InferenceExample]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [
        InferenceExample.from_row(
            row,
            row_number=index,
            path=path,
            dataset_name=dataset_name,
            task=task,
            source_dataset=source_dataset,
        )
        for index, row in enumerate(rows, start=1)
    ]
