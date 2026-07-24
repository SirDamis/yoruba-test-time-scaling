from __future__ import annotations

import hashlib
import ast
import csv
import json
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from .io_utils import write_json, write_jsonl
from .schema import BenchmarkItem, ValidationError


SOURCE_DEFAULTS: dict[str, dict[str, str]] = {
    "afrimgsm": {"task": "math", "answer_type": "number"},
    "afrimgsm_translate": {"task": "math", "answer_type": "number"},
    "afrimmlu": {"task": "qa", "answer_type": "choice"},
    "afrimmlu_translate": {"task": "qa", "answer_type": "choice"},
    "afriqa": {"task": "qa", "answer_type": "text"},
    "naijarc": {"task": "reading_comprehension", "answer_type": "text"},
}

QUESTION_KEYS = ["question", "query", "prompt", "input", "instruction"]
GOLD_KEYS = ["gold_answer", "answer", "target", "label", "output", "correct_answer"]
CHOICE_KEYS = ["choices", "options", "multiple_choice_targets"]
LANGUAGE_KEYS = ["language", "lang", "locale"]
YORUBA_CODES = {"yo", "yor", "yoruba", "Yoruba", "YOR", "YO"}
DOWNLOAD_OUTPUT_FIELDS = ("answer_type", "choices", "gold_answer", "question")


@dataclass(frozen=True)
class HFDatasetSpec:
    key: str
    hf_id: str
    config: str
    group: str
    default_splits: tuple[str, ...]
    trust_remote_code: bool = False
    data_files: dict[str, str] | None = None

    @property
    def output_dir(self) -> Path:
        return Path("data") / "normalized" / self.group / self.key


HF_DATASET_REGISTRY: dict[str, HFDatasetSpec] = {
    "afrimmlu": HFDatasetSpec(
        key="afrimmlu",
        hf_id="masakhane/afrimmlu",
        config="yor",
        group="question-answering",
        default_splits=("validation", "dev", "test"),
        data_files={
            "validation": "https://huggingface.co/datasets/masakhane/afrimmlu/resolve/main/data/yor/val.tsv",
            "dev": "https://huggingface.co/datasets/masakhane/afrimmlu/resolve/main/data/yor/dev.tsv",
            "test": "https://huggingface.co/datasets/masakhane/afrimmlu/resolve/main/data/yor/test.tsv",
        },
    ),
    "afrimmlu_translate": HFDatasetSpec(
        key="afrimmlu_translate",
        hf_id="masakhane/afrimmlu-translate-test",
        config="yor",
        group="question-answering",
        default_splits=("test",),
        data_files={
            "test": "https://huggingface.co/datasets/masakhane/afrimmlu-translate-test/resolve/main/data/yor/test.tsv",
        },
    ),
    "afriqa": HFDatasetSpec(
        key="afriqa",
        hf_id="masakhane/afriqa",
        config="yor",
        group="question-answering",
        default_splits=("train", "validation", "test"),
        trust_remote_code=True,
        data_files={
            "train": "https://github.com/masakhane-io/afriqa/raw/main/data/queries/yor/queries.afriqa.yor.en.train.json",
            "validation": "https://github.com/masakhane-io/afriqa/raw/main/data/queries/yor/queries.afriqa.yor.en.dev.json",
            "test": "https://github.com/masakhane-io/afriqa/raw/main/data/queries/yor/queries.afriqa.yor.en.test.json",
        },
    ),
    "naijarc": HFDatasetSpec(
        key="naijarc",
        hf_id="aremuadeolajr/NaijaRC",
        config="yor",
        group="question-answering",
        default_splits=("train", "validation", "test"),
        data_files={
            "train": "https://huggingface.co/datasets/aremuadeolajr/NaijaRC/resolve/main/yor/train.csv",
            "validation": "https://huggingface.co/datasets/aremuadeolajr/NaijaRC/resolve/main/yor/dev.csv",
            "test": "https://huggingface.co/datasets/aremuadeolajr/NaijaRC/resolve/main/yor/test.csv",
        },
    ),
    "afrimgsm": HFDatasetSpec(
        key="afrimgsm",
        hf_id="masakhane/afrimgsm",
        config="yor",
        group="math-reasoning",
        default_splits=("train", "test"),
        data_files={
            "train": "https://huggingface.co/datasets/masakhane/afrimgsm/resolve/main/data/yor/dev.tsv",
            "test": "https://huggingface.co/datasets/masakhane/afrimgsm/resolve/main/data/yor/test.tsv",
        },
    ),
    "afrimgsm_translate": HFDatasetSpec(
        key="afrimgsm_translate",
        hf_id="masakhane/afrimgsm-translate-test",
        config="yor",
        group="math-reasoning",
        default_splits=("test",),
        data_files={
            "test": "https://huggingface.co/datasets/masakhane/afrimgsm-translate-test/resolve/main/data/yor/test.tsv",
        },
    ),
}


def stable_id(source_dataset: str, raw: dict[str, Any]) -> str:
    explicit_id = raw.get("id") or raw.get("uid") or raw.get("example_id")
    if explicit_id:
        split = raw.get("split")
        if split not in (None, ""):
            return f"{source_dataset}_{split}_{explicit_id}"
        return f"{source_dataset}_{explicit_id}"
    digest = hashlib.sha1(repr(sorted(raw.items())).encode("utf-8")).hexdigest()[:12]
    return f"{source_dataset}_{digest}"


def first_present(raw: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return None


def coerce_choices(raw_choices: Any) -> list[str] | None:
    if raw_choices is None or raw_choices == "":
        return None
    if isinstance(raw_choices, str):
        stripped = raw_choices.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(choice) for choice in parsed]
            except (SyntaxError, ValueError):
                pass
        return [choice.strip() for choice in stripped.split("||") if choice.strip()]
    if isinstance(raw_choices, dict):
        return [f"{key}. {value}" for key, value in raw_choices.items()]
    if isinstance(raw_choices, (list, tuple)):
        return [str(choice) for choice in raw_choices]
    return [str(raw_choices)]


def choice_list_from_columns(raw: dict[str, Any]) -> list[str] | None:
    labels = ["A", "B", "C", "D"]
    values = []
    for label in labels:
        for key in (f"options_{label}", f"option_{label}", label, label.lower()):
            if key in raw and raw[key] not in (None, ""):
                values.append(f"{label}. {raw[key]}")
                break
    return values or None


def normalize_answer_label(answer: Any) -> str:
    if answer is None:
        return ""
    if isinstance(answer, int):
        return chr(ord("A") + answer) if 0 <= answer <= 25 else str(answer)
    value = str(answer).strip()
    if value.isdigit():
        number = int(value)
        return chr(ord("A") + number) if 0 <= number <= 3 else value
    return value.upper() if len(value) == 1 else value


def is_yoruba_record(raw: dict[str, Any], spec: HFDatasetSpec) -> bool:
    language = first_present(raw, LANGUAGE_KEYS)
    if language is None:
        return spec.config in {"yo", "yor"}
    return str(language).strip() in YORUBA_CODES


def with_split_metadata(item: BenchmarkItem, split: str, raw: dict[str, Any], spec: HFDatasetSpec) -> BenchmarkItem:
    metadata = dict(item.metadata)
    metadata.update(
        {
            "split": split,
            "hf_id": spec.hf_id,
            "hf_config": spec.config,
            "hf_group": spec.group,
            "raw_keys": sorted(raw.keys()),
        }
    )
    return BenchmarkItem(
        id=item.id,
        task=item.task,
        language=item.language,
        question=item.question,
        choices=item.choices,
        gold_answer=item.gold_answer,
        answer_type=item.answer_type,
        source_dataset=item.source_dataset,
        requires_yoruba_output=item.requires_yoruba_output,
        metadata=metadata,
    )


def compact_download_row(item: BenchmarkItem) -> dict[str, Any]:
    row = item.to_dict()
    return {field: row[field] for field in DOWNLOAD_OUTPUT_FIELDS}


def normalize_hf_record(source_dataset: str, raw: dict[str, Any], split: str, spec: HFDatasetSpec) -> BenchmarkItem:
    if source_dataset in ("afrimmlu", "afrimmlu_translate"):
        choices = coerce_choices(raw.get("choices"))
        row = {
            "id": stable_id(source_dataset, {**raw, "split": split}),
            "task": "qa",
            "language": "yo",
            "question": raw.get("question"),
            "choices": choices,
            "gold_answer": normalize_answer_label(raw.get("answer")),
            "answer_type": "choice",
            "source_dataset": source_dataset,
            "requires_yoruba_output": True,
            "metadata": {"subject": raw.get("subject")},
        }
    elif source_dataset == "afriqa":
        answers = raw.get("answers")
        row = {
            "id": stable_id(source_dataset, {**raw, "split": split}),
            "task": "qa",
            "language": "yo",
            "question": raw.get("question"),
            "choices": None,
            "gold_answer": first_answer(answers),
            "answer_type": "text",
            "source_dataset": source_dataset,
            "requires_yoruba_output": True,
            "metadata": {
                "answers": parse_answer_list(answers),
                "translated_question": raw.get("translated_question"),
                "translated_answer": raw.get("translated_answer"),
                "translation_type": raw.get("translation_type"),
            },
        }
    elif source_dataset == "naijarc":
        story = raw.get("story", "")
        question = raw.get("question", "")
        row = {
            "id": stable_id(source_dataset, {**raw, "split": split}),
            "task": "reading_comprehension",
            "language": "yo",
            "question": f"Àyọkà:\n{story}\n\nÌbéèrè:\n{question}".strip(),
            "choices": choice_list_from_columns(raw),
            "gold_answer": normalize_answer_label(raw.get("Answer") or raw.get("answer")),
            "answer_type": "choice",
            "source_dataset": source_dataset,
            "requires_yoruba_output": True,
            "metadata": {"year": raw.get("year"), "story_id": raw.get("story_id")},
        }
    elif source_dataset in ("afrimgsm", "afrimgsm_translate"):
        row = {
            "id": stable_id(source_dataset, {**raw, "split": split}),
            "task": "math",
            "language": "yo",
            "question": raw.get("question"),
            "choices": None,
            "gold_answer": str(raw.get("answer_number") or raw.get("answer", "")).strip(),
            "answer_type": "number",
            "source_dataset": source_dataset,
            "requires_yoruba_output": True,
            "metadata": {
                "rationale_answer": raw.get("answer"),
                "equation_solution": raw.get("equation_solution"),
            },
        }
    else:
        item = normalize_raw_record(source_dataset, raw)
        return with_split_metadata(item, split, raw, spec)

    return with_split_metadata(BenchmarkItem.from_dict(row), split, raw, spec)


def parse_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return [str(value)]
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
    return [stripped] if stripped else []


def first_answer(value: Any) -> str:
    answers = parse_answer_list(value)
    return answers[0] if answers else ""


def dataset_to_records(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "to_list"):
        return list(dataset.to_list())
    return [dict(row) for row in dataset]


def read_remote_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ttcs-yoruba/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def parse_remote_records(url: str) -> list[dict[str, Any]]:
    text = read_remote_text(url)
    lower_url = url.lower()
    if lower_url.endswith(".json") or lower_url.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if lower_url.endswith(".tsv"):
        return list(csv.DictReader(StringIO(text), delimiter="\t"))
    if lower_url.endswith(".csv"):
        return list(csv.DictReader(StringIO(text)))
    raise ValueError(f"Unsupported remote file format: {url}")


def download_yoruba_hf_dataset_stdlib(
    spec: HFDatasetSpec,
    output_root: str | Path,
    requested_splits: tuple[str, ...],
) -> dict[str, Any]:
    if not spec.data_files:
        raise RuntimeError(f"No stdlib download URLs registered for {spec.key}")

    output_dir = Path(output_root) / spec.group / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": spec.key,
        "hf_id": spec.hf_id,
        "hf_config": spec.config,
        "group": spec.group,
        "language": "yo",
        "download_backend": "stdlib_remote_files",
        "output_fields": list(DOWNLOAD_OUTPUT_FIELDS),
        "requested_splits": list(requested_splits),
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []

    for split in requested_splits:
        if split not in spec.data_files:
            available = ", ".join(spec.data_files)
            raise ValueError(f"Split {split!r} not registered for {spec.key}. Available splits: {available}")
        raw_records = parse_remote_records(spec.data_files[split])
        normalized_items = []
        skipped_empty_gold = 0
        for row_index, raw in enumerate(raw_records, start=1):
            if not is_yoruba_record(raw, spec):
                continue
            try:
                normalized_items.append(normalize_hf_record(spec.key, raw, split, spec))
            except ValidationError as exc:
                if "gold_answer must be a non-empty string" in str(exc):
                    skipped_empty_gold += 1
                    continue
                raise ValueError(f"Failed to normalize {spec.key}/{split} row {row_index}: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"Failed to normalize {spec.key}/{split} row {row_index}: {exc}") from exc
        output_path = output_dir / f"{split}.jsonl"
        output_rows = [compact_download_row(item) for item in normalized_items]
        write_jsonl(output_path, output_rows)
        manifest["splits"][split] = {
            "raw_rows": len(raw_records),
            "retained_yoruba_rows": len(normalized_items),
            "skipped_empty_gold_rows": skipped_empty_gold,
            "path": str(output_path),
            "source_url": spec.data_files[split],
        }
        all_rows.extend(output_rows)

    all_path = output_dir / "all.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(all_path, all_rows)
    manifest["all_path"] = str(all_path)
    manifest["total_retained_yoruba_rows"] = len(all_rows)
    write_json(manifest_path, manifest)
    return manifest


def download_yoruba_hf_dataset_with_datasets(
    spec: HFDatasetSpec,
    output_root: str | Path,
    requested_splits: tuple[str, ...],
) -> dict[str, Any]:
    from datasets import DatasetDict, load_dataset

    load_kwargs: dict[str, Any] = {}
    if spec.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    loaded = load_dataset(spec.hf_id, spec.config, **load_kwargs)
    if not isinstance(loaded, DatasetDict):
        loaded = DatasetDict({"train": loaded})

    output_dir = Path(output_root) / spec.group / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": spec.key,
        "hf_id": spec.hf_id,
        "hf_config": spec.config,
        "group": spec.group,
        "language": "yo",
        "download_backend": "datasets",
        "output_fields": list(DOWNLOAD_OUTPUT_FIELDS),
        "requested_splits": list(requested_splits),
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []

    for split in requested_splits:
        if split not in loaded:
            available = ", ".join(loaded.keys())
            raise ValueError(f"Split {split!r} not found for {spec.key}. Available splits: {available}")
        raw_records = dataset_to_records(loaded[split])
        normalized_items = []
        skipped_empty_gold = 0
        for row_index, raw in enumerate(raw_records, start=1):
            if not is_yoruba_record(raw, spec):
                continue
            try:
                normalized_items.append(normalize_hf_record(spec.key, raw, split, spec))
            except ValidationError as exc:
                if "gold_answer must be a non-empty string" in str(exc):
                    skipped_empty_gold += 1
                    continue
                raise ValueError(f"Failed to normalize {spec.key}/{split} row {row_index}: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"Failed to normalize {spec.key}/{split} row {row_index}: {exc}") from exc
        output_path = output_dir / f"{split}.jsonl"
        output_rows = [compact_download_row(item) for item in normalized_items]
        write_jsonl(output_path, output_rows)
        manifest["splits"][split] = {
            "raw_rows": len(raw_records),
            "retained_yoruba_rows": len(normalized_items),
            "skipped_empty_gold_rows": skipped_empty_gold,
            "path": str(output_path),
        }
        all_rows.extend(output_rows)

    all_path = output_dir / "all.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(all_path, all_rows)
    manifest["all_path"] = str(all_path)
    manifest["total_retained_yoruba_rows"] = len(all_rows)
    write_json(manifest_path, manifest)
    return manifest


def download_yoruba_hf_dataset(
    dataset_key: str,
    output_root: str | Path = "data/normalized",
    splits: list[str] | None = None,
    hf_id_override: str | None = None,
    config_override: str | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    if dataset_key not in HF_DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset key {dataset_key!r}. Expected one of {sorted(HF_DATASET_REGISTRY)}")

    base_spec = HF_DATASET_REGISTRY[dataset_key]
    spec = HFDatasetSpec(
        key=base_spec.key,
        hf_id=hf_id_override or base_spec.hf_id,
        config=config_override or base_spec.config,
        group=base_spec.group,
        default_splits=base_spec.default_splits,
        trust_remote_code=base_spec.trust_remote_code,
        data_files=None if (hf_id_override or config_override) else base_spec.data_files,
    )
    requested_splits = tuple(splits or spec.default_splits)

    if backend not in {"auto", "stdlib", "datasets"}:
        raise ValueError("backend must be one of: auto, stdlib, datasets")
    if backend == "stdlib":
        return download_yoruba_hf_dataset_stdlib(spec, output_root, requested_splits)
    if backend == "datasets":
        return download_yoruba_hf_dataset_with_datasets(spec, output_root, requested_splits)
    if spec.data_files:
        return download_yoruba_hf_dataset_stdlib(spec, output_root, requested_splits)
    return download_yoruba_hf_dataset_with_datasets(spec, output_root, requested_splits)


def normalize_raw_record(source_dataset: str, raw: dict[str, Any]) -> BenchmarkItem:
    defaults = SOURCE_DEFAULTS.get(source_dataset, {"task": "qa", "answer_type": "text"})
    language = first_present(raw, LANGUAGE_KEYS) or "yo"
    if language in {"yor", "yoruba", "Yoruba"}:
        language = "yo"

    question = first_present(raw, QUESTION_KEYS)
    gold_answer = first_present(raw, GOLD_KEYS)
    choices = coerce_choices(first_present(raw, CHOICE_KEYS))

    row = {
        "id": stable_id(source_dataset, raw),
        "task": raw.get("task", defaults["task"]),
        "language": language,
        "question": question,
        "choices": choices,
        "gold_answer": str(gold_answer) if gold_answer is not None else "",
        "answer_type": raw.get("answer_type", defaults["answer_type"]),
        "source_dataset": source_dataset,
        "requires_yoruba_output": bool(raw.get("requires_yoruba_output", True)),
        "metadata": {key: value for key, value in raw.items() if key not in set(QUESTION_KEYS + GOLD_KEYS + CHOICE_KEYS)},
    }
    return BenchmarkItem.from_dict(row)
