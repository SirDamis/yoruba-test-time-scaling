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
DOWNLOAD_OUTPUT_FIELDS = ("answer_type", "choices", "gold_answer", "question")

# Experiment languages (native benchmarks) + English baseline.
SUPPORTED_LANGUAGES = ("yor", "hau", "ibo", "swa", "amh")
BASELINE_LANGUAGES = ("eng",)
ALL_LANGUAGE_CODES = SUPPORTED_LANGUAGES + BASELINE_LANGUAGES

LANGUAGE_NAMES = {
    "yor": "Yoruba",
    "hau": "Hausa",
    "ibo": "Igbo",
    "swa": "Swahili",
    "amh": "Amharic",
    "eng": "English",
    "yo": "Yoruba",
}

# Codes that may appear in a dataset's own language column, per experiment code.
LANGUAGE_COLUMN_ALIASES: dict[str, set[str]] = {
    "yor": {"yo", "yor", "yoruba", "Yoruba", "YOR", "YO"},
    "hau": {"hau", "hausa", "Hausa"},
    "ibo": {"ibo", "igbo", "Igbo"},
    "swa": {"swa", "swahili", "Swahili", "swh"},
    "amh": {"amh", "amharic", "Amharic"},
    "eng": {"eng", "english", "English"},
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


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


# Base benchmark templates. Language-specific specs are built by
# ``build_language_spec`` as ``{base}_{lang}`` (e.g. ``afrimgsm_hau``),
# downloading from the matching per-language folder on Hugging Face.
@dataclass(frozen=True)
class _BaseDatasetTemplate:
    hf_id: str
    group: str
    default_splits: tuple[str, ...]
    # split -> remote filename inside data/<lang>/
    split_files: dict[str, str]
    trust_remote_code: bool = False


BASE_DATASET_TEMPLATES: dict[str, _BaseDatasetTemplate] = {
    "afrimmlu": _BaseDatasetTemplate(
        hf_id="masakhane/afrimmlu",
        group="question-answering",
        # Test-only by default: merging validation/dev/test risks duplicate items
        # (val/dev overlap) and mixes splits into evaluation.
        default_splits=("test",),
        split_files={"validation": "val.tsv", "dev": "dev.tsv", "test": "test.tsv"},
    ),
    "afrimgsm": _BaseDatasetTemplate(
        hf_id="masakhane/afrimgsm",
        group="math-reasoning",
        default_splits=("dev", "test"),
        split_files={"dev": "dev.tsv", "test": "test.tsv"},
    ),
}

TEMPLATE_LANGUAGES: dict[str, tuple[str, ...]] = {
    "afrimmlu": ALL_LANGUAGE_CODES,
    "afrimgsm": ALL_LANGUAGE_CODES,
}


def build_language_spec(base_key: str, lang: str) -> HFDatasetSpec:
    """Build a concrete spec for ``{base}_{lang}`` (e.g. ``afrimgsw_hau``)."""
    if base_key not in BASE_DATASET_TEMPLATES:
        raise ValueError(f"Unknown base dataset {base_key!r}")
    if lang not in ALL_LANGUAGE_CODES:
        raise ValueError(f"Unsupported language {lang!r}. Expected one of {sorted(ALL_LANGUAGE_CODES)}")
    template = BASE_DATASET_TEMPLATES[base_key]
    data_files = {
        split: f"https://huggingface.co/datasets/{template.hf_id}/resolve/main/data/{lang}/{filename}"
        for split, filename in template.split_files.items()
    }
    return HFDatasetSpec(
        key=f"{base_key}_{lang}",
        hf_id=template.hf_id,
        config=lang,
        group=template.group,
        default_splits=template.default_splits,
        trust_remote_code=template.trust_remote_code,
        data_files=data_files,
    )


# Legacy / translate-only entries (Yoruba). Native multi-language downloads use
# build_language_spec; these remain for backward compatibility.
HF_DATASET_REGISTRY: dict[str, HFDatasetSpec] = {
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


def resolve_dataset_spec(dataset_key: str) -> HFDatasetSpec:
    """Resolve a dataset key to a concrete spec.

    Accepts ``{base}_{lang}`` keys (e.g. ``afrimgsm_swa``), legacy bare keys
    (``afrimgsm`` → Yoruba), and the translate/legacy registry entries.
    """
    if "_" in dataset_key:
        base, _, suffix = dataset_key.rpartition("_")
        if base in BASE_DATASET_TEMPLATES and suffix in ALL_LANGUAGE_CODES:
            return build_language_spec(base, suffix)
    if dataset_key in HF_DATASET_REGISTRY:
        return HF_DATASET_REGISTRY[dataset_key]
    # Legacy bare template keys default to Yoruba.
    if dataset_key in BASE_DATASET_TEMPLATES:
        return build_language_spec(dataset_key, "yor")
    raise ValueError(
        f"Unknown dataset key {dataset_key!r}. Expected one of {sorted(available_dataset_keys())}"
    )


def available_dataset_keys() -> list[str]:
    keys = set(HF_DATASET_REGISTRY)
    for base_key, langs in TEMPLATE_LANGUAGES.items():
        keys.update(f"{base_key}_{lang}" for lang in langs)
    return sorted(keys)


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


def is_language_record(raw: dict[str, Any], spec: HFDatasetSpec) -> bool:
    """Keep records whose language column matches the spec's language.

    Files without a language column are folder-selected (the URL already pins
    the language), so they are kept. Legacy Yoruba aliases are honored.
    """
    language = first_present(raw, LANGUAGE_KEYS)
    if language is None:
        return True
    value = str(language).strip()
    if value in LANGUAGE_COLUMN_ALIASES.get(spec.config, {spec.config}):
        return True
    return value.lower() == spec.config.lower()


# Backward-compatible alias.
is_yoruba_record = is_language_record


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
    lang_code = spec.config if spec.config in ALL_LANGUAGE_CODES else "yor"
    if source_dataset.startswith("afrimmlu"):
        choices = coerce_choices(raw.get("choices"))
        row = {
            "id": stable_id(source_dataset, {**raw, "split": split}),
            "task": "qa",
            "language": lang_code,
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
            "language": lang_code,
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
            "language": lang_code,
            "question": f"Àyọkà:\n{story}\n\nÌbéèrè:\n{question}".strip(),
            "choices": choice_list_from_columns(raw),
            "gold_answer": normalize_answer_label(raw.get("Answer") or raw.get("answer")),
            "answer_type": "choice",
            "source_dataset": source_dataset,
            "requires_yoruba_output": True,
            "metadata": {"year": raw.get("year"), "story_id": raw.get("story_id")},
        }
    elif source_dataset.startswith("afrimgsm"):
        row = {
            "id": stable_id(source_dataset, {**raw, "split": split}),
            "task": "math",
            "language": lang_code,
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
    base_spec = resolve_dataset_spec(dataset_key)
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
    language = first_present(raw, LANGUAGE_KEYS) or "yor"
    normalized_language = str(language).strip().lower()
    if normalized_language in {"yo", "yor", "yoruba"}:
        language = "yor"
    elif normalized_language in LANGUAGE_COLUMN_ALIASES:
        language = normalized_language

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
