"""Build discriminative-PRM training data from PRM800K.

Rows follow ``prm/reference/ft.py``'s ``question``/``process``/``label`` contract
(one ``+``/``-`` per step). English step labels come from the released
``vicky23456/multilingual-PRM800K`` dataset; Yoruba is supplied as a translation
with the same schema. Labels are always true per-step PRM800K labels — no
outcome-derived supervision is used anywhere in this package.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ttcs_yoruba.io_utils import write_json, write_jsonl

from .config import BuildConfig, resolve_path
from .steps import QWEN_STEP_TAG, split_tagged_process

# Released multilingual PRM800K / Math-Shepherd (de, en, es, fr, ru, sw, zh — no Yoruba).
PRM800K_REPO_ID = "vicky23456/multilingual-PRM800K"
# Languages shipped in the released dataset; any other must be supplied locally.
RELEASED_PRM800K_LANGUAGES = {"de", "en", "es", "fr", "ru", "sw", "zh"}


def validate_prm800k_row(row: dict[str, Any], *, step_tag: str) -> dict[str, Any] | None:
    """Validate/normalise one released-style PRM800K row (``question``/``process``/``label``)."""
    question = row.get("question")
    process = row.get("process")
    label = row.get("label")
    if not question or not process or not isinstance(label, list) or not label:
        return None

    normalised: list[str] = []
    for value in label:
        if value in ("+", 1, True):
            normalised.append("+")
        elif value in ("-", 0, False):
            normalised.append("-")
        else:
            return None

    if process.count(step_tag) == len(normalised) - 1:
        # Steps joined without a trailing tag: add it so tag count == label count.
        process = process + step_tag
    if process.count(step_tag) != len(normalised):
        return None

    return {"question": str(question), "process": str(process), "label": normalised}


def _download_prm800k_file(repo_id: str, filename: str, raw_dir: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(resolve_path(raw_dir)),
        )
    )


def _read_prm800k_rows(path: Path, *, cap: int | None, seed: int) -> list[dict[str, Any]]:
    """Read a (large) pretty-printed JSON array, subsampling when a cap is set.

    The ``datasets`` library memory-maps via Arrow, so we prefer it for capped
    phase2 reads; it falls back to ``json.load`` for small/local files.
    """
    if cap is not None:
        try:
            from datasets import load_dataset

            dataset = load_dataset("json", data_files=str(path), split="train")
            count = min(int(cap), len(dataset))
            dataset = dataset.shuffle(seed=seed).select(range(count))
            return [dict(row) for row in dataset]
        except Exception:
            pass

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if cap is not None and len(data) > cap:
        data = random.Random(seed).sample(data, int(cap))
    return data


def build_prm800k_rows(
    source: dict[str, Any],
    *,
    seed: int,
    step_tag: str,
) -> list[dict[str, Any]]:
    """Load released multilingual PRM800K files + user-supplied translations.

    Released languages (en, de, es, fr, ru, sw, zh) are downloaded from
    ``vicky23456/multilingual-PRM800K``; any other language (notably ``yor``)
    must be supplied via ``paths`` with the same ``question/process/label``
    schema. ``phase1`` is taken in full; ``phase2_max_examples`` caps phase2.
    """
    languages = list(source.get("languages", []))
    repo_id = str(source.get("repo_id", PRM800K_REPO_ID))
    raw_dir = str(source.get("raw_dir", "prm/data/raw/multilingual-PRM800K"))
    paths = source.get("paths", {}) or {}
    include_phase1 = bool(source.get("phase1", True))
    phase2_cap = source.get("phase2_max_examples")
    per_language_cap = source.get("max_examples_per_language")
    rng = random.Random(seed)

    all_rows: list[dict[str, Any]] = []
    for lang in languages:
        user_paths = paths.get(lang)
        supplied = bool(user_paths)
        entries: list[tuple[Path, int | None]] = []
        if user_paths:
            if isinstance(user_paths, (str, Path)):
                user_paths = [user_paths]
            for path in user_paths:
                resolved = resolve_path(path)
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"PRM800K file for language {lang!r} not found: {resolved}. "
                        f"Add your translation under sources[].paths[{lang!r}]."
                    )
                entries.append((resolved, per_language_cap))
        else:
            if lang not in RELEASED_PRM800K_LANGUAGES:
                raise ValueError(
                    f"PRM800K language {lang!r} is not in the released dataset. "
                    f"Supply translations via sources[].paths[{lang!r}] (list of JSON files) "
                    f"using the {{question, process, label}} schema."
                )
            if include_phase1:
                entries.append(
                    (
                        _download_prm800k_file(
                            repo_id, f"prm800k/phase1_train_{lang}.new.json", raw_dir
                        ),
                        None,
                    )
                )
            if phase2_cap:
                entries.append(
                    (
                        _download_prm800k_file(
                            repo_id, f"prm800k/phase2_train_{lang}.new.json", raw_dir
                        ),
                        int(phase2_cap),
                    )
                )

        language_rows: list[dict[str, Any]] = []
        for path, cap in entries:
            for raw in _read_prm800k_rows(path, cap=cap, seed=seed):
                valid = validate_prm800k_row(raw, step_tag=step_tag)
                if valid is None:
                    continue
                language_rows.append(
                    {
                        "id": f"prm800k:{lang}:{len(language_rows)}",
                        "group_id": f"prm800k:{lang}:{len(language_rows)}",
                        "source": "prm800k",
                        "language": lang,
                        "task": "math",
                        "answer_type": "number",
                        "question": valid["question"],
                        "process": valid["process"],
                        "steps": split_tagged_process(valid["process"], step_tag),
                        "label": valid["label"],
                        "n_steps": len(valid["label"]),
                        "problem_source": "supplied" if supplied else "released",
                    }
                )
        if per_language_cap and len(language_rows) > per_language_cap:
            language_rows = rng.sample(language_rows, int(per_language_cap))
        all_rows.extend(language_rows)
    return all_rows


def iter_candidate_files(
    runs_dir: str | Path,
    run_ids: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """List ``runs/<id>/candidates.jsonl`` files (used for PRM scoring, not training)."""
    root = resolve_path(runs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Runs directory not found: {root}")
    if run_ids:
        directories = [root / run_id for run_id in run_ids]
    else:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    files: list[tuple[str, Path]] = []
    for directory in directories:
        candidates = directory / "candidates.jsonl"
        if candidates.exists():
            files.append((directory.name, candidates))
    return files


def build_dataset(config: BuildConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in config.sources:
        if not source.get("enabled", True):
            continue
        source_type = source.get("type")
        if source_type == "prm800k":
            rows.extend(build_prm800k_rows(source, seed=config.seed, step_tag=QWEN_STEP_TAG))
        elif source_type == "runs":
            raise ValueError(
                "The 'runs' source is not supported: this PRM is trained only on "
                "step-labelled PRM800K, never on outcome-derived labels."
            )
        else:
            raise ValueError(f"Unknown source type: {source_type!r} (expected 'prm800k')")
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by ``group_id`` so all rows for an example stay together."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    n_eval = int(round(len(group_ids) * max(0.0, min(1.0, eval_fraction))))
    n_eval = max(1, n_eval) if group_ids and eval_fraction > 0 else 0
    eval_ids = set(group_ids[:n_eval])

    train = [row for row in rows if row["group_id"] not in eval_ids]
    evaluate = [row for row in rows if row["group_id"] in eval_ids]
    return train, evaluate


def write_built_dataset(
    train: list[dict[str, Any]],
    evaluate: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = resolve_path(output_dir)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "eval.jsonl", evaluate)
    write_jsonl(output / "all.jsonl", all_rows)

    def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_language: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_task: dict[str, int] = {}
        for row in rows:
            by_language[str(row.get("language"))] = by_language.get(str(row.get("language")), 0) + 1
            by_source[str(row.get("source"))] = by_source.get(str(row.get("source")), 0) + 1
            by_task[str(row.get("task"))] = by_task.get(str(row.get("task")), 0) + 1
        return {
            "total": len(rows),
            "by_language": by_language,
            "by_source": by_source,
            "by_task": by_task,
        }

    manifest = {
        "output_dir": str(output),
        "counts": {
            "train": counts(train),
            "eval": counts(evaluate),
            "all": counts(all_rows),
        },
        "paths": {
            "train": str(output / "train.jsonl"),
            "eval": str(output / "eval.jsonl"),
            "all": str(output / "all.jsonl"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
