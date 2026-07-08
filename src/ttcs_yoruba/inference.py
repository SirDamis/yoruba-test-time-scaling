from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .backends import build_backend
from .config import DatasetConfig, InferenceMethodConfig, InferenceModelConfig, InferenceRunConfig
from .examples import InferenceExample, load_inference_examples
from .extraction import extract_answer, is_exact_match
from .io_utils import write_json
from .prompting import render_prompt
from .schema import BackendOutput, CandidateRecord
from .selection import select_candidate


def run_inference_pipeline(
    config: InferenceRunConfig,
    *,
    dataset_names: set[str] | None = None,
    model_names: set[str] | None = None,
    method_names: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    output_dir = Path(config.output_dir) / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    selections_path = output_dir / "selections.jsonl"
    manifest_path = output_dir / "manifest.json"

    datasets = filter_named(config.datasets, dataset_names)
    models = filter_named(config.models, model_names)
    methods = filter_named(config.methods, method_names)

    counts = {
        "datasets": len(datasets),
        "models": len(models),
        "methods": len(methods),
        "examples": 0,
        "candidate_rows": 0,
        "selection_rows": 0,
    }

    with candidates_path.open("w", encoding="utf-8", newline="\n") as candidate_handle, selections_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as selection_handle:
        for dataset in datasets:
            examples = load_dataset_examples(dataset, limit=limit)
            counts["examples"] += len(examples)
            for model in models:
                backend = build_backend(model, default_timeout_s=config.default_request_timeout_s)
                for method in methods:
                    for example in examples:
                        candidate_rows = run_example_candidates(
                            config=config,
                            dataset=dataset,
                            model=model,
                            method=method,
                            example=example,
                            backend=backend,
                        )
                        for row in candidate_rows:
                            write_jsonl_row(candidate_handle, row)
                        counts["candidate_rows"] += len(candidate_rows)

                        selection_row = build_selection_row(
                            config=config,
                            dataset=dataset,
                            model=model,
                            method=method,
                            example=example,
                            candidate_rows=candidate_rows,
                        )
                        write_jsonl_row(selection_handle, selection_row)
                        counts["selection_rows"] += 1

    manifest = {
        "run_id": config.run_id,
        "output_dir": str(output_dir),
        "candidates_path": str(candidates_path),
        "selections_path": str(selections_path),
        "datasets": [dataset.name for dataset in datasets],
        "models": [{"name": model.name, "size_label": model.size_label} for model in models],
        "methods": [
            {
                "name": method.name,
                "prompt_style": method.prompt_style,
                "selection": method.selection,
                "n": method.n,
            }
            for method in methods
        ],
        "counts": counts,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_dataset_examples(dataset: DatasetConfig, *, limit: int | None) -> list[InferenceExample]:
    effective_limit = limit if limit is not None else dataset.limit
    return load_inference_examples(
        dataset.path,
        dataset_name=dataset.name,
        task=dataset.task,
        source_dataset=dataset.source_dataset,
        limit=effective_limit,
    )


def run_example_candidates(
    *,
    config: InferenceRunConfig,
    dataset: DatasetConfig,
    model: InferenceModelConfig,
    method: InferenceMethodConfig,
    example: InferenceExample,
    backend,
) -> list[dict[str, object]]:
    prompt = render_prompt(example, method.prompt_style)
    rows: list[dict[str, object]] = []
    for sample_index in range(method.n):
        seed = None if config.seed is None else config.seed + sample_index
        started = time.monotonic()
        try:
            output = backend.generate(
                system_prompt=prompt.system,
                user_prompt=prompt.user,
                temperature=method.temperature,
                max_tokens=method.max_tokens,
                seed=seed,
            )
        except Exception as exc:
            if not config.continue_on_error:
                raise
            output = BackendOutput(
                response="",
                token_count=0,
                latency_s=time.monotonic() - started,
                metadata={"error": repr(exc)},
            )

        extracted = extract_answer(output.response, example.answer_type, example.choices)
        record = CandidateRecord(
            run_id=config.run_id,
            example_id=example.id,
            task=example.task,
            source_dataset=example.source_dataset,
            model=model.name,
            model_size_label=model.size_label,
            method=method.name,
            prompt_style=method.prompt_style,
            selection=method.selection,
            sample_index=sample_index,
            n=method.n,
            reasoning_language=method.reasoning_language,
            prompt=prompt.as_text(),
            response=output.response,
            extracted_answer=extracted,
            token_count=output.token_count,
            latency_s=output.latency_s,
            estimated_cost=(output.token_count / 1000.0) * model.cost_per_1k_tokens,
            metadata={
                "dataset": dataset.name,
                "answer_type": example.answer_type,
                "choices": example.choices,
                "gold_answer": example.gold_answer,
                "backend": model.backend,
                "backend_metadata": output.metadata,
            },
        )
        rows.append(record.to_dict())
    return rows


def build_selection_row(
    *,
    config: InferenceRunConfig,
    dataset: DatasetConfig,
    model: InferenceModelConfig,
    method: InferenceMethodConfig,
    example: InferenceExample,
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    selected = select_candidate(candidate_rows, method.selection)
    return {
        "run_id": config.run_id,
        "dataset": dataset.name,
        "example_id": example.id,
        "task": example.task,
        "source_dataset": example.source_dataset,
        "model": model.name,
        "model_size_label": model.size_label,
        "method": method.name,
        "prompt_style": method.prompt_style,
        "selection": method.selection,
        "n": method.n,
        "reasoning_language": method.reasoning_language,
        "selected_sample_index": selected.selected_sample_index,
        "selected_answer": selected.selected_answer,
        "gold_answer": example.gold_answer,
        "answer_type": example.answer_type,
        "is_correct": is_exact_match(selected.selected_answer, example.gold_answer, example.answer_type),
        "candidate_count": len(candidate_rows),
        "vote_counts": selected.vote_counts,
    }


def filter_named(items: Iterable[object], selected_names: set[str] | None) -> list:
    item_list = list(items)
    if selected_names is None:
        return item_list
    available_names = {str(getattr(item, "name")) for item in item_list}
    missing_names = selected_names - available_names
    if missing_names:
        raise ValueError(f"Unknown names in filter: {sorted(missing_names)}. Available: {sorted(available_names)}")
    return [item for item in item_list if getattr(item, "name") in selected_names]


def write_jsonl_row(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    handle.write("\n")
