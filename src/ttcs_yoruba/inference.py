from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .backends import build_backend
from .config import DatasetConfig, InferenceMethodConfig, InferenceModelConfig, InferenceRunConfig
from .examples import InferenceExample, load_inference_examples
from .extraction import extract_answer, is_exact_match
from .io_utils import write_json
from .prompting import render_prompt
from .schema import BackendOutput, CandidateRecord
from .selection import select_candidate

# Checkpoint file listing fully finished work units (written after each unit is flushed).
CHECKPOINT_FILENAME = "completed_units.jsonl"


def log_progress(message: str, *, enabled: bool = True) -> None:
    """Print a progress line to stderr (keeps final stdout JSON clean)."""
    if not enabled:
        return
    print(message, file=sys.stderr, flush=True)


def _format_progress_line(
    *,
    run_id: str,
    done: int,
    total: int,
    dataset: str,
    model: str,
    method: str,
    example_id: str,
    is_correct: bool | None = None,
    latency_s: float | None = None,
    skipped: int = 0,
    correct_so_far: int | None = None,
    scored_so_far: int | None = None,
) -> str:
    pct = (100.0 * done / total) if total else 100.0
    parts = [
        f"[{run_id}]",
        f"{done}/{total} ({pct:5.1f}%)",
        f"{dataset} | {model} | {method} | {example_id}",
    ]
    if is_correct is not None:
        parts.append("OK" if is_correct else "WRONG")
    if correct_so_far is not None and scored_so_far is not None and scored_so_far > 0:
        acc = 100.0 * correct_so_far / scored_so_far
        parts.append(f"acc={acc:.1f}% ({correct_so_far}/{scored_so_far})")
    if latency_s is not None:
        parts.append(f"{latency_s:.1f}s")
    if skipped:
        parts.append(f"skipped={skipped}")
    return "  ".join(parts)


def derive_sample_seed(base_seed: int | None, example_id: str, sample_index: int) -> int | None:
    """Derive a reproducible seed unique to (example, sample_index).

    Avoids reusing ``base_seed + sample_index`` for every example, which made
    the i-th draw share the same RNG state across all questions.
    """
    if base_seed is None:
        return None
    payload = f"{base_seed}\0{example_id}\0{sample_index}".encode("utf-8")
    # 31-bit positive int: portable for torch.manual_seed and common API seeds.
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**31)


def standalone_unit_key(dataset: str, model: str, method: str, example_id: str) -> str:
    return f"standalone\t{dataset}\t{model}\t{method}\t{example_id}"


def nested_unit_key(dataset: str, model: str, group_id: str, example_id: str) -> str:
    return f"nested\t{dataset}\t{model}\t{group_id}\t{example_id}"


def load_completed_units(path: Path) -> set[str]:
    """Load finished work-unit keys from a checkpoint JSONL file."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                # Truncated last line after a crash — ignore and continue.
                continue
            key = row.get("key")
            if isinstance(key, str) and key:
                done.add(key)
            elif all(k in row for k in ("unit_type", "dataset", "model", "example_id")):
                # Backward-compatible structured rows.
                if row["unit_type"] == "standalone":
                    done.add(
                        standalone_unit_key(
                            str(row["dataset"]),
                            str(row["model"]),
                            str(row["method"]),
                            str(row["example_id"]),
                        )
                    )
                elif row["unit_type"] == "nested":
                    done.add(
                        nested_unit_key(
                            str(row["dataset"]),
                            str(row["model"]),
                            str(row["group_id"]),
                            str(row["example_id"]),
                        )
                    )
    return done


def candidate_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identity for de-duplicating candidate rows (last write wins)."""
    meta = row.get("metadata") or {}
    dataset = str(row.get("dataset") or meta.get("dataset") or "")
    return (
        dataset,
        str(row.get("model", "")),
        str(row.get("method", "")),
        int(row.get("n") or 1),
        str(row.get("example_id", "")),
        int(row.get("sample_index") or 0),
    )


def selection_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identity for de-duplicating selection rows (last write wins)."""
    return (
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("method", "")),
        int(row.get("n") or 1),
        str(row.get("example_id", "")),
    )


def dedupe_rows_keep_last(
    rows: list[dict[str, Any]],
    identity_fn,
) -> tuple[list[dict[str, Any]], int]:
    """Keep the last row for each identity. Returns (rows, num_dropped)."""
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = identity_fn(row)
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    deduped = [by_key[key] for key in order]
    return deduped, len(rows) - len(deduped)


def _unit_key_from_selection_row(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", ""))
    model = str(row.get("model", ""))
    example_id = str(row.get("example_id", ""))
    group_id = row.get("nested_group_id")
    if row.get("nested") or group_id:
        return nested_unit_key(dataset, model, str(group_id or ""), example_id)
    return standalone_unit_key(dataset, model, str(row.get("method", "")), example_id)


def _unit_key_from_candidate_row(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    dataset = str(row.get("dataset") or meta.get("dataset") or "")
    model = str(row.get("model", ""))
    example_id = str(row.get("example_id", ""))
    group_id = meta.get("nested_group_id") or row.get("nested_group_id")
    if meta.get("nested") or group_id or row.get("nested"):
        return nested_unit_key(dataset, model, str(group_id or ""), example_id)
    return standalone_unit_key(dataset, model, str(row.get("method", "")), example_id)


def load_completed_units_from_selections(
    path: Path,
    *,
    nested_methods_by_group: dict[str, set[str]],
) -> set[str]:
    """Infer finished work units from selection rows already on disk.

    Nested units count as complete only when every expected method for the
    group appears (avoids treating a partial nested write as done).
    """
    if not path.exists():
        return set()

    from .io_utils import read_jsonl

    try:
        rows = read_jsonl(path)
    except ValueError:
        # Truncated / corrupt last line — skip incomplete parse; compact will help later.
        rows = _read_jsonl_lenient(path)

    nested_methods_seen: dict[str, set[str]] = defaultdict(set)
    standalone_done: set[str] = set()

    for row in rows:
        key = _unit_key_from_selection_row(row)
        if key.startswith("nested\t"):
            nested_methods_seen[key].add(str(row.get("method", "")))
        else:
            standalone_done.add(key)

    done = set(standalone_done)
    for key, methods_seen in nested_methods_seen.items():
        # key: nested\tdataset\tmodel\tgroup_id\texample_id
        parts = key.split("\t")
        group_id = parts[3] if len(parts) >= 5 else ""
        expected = nested_methods_by_group.get(group_id)
        if expected is None:
            # Unknown group (legacy): treat as complete if any selection exists.
            done.add(key)
        elif expected <= methods_seen:
            done.add(key)
    return done


def _read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows


def rewrite_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL file (write temp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass
    tmp_path.replace(path)


def prepare_resume_artifacts(
    *,
    candidates_path: Path,
    selections_path: Path,
    nested_methods_by_group: dict[str, set[str]],
) -> tuple[set[str], dict[str, int]]:
    """De-dupe on-disk artifacts, drop incomplete units, return completed keys.

    Handles the crash window where candidates/selections were written but the
    checkpoint was not updated, and cleans any duplicate appends.
    """
    from .io_utils import read_jsonl

    stats = {
        "candidate_duplicates_dropped": 0,
        "selection_duplicates_dropped": 0,
        "incomplete_units_stripped": 0,
    }

    candidates: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    if candidates_path.exists():
        try:
            candidates = read_jsonl(candidates_path)
        except ValueError:
            candidates = _read_jsonl_lenient(candidates_path)
    if selections_path.exists():
        try:
            selections = read_jsonl(selections_path)
        except ValueError:
            selections = _read_jsonl_lenient(selections_path)

    candidates, dropped_c = dedupe_rows_keep_last(candidates, candidate_identity)
    selections, dropped_s = dedupe_rows_keep_last(selections, selection_identity)
    stats["candidate_duplicates_dropped"] = dropped_c
    stats["selection_duplicates_dropped"] = dropped_s

    # Nested: require all expected methods; strip partial nested units.
    nested_methods_seen: dict[str, set[str]] = defaultdict(set)
    for row in selections:
        key = _unit_key_from_selection_row(row)
        if key.startswith("nested\t"):
            nested_methods_seen[key].add(str(row.get("method", "")))

    incomplete: set[str] = set()
    for key, methods_seen in nested_methods_seen.items():
        parts = key.split("\t")
        group_id = parts[3] if len(parts) >= 5 else ""
        expected = nested_methods_by_group.get(group_id)
        if expected is not None and not expected <= methods_seen:
            incomplete.add(key)

    # Standalone candidates without a matching selection → incomplete (partial write).
    selection_units = {_unit_key_from_selection_row(row) for row in selections}
    for row in candidates:
        key = _unit_key_from_candidate_row(row)
        if key.startswith("standalone\t") and key not in selection_units:
            incomplete.add(key)

    if incomplete:
        stats["incomplete_units_stripped"] = len(incomplete)
        candidates = [row for row in candidates if _unit_key_from_candidate_row(row) not in incomplete]
        selections = [row for row in selections if _unit_key_from_selection_row(row) not in incomplete]

    if candidates_path.exists() or candidates:
        rewrite_jsonl_atomic(candidates_path, candidates)
    if selections_path.exists() or selections:
        rewrite_jsonl_atomic(selections_path, selections)

    completed = load_completed_units_from_selections(
        selections_path,
        nested_methods_by_group=nested_methods_by_group,
    )
    return completed, stats


def append_completed_unit(path: Path, key: str, *, extra: dict[str, Any] | None = None) -> None:
    """Append one completed unit and fsync so a crash keeps the checkpoint durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key, **(extra or {})}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass


def write_unit_batch(
    candidate_handle: Any,
    selection_handle: Any,
    *,
    candidate_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
) -> None:
    """Write all rows for one work unit, then flush (minimizes partial-unit windows)."""
    for row in candidate_rows:
        write_jsonl_row(candidate_handle, row)
    for row in selection_rows:
        write_jsonl_row(selection_handle, row)
    flush_handles(candidate_handle, selection_handle)


def effective_max_concurrent(config: InferenceRunConfig, model: InferenceModelConfig) -> int:
    """Concurrency for this model. Transformers generate is not thread-safe → 1."""
    requested = max(1, int(config.max_concurrent))
    if model.backend != "openai_compatible" and requested > 1:
        return 1
    return requested


def run_concurrent_map(
    items: list[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
) -> list[Any]:
    """Run ``worker`` over ``items`` with up to ``max_workers`` threads.

    Results are returned in the **same order** as ``items`` (not completion order).

    If some workers fail, successful results are still returned in-order with
    ``None`` placeholders for failures, and the first exception is raised
    *after* the full wave finishes (so callers can checkpoint successes first
    if they inspect results — see ``run_concurrent_map_tolerant``).
    """
    results, errors = run_concurrent_map_tolerant(items, worker, max_workers=max_workers)
    if errors:
        raise errors[0][1]
    return results


def run_concurrent_map_tolerant(
    items: list[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
) -> tuple[list[Any | None], list[tuple[int, BaseException]]]:
    """Like ``run_concurrent_map`` but returns ``(results, errors)`` without raising.

    ``results[i]`` is the worker return value, or ``None`` if that item failed.
    ``errors`` is a list of ``(index, exception)`` for failed items.
    """
    if not items:
        return [], []
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        results: list[Any | None] = []
        errors: list[tuple[int, BaseException]] = []
        for idx, item in enumerate(items):
            try:
                results.append(worker(item))
            except BaseException as exc:  # noqa: BLE001 — surface to caller
                results.append(None)
                errors.append((idx, exc))
        return results, errors

    results = [None] * len(items)
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(worker, item): idx for idx, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except BaseException as exc:  # noqa: BLE001 — collect, re-raise later
                results[idx] = None
                errors.append((idx, exc))
    errors.sort(key=lambda pair: pair[0])
    return results, errors


def run_inference_pipeline(
    config: InferenceRunConfig,
    *,
    dataset_names: set[str] | None = None,
    model_names: set[str] | None = None,
    method_names: set[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    overwrite: bool = False,
    progress: bool = True,
    max_concurrent: int | None = None,
) -> dict[str, object]:
    """Run inference and write candidates/selections under ``runs/<run_id>/``.

    Resume / checkpoint
    -------------------
    Progress is tracked in ``completed_units.jsonl`` and recovered from on-disk
    ``selections.jsonl`` if the checkpoint lags behind a crash. Each entry is
    one finished work unit:

    - standalone: ``(dataset, model, method, example_id)``
    - nested group: ``(dataset, model, nested_group_id, example_id)`` for **all**
      k-slices of that example (buffered, then written as one batch)

    ``resume=True`` (default): de-dupe artifacts, skip completed units, append.
    ``overwrite=True``: delete prior artifacts for this run_id and start clean.
    ``progress=True`` (default): print per-unit progress to stderr.
    ``max_concurrent``: override config concurrency for openai_compatible backends.
    """
    if overwrite and resume:
        # Overwrite wins: start a fresh run directory state.
        resume = False

    if max_concurrent is not None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        config = replace(config, max_concurrent=max_concurrent)

    output_dir = Path(config.output_dir) / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    selections_path = output_dir / "selections.jsonl"
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    manifest_path = output_dir / "manifest.json"

    if overwrite:
        for path in (candidates_path, selections_path, checkpoint_path, manifest_path):
            if path.exists():
                path.unlink()

    datasets = filter_named(config.datasets, dataset_names)
    models = filter_named(config.models, model_names)
    methods = filter_named(config.methods, method_names)
    standalone_methods, nested_groups = partition_methods(methods)
    nested_methods_by_group = {
        group_id: {m.name for m in group_methods}
        for group_id, group_methods in nested_groups.items()
    }

    resume_stats: dict[str, int] = {
        "candidate_duplicates_dropped": 0,
        "selection_duplicates_dropped": 0,
        "incomplete_units_stripped": 0,
    }
    if resume:
        # Checkpoint + on-disk selections (covers write-then-crash-before-checkpoint).
        completed = load_completed_units(checkpoint_path)
        if candidates_path.exists() or selections_path.exists():
            artifact_done, resume_stats = prepare_resume_artifacts(
                candidates_path=candidates_path,
                selections_path=selections_path,
                nested_methods_by_group=nested_methods_by_group,
            )
            completed |= artifact_done
        file_mode = "a" if (candidates_path.exists() or selections_path.exists()) else "w"
    else:
        completed = set()
        file_mode = "w"

    counts = {
        "datasets": len(datasets),
        "models": len(models),
        "methods": len(methods),
        "standalone_methods": len(standalone_methods),
        "nested_groups": len(nested_groups),
        "examples": 0,
        "candidate_rows": 0,
        "selection_rows": 0,
        "model_generations": 0,
        "units_skipped": 0,
        "units_completed_this_run": 0,
        "resume": resume and file_mode == "a",
        "overwrite": overwrite,
        "completed_units_loaded": len(completed),
        "max_concurrent_config": config.max_concurrent,
        **resume_stats,
    }

    # Preload examples so we can report total planned work units.
    examples_by_dataset: dict[str, list[InferenceExample]] = {}
    total_units = 0
    for dataset in datasets:
        examples = load_dataset_examples(dataset, limit=limit)
        examples_by_dataset[dataset.name] = examples
        counts["examples"] += len(examples)
        total_units += len(examples) * len(models) * (
            len(standalone_methods) + len(nested_groups)
        )

    units_processed = 0  # completed this run + skipped (for progress denominator walk)
    correct_so_far = 0
    scored_so_far = 0
    run_started = time.monotonic()

    log_progress(
        f"[{config.run_id}] starting  "
        f"datasets={ [d.name for d in datasets] }  "
        f"models={ [m.name for m in models] }  "
        f"methods={ [m.name for m in methods] }  "
        f"total_units={total_units}  "
        f"already_done={len(completed)}  "
        f"max_concurrent={config.max_concurrent}  "
        f"overwrite={overwrite}  resume={resume and file_mode == 'a'}  "
        f"output={output_dir}",
        enabled=progress,
    )

    with candidates_path.open(file_mode, encoding="utf-8", newline="\n") as candidate_handle, selections_path.open(
        file_mode, encoding="utf-8", newline="\n"
    ) as selection_handle:
        for dataset in datasets:
            examples = examples_by_dataset[dataset.name]
            log_progress(
                f"[{config.run_id}] dataset={dataset.name}  examples={len(examples)}",
                enabled=progress,
            )
            for model in models:
                concurrency = effective_max_concurrent(config, model)
                log_progress(
                    f"[{config.run_id}] loading model={model.name} ({model.model})  "
                    f"backend={model.backend}  concurrent={concurrency} ...",
                    enabled=progress,
                )
                model_load_started = time.monotonic()
                backend = build_backend(model, default_timeout_s=config.default_request_timeout_s)
                attn = getattr(backend, "attn_implementation", None)
                attn_msg = f"  attn={attn}" if attn else ""
                log_progress(
                    f"[{config.run_id}] model ready={model.name}  "
                    f"load_s={time.monotonic() - model_load_started:.1f}{attn_msg}",
                    enabled=progress,
                )

                # Independent methods: sample N times per method (legacy / non-nested).
                for method in standalone_methods:
                    log_progress(
                        f"[{config.run_id}] method={method.name}  "
                        f"style={method.prompt_style}  n={method.n}  "
                        f"max_tokens={method.max_tokens}  concurrent={concurrency}",
                        enabled=progress,
                    )
                    pending: list[InferenceExample] = []
                    for example in examples:
                        unit = standalone_unit_key(dataset.name, model.name, method.name, example.id)
                        if unit in completed:
                            counts["units_skipped"] += 1
                            units_processed += 1
                            continue
                        pending.append(example)

                    def _standalone_worker(
                        example: InferenceExample,
                        *,
                        _method: InferenceMethodConfig = method,
                    ) -> dict[str, Any]:
                        unit_started = time.monotonic()
                        candidate_rows = run_example_candidates(
                            config=config,
                            dataset=dataset,
                            model=model,
                            method=_method,
                            example=example,
                            backend=backend,
                        )
                        selection_row = build_selection_row(
                            config=config,
                            dataset=dataset,
                            model=model,
                            method=_method,
                            example=example,
                            candidate_rows=candidate_rows,
                        )
                        latency_s = float(
                            sum(float(r.get("latency_s") or 0.0) for r in candidate_rows)
                        ) or (time.monotonic() - unit_started)
                        return {
                            "example": example,
                            "candidate_rows": candidate_rows,
                            "selection_row": selection_row,
                            "latency_s": latency_s,
                        }

                    # Wave concurrent requests so vLLM can continuous-batch.
                    # Checkpoint successes even if some items in the wave fail.
                    for wave_start in range(0, len(pending), concurrency):
                        wave = pending[wave_start : wave_start + concurrency]
                        wave_results, wave_errors = run_concurrent_map_tolerant(
                            wave, _standalone_worker, max_workers=concurrency
                        )
                        for result in wave_results:
                            if result is None:
                                continue
                            example = result["example"]
                            candidate_rows = result["candidate_rows"]
                            selection_row = result["selection_row"]
                            unit = standalone_unit_key(
                                dataset.name, model.name, method.name, example.id
                            )
                            write_unit_batch(
                                candidate_handle,
                                selection_handle,
                                candidate_rows=candidate_rows,
                                selection_rows=[selection_row],
                            )
                            counts["model_generations"] += len(candidate_rows)
                            counts["candidate_rows"] += len(candidate_rows)
                            counts["selection_rows"] += 1

                            append_completed_unit(
                                checkpoint_path,
                                unit,
                                extra={
                                    "unit_type": "standalone",
                                    "dataset": dataset.name,
                                    "model": model.name,
                                    "method": method.name,
                                    "example_id": example.id,
                                },
                            )
                            completed.add(unit)
                            counts["units_completed_this_run"] += 1
                            units_processed += 1

                            is_correct = bool(selection_row.get("is_correct"))
                            scored_so_far += 1
                            if is_correct:
                                correct_so_far += 1
                            log_progress(
                                _format_progress_line(
                                    run_id=config.run_id,
                                    done=units_processed,
                                    total=total_units,
                                    dataset=dataset.name,
                                    model=model.name,
                                    method=method.name,
                                    example_id=example.id,
                                    is_correct=is_correct,
                                    latency_s=float(result["latency_s"]),
                                    skipped=int(counts["units_skipped"]),
                                    correct_so_far=correct_so_far,
                                    scored_so_far=scored_so_far,
                                ),
                                enabled=progress,
                            )
                        if wave_errors:
                            # Failures were not checkpointed; resume will retry them.
                            raise wave_errors[0][1]

                # Nested groups: true greedy N=1 (if configured) + sample once at max k for TTC.
                # Buffer all k slices, write as one batch, then checkpoint.
                for group_id, group_methods in nested_groups.items():
                    greedy_n1_methods, pool_methods = split_nested_group_methods(group_methods)
                    pool_n = max((m.n for m in pool_methods), default=0)
                    sample_method = (
                        pick_nested_sample_method(pool_methods, pool_n=pool_n) if pool_n > 0 else None
                    )
                    greedy_method = greedy_n1_methods[0] if greedy_n1_methods else None
                    sorted_methods = sorted(group_methods, key=lambda m: m.n)
                    log_progress(
                        f"[{config.run_id}] nested_group={group_id}  "
                        f"methods={[m.name for m in sorted_methods]}  pool_n={pool_n}  "
                        f"concurrent={concurrency}",
                        enabled=progress,
                    )
                    pending_nested: list[InferenceExample] = []
                    for example in examples:
                        unit = nested_unit_key(dataset.name, model.name, group_id, example.id)
                        if unit in completed:
                            counts["units_skipped"] += 1
                            units_processed += 1
                            continue
                        pending_nested.append(example)

                    def _nested_worker(
                        example: InferenceExample,
                        *,
                        _group_id: str = group_id,
                        _pool_n: int = pool_n,
                        _sample_method: InferenceMethodConfig | None = sample_method,
                        _greedy_method: InferenceMethodConfig | None = greedy_method,
                        _sorted_methods: list[InferenceMethodConfig] = sorted_methods,
                    ) -> dict[str, Any]:
                        unit_started = time.monotonic()
                        greedy_rows: list[dict[str, object]] = []
                        if _greedy_method is not None:
                            greedy_rows = run_example_candidates(
                                config=config,
                                dataset=dataset,
                                model=model,
                                method=_greedy_method,
                                example=example,
                                backend=backend,
                            )
                        pool: list[dict[str, object]] = []
                        if _sample_method is not None:
                            pool = run_example_candidates(
                                config=config,
                                dataset=dataset,
                                model=model,
                                method=_sample_method,
                                example=example,
                                backend=backend,
                            )

                        batch_candidates: list[dict[str, object]] = []
                        batch_selections: list[dict[str, object]] = []
                        for method in _sorted_methods:
                            if is_nested_greedy_n1(method):
                                sliced = materialize_nested_slice(
                                    greedy_rows,
                                    method=method,
                                    nested_group_id=_group_id,
                                    pool_n=_pool_n,
                                    nested_greedy_n1=True,
                                )
                            else:
                                sliced = materialize_nested_slice(
                                    pool,
                                    method=method,
                                    nested_group_id=_group_id,
                                    pool_n=_pool_n,
                                    nested_greedy_n1=False,
                                )
                            batch_candidates.extend(sliced)
                            selection_row = build_selection_row(
                                config=config,
                                dataset=dataset,
                                model=model,
                                method=method,
                                example=example,
                                candidate_rows=sliced,
                            )
                            selection_row["nested"] = True
                            selection_row["nested_group_id"] = _group_id
                            selection_row["nested_pool_n"] = _pool_n
                            if is_nested_greedy_n1(method):
                                selection_row["nested_greedy_n1"] = True
                            batch_selections.append(selection_row)

                        # Sum latency only over actual generations (not nested k-slices).
                        latency_s = float(
                            sum(float(r.get("latency_s") or 0.0) for r in greedy_rows)
                            + sum(float(r.get("latency_s") or 0.0) for r in pool)
                        ) or (time.monotonic() - unit_started)
                        return {
                            "example": example,
                            "batch_candidates": batch_candidates,
                            "batch_selections": batch_selections,
                            "latency_s": latency_s,
                            "gen_count": len(greedy_rows) + len(pool),
                        }

                    for wave_start in range(0, len(pending_nested), concurrency):
                        wave = pending_nested[wave_start : wave_start + concurrency]
                        wave_results, wave_errors = run_concurrent_map_tolerant(
                            wave, _nested_worker, max_workers=concurrency
                        )
                        for result in wave_results:
                            if result is None:
                                continue
                            example = result["example"]
                            batch_candidates = result["batch_candidates"]
                            batch_selections = result["batch_selections"]
                            unit = nested_unit_key(dataset.name, model.name, group_id, example.id)
                            write_unit_batch(
                                candidate_handle,
                                selection_handle,
                                candidate_rows=batch_candidates,
                                selection_rows=batch_selections,
                            )
                            counts["model_generations"] += int(result["gen_count"])
                            counts["candidate_rows"] += len(batch_candidates)
                            counts["selection_rows"] += len(batch_selections)

                            append_completed_unit(
                                checkpoint_path,
                                unit,
                                extra={
                                    "unit_type": "nested",
                                    "dataset": dataset.name,
                                    "model": model.name,
                                    "group_id": group_id,
                                    "example_id": example.id,
                                    "methods": [m.name for m in sorted_methods],
                                    "pool_n": pool_n,
                                    "greedy_n1": greedy_method is not None,
                                },
                            )
                            completed.add(unit)
                            counts["units_completed_this_run"] += 1
                            units_processed += 1

                            for selection_row in batch_selections:
                                scored_so_far += 1
                                if selection_row.get("is_correct"):
                                    correct_so_far += 1
                            any_correct = any(bool(r.get("is_correct")) for r in batch_selections)
                            method_label = (
                                sample_method.name if sample_method is not None else group_id
                            )
                            log_progress(
                                _format_progress_line(
                                    run_id=config.run_id,
                                    done=units_processed,
                                    total=total_units,
                                    dataset=dataset.name,
                                    model=model.name,
                                    method=method_label,
                                    example_id=example.id,
                                    is_correct=any_correct,
                                    latency_s=float(result["latency_s"]),
                                    skipped=int(counts["units_skipped"]),
                                    correct_so_far=correct_so_far,
                                    scored_so_far=scored_so_far,
                                ),
                                enabled=progress,
                            )
                        if wave_errors:
                            raise wave_errors[0][1]

    elapsed = time.monotonic() - run_started
    log_progress(
        f"[{config.run_id}] done  "
        f"completed_this_run={counts['units_completed_this_run']}  "
        f"skipped={counts['units_skipped']}  "
        f"generations={counts['model_generations']}  "
        f"elapsed_s={elapsed:.1f}  "
        f"acc={ (100.0 * correct_so_far / scored_so_far) if scored_so_far else 0.0 :.1f}% "
        f"({correct_so_far}/{scored_so_far})",
        enabled=progress,
    )

    manifest = {
        "run_id": config.run_id,
        "output_dir": str(output_dir),
        "candidates_path": str(candidates_path),
        "selections_path": str(selections_path),
        "checkpoint_path": str(checkpoint_path),
        "datasets": [dataset.name for dataset in datasets],
        "models": [{"name": model.name, "size_label": model.size_label} for model in models],
        "methods": [
            {
                "name": method.name,
                "prompt_style": method.prompt_style,
                "selection": method.selection,
                "n": method.n,
                "max_tokens": method.max_tokens,
                "nested_group_id": method.nested_group_id,
                "nested_max_n": method.nested_max_n,
            }
            for method in methods
        ],
        "nested_groups": {
            group_id: {
                "methods": [m.name for m in sorted(group, key=lambda x: x.n)],
                "pool_n": max((m.n for m in group if not is_nested_greedy_n1(m)), default=0),
                "greedy_n1": any(is_nested_greedy_n1(m) for m in group),
            }
            for group_id, group in nested_groups.items()
        },
        "counts": counts,
        "total_completed_units": len(completed),
        "max_concurrent": config.max_concurrent,
    }
    write_json(manifest_path, manifest)
    return manifest


def flush_handles(*handles: Any) -> None:
    for handle in handles:
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass


def partition_methods(
    methods: list[InferenceMethodConfig],
) -> tuple[list[InferenceMethodConfig], dict[str, list[InferenceMethodConfig]]]:
    """Split methods into standalone vs nested groups."""
    nested_groups: dict[str, list[InferenceMethodConfig]] = defaultdict(list)
    standalone: list[InferenceMethodConfig] = []
    for method in methods:
        if method.nested_group_id:
            nested_groups[method.nested_group_id].append(method)
        else:
            standalone.append(method)
    return standalone, dict(nested_groups)


def is_nested_greedy_n1(method: InferenceMethodConfig) -> bool:
    """True when this method is a true-greedy N=1 arm inside a nested group.

    Detected as ``n == 1``, ``selection == first``, and non-positive temperature
    (``None`` or ``<= 0``). Config expansion sets this when ``greedy_n1`` is true.
    """
    if method.n != 1 or method.selection != "first":
        return False
    return method.temperature is None or method.temperature <= 0.0


def split_nested_group_methods(
    group_methods: list[InferenceMethodConfig],
) -> tuple[list[InferenceMethodConfig], list[InferenceMethodConfig]]:
    """Split nested methods into true-greedy N=1 arm(s) vs stochastic pool arms."""
    greedy_n1: list[InferenceMethodConfig] = []
    pool_methods: list[InferenceMethodConfig] = []
    for method in group_methods:
        if is_nested_greedy_n1(method):
            greedy_n1.append(method)
        else:
            pool_methods.append(method)
    return greedy_n1, pool_methods


def pick_nested_sample_method(
    group_methods: list[InferenceMethodConfig],
    *,
    pool_n: int,
) -> InferenceMethodConfig:
    """Choose generation settings for the shared nested stochastic pool.

    Uses the largest-N method's prompt/max_tokens. Sampling temperature comes from
    the largest-N method (typically temp>0 for BoN). All pool samples share that
    temperature so prefixes are nested draws from one process.

    True-greedy N=1 methods should be excluded from ``group_methods`` (see
    ``split_nested_group_methods``); they are generated separately.
    """
    if not group_methods:
        raise ValueError("pick_nested_sample_method requires at least one pool method")
    if pool_n < 1:
        raise ValueError(f"pick_nested_sample_method requires pool_n >= 1, got {pool_n}")
    largest = max(group_methods, key=lambda m: m.n)
    # Prefer a method whose n equals pool_n (should be largest).
    for method in group_methods:
        if method.n == pool_n:
            largest = method
            break
    # Generation method always has n=pool_n and the pool sampling temperature.
    return replace(largest, n=pool_n, name=f"{largest.nested_group_id or largest.name}_pool_n{pool_n}")


def materialize_nested_slice(
    pool: list[dict[str, object]],
    *,
    method: InferenceMethodConfig,
    nested_group_id: str,
    pool_n: int,
    nested_greedy_n1: bool = False,
) -> list[dict[str, object]]:
    """Project the first method.n pool samples into a condition with method.n.

    For true-greedy N=1, ``pool`` is the separate greedy generation (length 1).
    """
    k = method.n
    if k > len(pool):
        raise ValueError(
            f"Nested method {method.name!r} requests n={k} but pool only has {len(pool)} samples"
        )
    sliced: list[dict[str, object]] = []
    for row in pool[:k]:
        new_row = dict(row)
        meta = dict(new_row.get("metadata") or {})
        meta["nested"] = True
        meta["nested_group_id"] = nested_group_id
        meta["nested_pool_n"] = pool_n
        if nested_greedy_n1:
            meta["nested_greedy_n1"] = True
        new_row["metadata"] = meta
        new_row["method"] = method.name
        new_row["n"] = k
        new_row["selection"] = method.selection
        new_row["prompt_style"] = method.prompt_style
        new_row["reasoning_language"] = method.reasoning_language
        sliced.append(new_row)
    return sliced


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
        seed = derive_sample_seed(config.seed, example.id, sample_index)
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
                "question": example.question,
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
    selected = select_candidate(
        candidate_rows,
        method.selection,
        answer_type=example.answer_type,
    )
    row: dict[str, Any] = {
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
    if selected.metadata:
        row["selection_metadata"] = selected.metadata
    return row


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
