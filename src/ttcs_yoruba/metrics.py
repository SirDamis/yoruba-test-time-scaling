"""Aggregation metrics for TTC scaling experiments (E2 / E3 / E4)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .extraction import is_exact_match
from .io_utils import read_json, read_jsonl


@dataclass
class ConditionMetrics:
    """Metrics for one (dataset, model, method, n) condition."""

    dataset: str
    model: str
    model_size_label: str
    method: str
    prompt_style: str
    reasoning_language: str
    selection: str
    n: int
    total_examples: int = 0
    select_correct: int = 0
    pass_at_n_correct: int = 0
    total_candidates: int = 0
    total_tokens: int = 0
    total_latency_s: float = 0.0
    total_estimated_cost: float = 0.0
    run_ids: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.select_correct / self.total_examples if self.total_examples else 0.0

    @property
    def pass_at_n_rate(self) -> float:
        return self.pass_at_n_correct / self.total_examples if self.total_examples else 0.0

    @property
    def gap(self) -> float:
        return self.pass_at_n_rate - self.accuracy

    @property
    def mean_tokens_per_example(self) -> float:
        return self.total_tokens / self.total_examples if self.total_examples else 0.0

    @property
    def mean_latency_s_per_example(self) -> float:
        return self.total_latency_s / self.total_examples if self.total_examples else 0.0

    @property
    def tokens_per_correct(self) -> float | None:
        if self.select_correct <= 0:
            return None
        return self.total_tokens / self.select_correct

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "model_size_label": self.model_size_label,
            "method": self.method,
            "prompt_style": self.prompt_style,
            "reasoning_language": self.reasoning_language,
            "selection": self.selection,
            "n": self.n,
            "total_examples": self.total_examples,
            "select_correct": self.select_correct,
            "accuracy": self.accuracy,
            "pass_at_n_correct": self.pass_at_n_correct,
            "pass_at_n_rate": self.pass_at_n_rate,
            "gap": self.gap,
            "total_candidates": self.total_candidates,
            "total_tokens": self.total_tokens,
            "mean_tokens_per_example": self.mean_tokens_per_example,
            "total_latency_s": self.total_latency_s,
            "mean_latency_s_per_example": self.mean_latency_s_per_example,
            "total_estimated_cost": self.total_estimated_cost,
            "tokens_per_correct": self.tokens_per_correct,
            "run_ids": sorted(set(self.run_ids)),
        }


def candidate_dataset(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    return str(row.get("dataset") or meta.get("dataset") or "")


def condition_key_from_candidate(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        candidate_dataset(row),
        str(row.get("model", "")),
        str(row.get("method", "")),
        int(row.get("n") or 1),
    )


def condition_key_from_selection(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("method", "")),
        int(row.get("n") or 1),
    )


def candidate_row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    meta = row.get("metadata") or {}
    return (
        candidate_dataset(row),
        str(row.get("model", "")),
        str(row.get("method", "")),
        int(row.get("n") or 1),
        str(row.get("example_id", "")),
        int(row.get("sample_index") or 0),
    )


def selection_row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("method", "")),
        int(row.get("n") or 1),
        str(row.get("example_id", "")),
    )


def _dedupe_keep_last(rows: list[dict[str, Any]], identity_fn) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = identity_fn(row)
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    return [by_key[key] for key in order]


def dedupe_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate candidate rows from resume appends (last write wins)."""
    return _dedupe_keep_last(rows, candidate_row_identity)


def dedupe_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate selection rows from resume appends (last write wins)."""
    return _dedupe_keep_last(rows, selection_row_identity)


# Directories that are never real experiment runs (smoke, samples, caches).
DEFAULT_RUN_DIR_EXCLUDE_NAMES = frozenset(
    {
        "sample_prompts",
        "__pycache__",
        ".git",
        "_smoke",
        "smoke",
        "synthetic",
        "synthetic_e2",
    }
)


def find_run_dirs(
    base_dir: Path,
    *,
    require_manifest: bool = True,
    require_selections: bool = False,
    exclude_names: frozenset[str] | set[str] | None = None,
    include_hidden: bool = False,
) -> list[Path]:
    """List experiment run directories under ``base_dir``.

    A valid run directory must contain ``candidates.jsonl``. By default it must
    also contain ``manifest.json`` so smoke folders / loose dumps are skipped.

    Excludes known non-run names (``sample_prompts``, ``_smoke``, ``synthetic*``,
    names starting with ``_`` unless ``include_hidden`` is true).
    """
    if not base_dir.exists():
        return []

    excluded = set(DEFAULT_RUN_DIR_EXCLUDE_NAMES)
    if exclude_names is not None:
        excluded |= set(exclude_names)

    runs: list[Path] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name in excluded:
            continue
        if name.startswith(".") or (name.startswith("_") and not include_hidden):
            continue
        # Common smoke / synthetic naming patterns.
        lower = name.lower()
        if "smoke" in lower or lower.startswith("synthetic"):
            continue
        if not (path / "candidates.jsonl").exists():
            continue
        if require_manifest and not (path / "manifest.json").exists():
            continue
        if require_selections and not (path / "selections.jsonl").exists():
            continue
        # Skip empty candidate dumps.
        try:
            if (path / "candidates.jsonl").stat().st_size == 0:
                continue
        except OSError:
            continue
        runs.append(path)
    return runs


def aggregate_run_dir(run_dir: Path) -> list[ConditionMetrics]:
    """Aggregate one run directory into per-condition metrics."""
    candidates_path = run_dir / "candidates.jsonl"
    selections_path = run_dir / "selections.jsonl"
    manifest_path = run_dir / "manifest.json"

    if not candidates_path.exists():
        return []

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    run_id = str(manifest.get("run_id", run_dir.name))

    candidates = dedupe_candidate_rows(read_jsonl(candidates_path))
    selections = (
        dedupe_selection_rows(read_jsonl(selections_path)) if selections_path.exists() else []
    )

    # Pass@N: group candidates by condition + example.
    by_condition_example: dict[tuple[str, str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    condition_meta: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    token_totals: dict[tuple[str, str, str, int], int] = defaultdict(int)
    latency_totals: dict[tuple[str, str, str, int], float] = defaultdict(float)
    cost_totals: dict[tuple[str, str, str, int], float] = defaultdict(float)
    candidate_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)

    for row in candidates:
        key = condition_key_from_candidate(row)
        example_id = str(row.get("example_id", ""))
        by_condition_example[key][example_id].append(row)
        candidate_counts[key] += 1
        token_totals[key] += int(row.get("token_count") or 0)
        latency_totals[key] += float(row.get("latency_s") or 0.0)
        cost_totals[key] += float(row.get("estimated_cost") or 0.0)
        if key not in condition_meta:
            condition_meta[key] = {
                "prompt_style": str(row.get("prompt_style", "")),
                "reasoning_language": str(row.get("reasoning_language", "")),
                "selection": str(row.get("selection", "")),
                "model_size_label": str(row.get("model_size_label", "")),
            }

    # Select@N from selection rows.
    selection_correct: dict[tuple[str, str, str, int], dict[str, bool]] = defaultdict(dict)
    for row in selections:
        key = condition_key_from_selection(row)
        example_id = str(row.get("example_id", ""))
        selection_correct[key][example_id] = bool(row.get("is_correct", False))
        if key not in condition_meta:
            condition_meta[key] = {
                "prompt_style": str(row.get("prompt_style", "")),
                "reasoning_language": str(row.get("reasoning_language", "")),
                "selection": str(row.get("selection", "")),
                "model_size_label": str(row.get("model_size_label", "")),
            }

    all_keys = set(by_condition_example) | set(selection_correct)
    results: list[ConditionMetrics] = []

    for key in sorted(all_keys):
        dataset, model, method, n = key
        meta = condition_meta.get(key, {})
        example_map = by_condition_example.get(key, {})

        # Prefer selection example set if present; else candidates.
        example_ids = sorted(set(example_map) | set(selection_correct.get(key, {})))
        if not example_ids and not example_map:
            continue

        select_map = selection_correct.get(key, {})
        if select_map:
            total_examples = len(select_map)
            select_ok = sum(1 for value in select_map.values() if value)
            example_ids_for_pass = list(select_map.keys())
        else:
            total_examples = len(example_ids)
            select_ok = 0
            example_ids_for_pass = example_ids

        pass_correct = 0
        for example_id in example_ids_for_pass:
            cands = example_map.get(example_id, [])
            any_ok = False
            for cand in cands:
                meta_row = cand.get("metadata") or {}
                gold = meta_row.get("gold_answer", "")
                answer_type = meta_row.get("answer_type", "")
                pred = cand.get("extracted_answer", "")
                if is_exact_match(str(pred), str(gold or ""), str(answer_type or "")):
                    any_ok = True
                    break
            if any_ok:
                pass_correct += 1

        metrics = ConditionMetrics(
            dataset=dataset,
            model=model,
            model_size_label=str(meta.get("model_size_label", "")),
            method=method,
            prompt_style=str(meta.get("prompt_style", "")),
            reasoning_language=str(meta.get("reasoning_language", "")),
            selection=str(meta.get("selection", "")),
            n=n,
            total_examples=total_examples,
            select_correct=select_ok,
            pass_at_n_correct=pass_correct,
            total_candidates=candidate_counts.get(key, 0),
            total_tokens=token_totals.get(key, 0),
            total_latency_s=latency_totals.get(key, 0.0),
            total_estimated_cost=cost_totals.get(key, 0.0),
            run_ids=[run_id],
        )
        results.append(metrics)

    return results


def _clone_condition_metrics(item: ConditionMetrics) -> ConditionMetrics:
    return ConditionMetrics(
        dataset=item.dataset,
        model=item.model,
        model_size_label=item.model_size_label,
        method=item.method,
        prompt_style=item.prompt_style,
        reasoning_language=item.reasoning_language,
        selection=item.selection,
        n=item.n,
        total_examples=item.total_examples,
        select_correct=item.select_correct,
        pass_at_n_correct=item.pass_at_n_correct,
        total_candidates=item.total_candidates,
        total_tokens=item.total_tokens,
        total_latency_s=item.total_latency_s,
        total_estimated_cost=item.total_estimated_cost,
        run_ids=list(item.run_ids),
    )


def _contribution_run_ids(item: ConditionMetrics, *, anon_counter: list[int]) -> frozenset[str]:
    """Stable run-id set used to detect duplicate merge contributions."""
    rids = {str(r) for r in item.run_ids if r not in (None, "")}
    if rids:
        return frozenset(rids)
    # No run_id: treat each such item as unique so we still sum true shards
    # that forgot to set run_id (cannot safely dedupe them).
    anon_counter[0] += 1
    return frozenset({f"__anon_{anon_counter[0]}"})


def _add_into(acc: ConditionMetrics, item: ConditionMetrics) -> None:
    acc.total_examples += item.total_examples
    acc.select_correct += item.select_correct
    acc.pass_at_n_correct += item.pass_at_n_correct
    acc.total_candidates += item.total_candidates
    acc.total_tokens += item.total_tokens
    acc.total_latency_s += item.total_latency_s
    acc.total_estimated_cost += item.total_estimated_cost
    for rid in item.run_ids:
        if rid not in (None, "") and rid not in acc.run_ids:
            acc.run_ids.append(str(rid))
    if not acc.model_size_label and item.model_size_label:
        acc.model_size_label = item.model_size_label
    if not acc.prompt_style and item.prompt_style:
        acc.prompt_style = item.prompt_style
    if not acc.reasoning_language and item.reasoning_language:
        acc.reasoning_language = item.reasoning_language
    if not acc.selection and item.selection:
        acc.selection = item.selection


def merge_condition_metrics(items: Iterable[ConditionMetrics]) -> list[ConditionMetrics]:
    """Merge metrics that share the same condition key (dataset, model, method, n).

    **Safe merge rules (avoids double-counting the same run):**

    - Contributions are tracked by ``run_id`` set.
    - If an incoming item's run_ids are **already fully covered** by the
      accumulator, it is treated as a duplicate and **skipped** (or replaced
      if it reports strictly more examples for the same run set).
    - If run_ids are **disjoint**, counts are **summed** (true multi-shard merge).
    - If run_ids **partially overlap**, the incoming item is **skipped**
      (conservative: avoids partial double-count when we cannot split examples).
    - Items with empty ``run_ids`` each get a unique anonymous id (summed).
    """
    buckets: dict[tuple[str, str, str, int], ConditionMetrics] = {}
    covered_runs: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    # Track the full run-id set of the last full contribution for replace logic.
    contribution_sets: dict[tuple[str, str, str, int], list[frozenset[str]]] = defaultdict(list)
    anon_counter = [0]

    for item in items:
        key = (item.dataset, item.model, item.method, item.n)
        incoming = _contribution_run_ids(item, anon_counter=anon_counter)

        if key not in buckets:
            buckets[key] = _clone_condition_metrics(item)
            # Normalize run_ids on the clone for anonymous contributions.
            if not item.run_ids:
                buckets[key].run_ids = sorted(incoming)
            covered_runs[key] = set(incoming)
            contribution_sets[key].append(incoming)
            continue

        acc = buckets[key]
        covered = covered_runs[key]
        overlap = incoming & covered

        if not overlap:
            # Disjoint shards — safe to sum.
            _add_into(acc, item)
            covered |= set(incoming)
            contribution_sets[key].append(incoming)
            continue

        if incoming <= covered:
            # Duplicate (or subset) of already-merged runs.
            # Replace only if this is the exact same run set and larger.
            if incoming in contribution_sets[key] and item.total_examples > acc.total_examples:
                # Only safe to replace when the accumulator is solely from this same set.
                if covered == set(incoming) and len(contribution_sets[key]) == 1:
                    buckets[key] = _clone_condition_metrics(item)
                    if not item.run_ids:
                        buckets[key].run_ids = sorted(incoming)
                    covered_runs[key] = set(incoming)
                    contribution_sets[key] = [incoming]
            # else: skip duplicate
            continue

        # Partial overlap: cannot safely sum without double-counting shared runs.
        # Skip the incoming contribution.
        continue

    return [buckets[key] for key in sorted(buckets)]


def aggregate_runs(run_dirs: list[Path], *, merge: bool = True) -> list[ConditionMetrics]:
    """Aggregate one or more run directories.

    When ``merge`` is true, conditions with the same (dataset, model, method, n)
    are combined **only across distinct ``run_id``s**. The same run is never
    counted twice.
    """
    collected: list[ConditionMetrics] = []
    seen_dirs: set[Path] = set()
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        collected.extend(aggregate_run_dir(run_dir))
    if merge:
        return merge_condition_metrics(collected)
    return collected


def metrics_to_csv_rows(metrics: list[ConditionMetrics]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in metrics]


def write_metrics_csv(path: Path, metrics: list[ConditionMetrics]) -> None:
    rows = metrics_to_csv_rows(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    lines = [",".join(fieldnames)]
    for row in rows:
        values = []
        for name in fieldnames:
            value = row[name]
            if isinstance(value, list):
                text = ";".join(str(v) for v in value)
            elif value is None:
                text = ""
            else:
                text = str(value)
            if any(ch in text for ch in [",", '"', "\n"]):
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_accuracy_vs_n(
    metrics: list[ConditionMetrics],
    output_path: Path,
    *,
    dataset: str | None = None,
) -> Path | None:
    """Plot accuracy vs N, one line per model (optionally filter to one dataset)."""
    return _plot_scaling(
        metrics,
        output_path,
        x_attr="n",
        x_label="N (number of samples)",
        title="Accuracy vs N",
        dataset=dataset,
        use_tokens_x=False,
    )


def plot_accuracy_vs_tokens(
    metrics: list[ConditionMetrics],
    output_path: Path,
    *,
    dataset: str | None = None,
) -> Path | None:
    """Plot accuracy vs total tokens, one line per model."""
    return _plot_scaling(
        metrics,
        output_path,
        x_attr="total_tokens",
        x_label="Total completion tokens",
        title="Accuracy vs Total Tokens",
        dataset=dataset,
        use_tokens_x=True,
    )


def _plot_scaling(
    metrics: list[ConditionMetrics],
    output_path: Path,
    *,
    x_attr: str,
    x_label: str,
    title: str,
    dataset: str | None,
    use_tokens_x: bool,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    filtered = [m for m in metrics if dataset is None or m.dataset == dataset]
    if not filtered:
        return None

    # Group by (dataset, model) series, sorted by x.
    series: dict[tuple[str, str], list[ConditionMetrics]] = defaultdict(list)
    for item in filtered:
        series[(item.dataset, item.model)].append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for (ds, model), points in sorted(series.items()):
        points = sorted(points, key=lambda p: (p.n, getattr(p, x_attr)))
        xs = [getattr(p, x_attr) for p in points]
        ys = [p.accuracy for p in points]
        label = f"{model}" if dataset or len({m.dataset for m in filtered}) == 1 else f"{model} | {ds}"
        ax.plot(xs, ys, marker="o", label=label)
        # Annotate N on token plots for readability.
        if use_tokens_x:
            for p in points:
                ax.annotate(
                    f"N={p.n}",
                    (p.total_tokens, p.accuracy),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (select@N)")
    plot_title = title if dataset is None else f"{title} ({dataset})"
    ax.set_title(plot_title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
