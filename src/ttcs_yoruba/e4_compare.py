"""E4: small model + TTC vs larger model greedy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_json, write_json
from .metrics import ConditionMetrics, aggregate_runs, find_run_dirs


@dataclass(frozen=True)
class E4ComparisonConfig:
    """Which models/Ns define the E4 paper comparison."""

    small_model: str = "qwen3-4b"
    large_model: str = "qwen3-32b"
    large_n: int = 1
    # Optional intermediate ladder models to include in full tables (not required for match).
    include_models: tuple[str, ...] = ()
    datasets: tuple[str, ...] | None = None
    # Method filters: exact name, prefix (``english_cot_ttc``), or glob (``english_cot_ttc*``).
    # Prevents collisions when multiple methods share the same model and N.
    small_method: str | None = "english_cot_ttc"
    large_method: str | None = "english_cot_ttc"
    # Method filter for include_models ladder rows (defaults to large_method).
    ladder_method: str | None = None


def condition_metrics_from_dict(row: dict[str, Any]) -> ConditionMetrics:
    metrics = ConditionMetrics(
        dataset=str(row.get("dataset", "")),
        model=str(row.get("model", "")),
        model_size_label=str(row.get("model_size_label", "")),
        method=str(row.get("method", "")),
        prompt_style=str(row.get("prompt_style", "")),
        reasoning_language=str(row.get("reasoning_language", "")),
        selection=str(row.get("selection", "")),
        n=int(row.get("n") or 1),
        total_examples=int(row.get("total_examples") or 0),
        select_correct=int(row.get("select_correct") or 0),
        pass_at_n_correct=int(row.get("pass_at_n_correct") or 0),
        total_candidates=int(row.get("total_candidates") or 0),
        total_tokens=int(row.get("total_tokens") or 0),
        total_latency_s=float(row.get("total_latency_s") or 0.0),
        total_estimated_cost=float(row.get("total_estimated_cost") or 0.0),
        run_ids=list(row.get("run_ids") or []),
    )
    return metrics


def load_metrics_from_json(path: Path) -> list[ConditionMetrics]:
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("conditions") or []
    return [condition_metrics_from_dict(row) for row in rows]


def load_metrics_from_runs(runs_dir: Path, *, run_id: str | None = None) -> list[ConditionMetrics]:
    if run_id:
        run_dirs = [runs_dir / run_id]
    else:
        run_dirs = find_run_dirs(runs_dir)
    run_dirs = [path for path in run_dirs if path.exists()]
    if not run_dirs:
        return []
    return aggregate_runs(run_dirs, merge=True)


def efficiency_fields(m: ConditionMetrics) -> dict[str, Any]:
    """Token / latency / cost efficiency derived from condition metrics."""
    tokens_per_correct = m.tokens_per_correct
    latency_per_correct = (
        None if m.select_correct <= 0 else m.total_latency_s / m.select_correct
    )
    accuracy_per_1k_tokens = (
        None if m.total_tokens <= 0 else m.accuracy / (m.total_tokens / 1000.0)
    )
    accuracy_per_cost = (
        None
        if m.total_estimated_cost <= 0
        else m.accuracy / m.total_estimated_cost
    )
    return {
        "tokens_per_correct": tokens_per_correct,
        "latency_s_per_correct": latency_per_correct,
        "mean_tokens_per_example": m.mean_tokens_per_example,
        "mean_latency_s_per_example": m.mean_latency_s_per_example,
        "accuracy_per_1k_tokens": accuracy_per_1k_tokens,
        "accuracy_per_cost": accuracy_per_cost,
        "total_estimated_cost": m.total_estimated_cost,
    }


def method_matches(method_name: str, filter_spec: str | None) -> bool:
    """Return True if ``method_name`` matches an optional filter.

    Filter forms:
    - ``None``: match any method
    - exact: ``english_cot_ttc_n4``
    - prefix: ``english_cot_ttc`` matches ``english_cot_ttc`` and ``english_cot_ttc_n4``
    - glob suffix ``*``: ``english_cot_ttc*`` same as prefix
    """
    if filter_spec is None or filter_spec == "" or filter_spec == "*":
        return True
    spec = filter_spec.strip()
    if spec.endswith("*"):
        return method_name.startswith(spec[:-1])
    if method_name == spec:
        return True
    # Prefix so ``english_cot_ttc`` matches expanded ``english_cot_ttc_n8``.
    return method_name.startswith(spec + "_") or method_name.startswith(spec + "-")


def index_metrics(
    metrics: Iterable[ConditionMetrics],
) -> dict[tuple[str, str, str, int], ConditionMetrics]:
    """Index by (dataset, model, method, n). If exact key duplicates, keep more examples."""
    index: dict[tuple[str, str, str, int], ConditionMetrics] = {}
    for item in metrics:
        key = (item.dataset, item.model, item.method, item.n)
        existing = index.get(key)
        if existing is None or item.total_examples > existing.total_examples:
            index[key] = item
    return index


def resolve_condition(
    indexed: dict[tuple[str, str, str, int], ConditionMetrics],
    *,
    dataset: str,
    model: str,
    n: int,
    method_filter: str | None,
) -> ConditionMetrics | None:
    """Pick one condition for (dataset, model, n) under an optional method filter.

    If several methods match, prefer the largest ``total_examples``, then
    lexicographically smallest method name (stable).
    """
    matches = [
        metrics
        for (ds, mo, method, nn), metrics in indexed.items()
        if ds == dataset and mo == model and nn == n and method_matches(method, method_filter)
    ]
    if not matches:
        return None
    matches.sort(key=lambda m: (-m.total_examples, m.method))
    return matches[0]


def build_e4_comparison(
    metrics: list[ConditionMetrics],
    *,
    config: E4ComparisonConfig | None = None,
) -> dict[str, Any]:
    """Compare small-model TTC curve against large-model greedy baseline.

    For each dataset and each small-model N:
    - accuracy / tokens / latency for small@N
    - large@large_n baseline
    - deltas and whether small matches or beats large
    Also reports the minimal N (if any) where small accuracy >= large accuracy.

    Conditions are keyed by ``(dataset, model, method, n)``. Use
    ``small_method`` / ``large_method`` filters when multiple methods exist.
    """
    cfg = config or E4ComparisonConfig()
    indexed = index_metrics(metrics)
    ladder_method = cfg.ladder_method if cfg.ladder_method is not None else cfg.large_method

    datasets = sorted({m.dataset for m in metrics if m.dataset})
    if cfg.datasets is not None:
        allowed = set(cfg.datasets)
        datasets = [d for d in datasets if d in allowed]

    per_dataset: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        large = resolve_condition(
            indexed,
            dataset=dataset,
            model=cfg.large_model,
            n=cfg.large_n,
            method_filter=cfg.large_method,
        )
        small_ns = sorted(
            {
                m.n
                for m in metrics
                if m.dataset == dataset
                and m.model == cfg.small_model
                and method_matches(m.method, cfg.small_method)
            }
        )

        large_payload = None if large is None else _condition_payload(large)
        comparisons: list[dict[str, Any]] = []
        min_n_match: int | None = None

        for n in small_ns:
            small = resolve_condition(
                indexed,
                dataset=dataset,
                model=cfg.small_model,
                n=n,
                method_filter=cfg.small_method,
            )
            if small is None:
                continue
            row = {
                "dataset": dataset,
                "small_model": cfg.small_model,
                "small_method": small.method,
                "small_n": n,
                "small": _condition_payload(small),
                "large_model": cfg.large_model,
                "large_method": None if large is None else large.method,
                "large_n": cfg.large_n,
                "large": large_payload,
            }
            if large is not None:
                acc_delta = small.accuracy - large.accuracy
                matches = small.accuracy + 1e-12 >= large.accuracy
                token_ratio = (
                    None
                    if large.total_tokens <= 0
                    else small.total_tokens / large.total_tokens
                )
                latency_ratio = (
                    None
                    if large.total_latency_s <= 0
                    else small.total_latency_s / large.total_latency_s
                )
                row.update(
                    {
                        "accuracy_delta": acc_delta,
                        "matches_or_beats_large": matches,
                        "token_ratio_vs_large": token_ratio,
                        "latency_ratio_vs_large": latency_ratio,
                    }
                )
                if matches and min_n_match is None:
                    min_n_match = n
            else:
                row.update(
                    {
                        "accuracy_delta": None,
                        "matches_or_beats_large": None,
                        "token_ratio_vs_large": None,
                        "latency_ratio_vs_large": None,
                    }
                )
            comparisons.append(row)
            paper_rows.append(row)

        # Optional ladder models (e.g. 14B) at greedy N for context.
        ladder: list[dict[str, Any]] = []
        for model_name in cfg.include_models:
            m = resolve_condition(
                indexed,
                dataset=dataset,
                model=model_name,
                n=cfg.large_n,
                method_filter=ladder_method,
            )
            if m is not None:
                ladder.append(_condition_payload(m))

        per_dataset.append(
            {
                "dataset": dataset,
                "large_baseline": large_payload,
                "small_model": cfg.small_model,
                "small_method_filter": cfg.small_method,
                "large_method_filter": cfg.large_method,
                "comparisons": comparisons,
                "min_n_to_match_large": min_n_match,
                "ladder_greedy": ladder,
            }
        )

    summary = {
        "config": {
            "small_model": cfg.small_model,
            "large_model": cfg.large_model,
            "large_n": cfg.large_n,
            "include_models": list(cfg.include_models),
            "datasets": list(cfg.datasets) if cfg.datasets is not None else None,
            "small_method": cfg.small_method,
            "large_method": cfg.large_method,
            "ladder_method": ladder_method,
        },
        "num_datasets": len(per_dataset),
        "datasets": per_dataset,
        "paper_table": paper_rows,
        "headline": _headline_summary(per_dataset, cfg),
    }
    return summary


def _condition_payload(m: ConditionMetrics) -> dict[str, Any]:
    base = m.to_dict()
    base.update(efficiency_fields(m))
    return base


def _headline_summary(per_dataset: list[dict[str, Any]], cfg: E4ComparisonConfig) -> dict[str, Any]:
    """Aggregate across datasets: how often small@N matches large greedy."""
    matched_datasets = []
    unmatched_datasets = []
    missing_large = []
    min_ns: list[int] = []

    for block in per_dataset:
        dataset = block["dataset"]
        if block["large_baseline"] is None:
            missing_large.append(dataset)
            continue
        min_n = block.get("min_n_to_match_large")
        if min_n is None:
            unmatched_datasets.append(dataset)
        else:
            matched_datasets.append({"dataset": dataset, "min_n": min_n})
            min_ns.append(int(min_n))

    return {
        "small_model": cfg.small_model,
        "large_model": cfg.large_model,
        "large_n": cfg.large_n,
        "datasets_matched": matched_datasets,
        "datasets_unmatched": unmatched_datasets,
        "datasets_missing_large_baseline": missing_large,
        "median_min_n_to_match": _median_int(min_ns) if min_ns else None,
    }


def _median_int(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def print_e4_table(report: dict[str, Any]) -> None:
    """Print a paper-style summary table to stdout."""
    rows = report.get("paper_table") or []
    if not rows:
        print("No E4 comparison rows.")
        return

    headers = [
        "Dataset",
        "Small@N",
        "S-Acc",
        "L-Acc",
        "ΔAcc",
        "Match?",
        "S-Tok/ex",
        "L-Tok/ex",
        "S-Tok/corr",
        "L-Tok/corr",
        "S-Lat/ex",
        "L-Lat/ex",
    ]
    print("\n" + "  ".join(headers))
    print("-" * 120)
    for row in rows:
        small = row.get("small") or {}
        large = row.get("large") or {}
        match = row.get("matches_or_beats_large")
        match_s = "—" if match is None else ("yes" if match else "no")
        delta = row.get("accuracy_delta")
        delta_s = "—" if delta is None else f"{delta:+.1%}"
        large_acc = "—" if not large else f"{large.get('accuracy', 0.0):.1%}"
        s_tpc = small.get("tokens_per_correct")
        l_tpc = large.get("tokens_per_correct") if large else None
        print(
            "  ".join(
                [
                    str(row.get("dataset", ""))[:12],
                    f"{row.get('small_model', '')}@N={row.get('small_n')}",
                    f"{small.get('accuracy', 0.0):.1%}",
                    large_acc,
                    delta_s,
                    match_s,
                    f"{small.get('mean_tokens_per_example', 0.0):.1f}",
                    "—" if not large else f"{large.get('mean_tokens_per_example', 0.0):.1f}",
                    "—" if s_tpc is None else f"{s_tpc:.1f}",
                    "—" if l_tpc is None else f"{l_tpc:.1f}",
                    f"{small.get('mean_latency_s_per_example', 0.0):.2f}",
                    "—" if not large else f"{large.get('mean_latency_s_per_example', 0.0):.2f}",
                ]
            )
        )

    headline = report.get("headline") or {}
    print("\nHeadline")
    print(
        f"  Small={headline.get('small_model')}  Large={headline.get('large_model')}@N={headline.get('large_n')}"
    )
    print(f"  Matched datasets: {headline.get('datasets_matched')}")
    print(f"  Unmatched datasets: {headline.get('datasets_unmatched')}")
    print(f"  Median min N to match large: {headline.get('median_min_n_to_match')}")


def write_e4_paper_csv(path: Path, report: dict[str, Any]) -> None:
    """Flat CSV for paper tables / spreadsheets."""
    rows = report.get("paper_table") or []
    flat: list[dict[str, Any]] = []
    for row in rows:
        small = row.get("small") or {}
        large = row.get("large") or {}
        flat.append(
            {
                "dataset": row.get("dataset"),
                "small_model": row.get("small_model"),
                "small_method": row.get("small_method"),
                "small_n": row.get("small_n"),
                "small_accuracy": small.get("accuracy"),
                "small_pass_at_n_rate": small.get("pass_at_n_rate"),
                "small_mean_tokens_per_example": small.get("mean_tokens_per_example"),
                "small_mean_latency_s_per_example": small.get("mean_latency_s_per_example"),
                "small_tokens_per_correct": small.get("tokens_per_correct"),
                "small_latency_s_per_correct": small.get("latency_s_per_correct"),
                "small_accuracy_per_1k_tokens": small.get("accuracy_per_1k_tokens"),
                "small_total_estimated_cost": small.get("total_estimated_cost"),
                "large_model": row.get("large_model"),
                "large_method": row.get("large_method"),
                "large_n": row.get("large_n"),
                "large_accuracy": None if not large else large.get("accuracy"),
                "large_mean_tokens_per_example": None
                if not large
                else large.get("mean_tokens_per_example"),
                "large_mean_latency_s_per_example": None
                if not large
                else large.get("mean_latency_s_per_example"),
                "large_tokens_per_correct": None if not large else large.get("tokens_per_correct"),
                "large_latency_s_per_correct": None
                if not large
                else large.get("latency_s_per_correct"),
                "accuracy_delta": row.get("accuracy_delta"),
                "matches_or_beats_large": row.get("matches_or_beats_large"),
                "token_ratio_vs_large": row.get("token_ratio_vs_large"),
                "latency_ratio_vs_large": row.get("latency_ratio_vs_large"),
            }
        )
    if not flat:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    # Reuse simple CSV writer from metrics.
    fieldnames = list(flat[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(fieldnames)]
    for row in flat:
        values = []
        for name in fieldnames:
            value = row[name]
            if value is None:
                text = ""
            elif isinstance(value, bool):
                text = "true" if value else "false"
            else:
                text = str(value)
            if any(ch in text for ch in [",", '"', "\n"]):
                text = '"' + text.replace('"', '""') + '"'
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_e4_small_vs_large(
    report: dict[str, Any],
    output_path: Path,
    *,
    dataset: str | None = None,
) -> Path | None:
    """Plot small-model accuracy vs N with large greedy as horizontal reference."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    blocks = report.get("datasets") or []
    if dataset is not None:
        blocks = [b for b in blocks if b.get("dataset") == dataset]
    if not blocks:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for block in blocks:
        comps = block.get("comparisons") or []
        if not comps:
            continue
        xs = [c["small_n"] for c in comps]
        ys = [(c.get("small") or {}).get("accuracy", 0.0) for c in comps]
        label = f"{block.get('small_model')} ({block.get('dataset')})"
        ax.plot(xs, ys, marker="o", label=label)

        large = block.get("large_baseline")
        if large is not None:
            ax.axhline(
                large.get("accuracy", 0.0),
                linestyle="--",
                alpha=0.7,
                label=f"{report['config']['large_model']}@N={report['config']['large_n']} ({block.get('dataset')})",
            )

    ax.set_xlabel("N (small model samples)")
    ax.set_ylabel("Accuracy")
    title = "E4: Small + TTC vs Large Greedy"
    if dataset:
        title = f"{title} ({dataset})"
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_e4_efficiency(
    report: dict[str, Any],
    output_path: Path,
    *,
    dataset: str | None = None,
    x_key: str = "mean_tokens_per_example",
    x_label: str = "Mean tokens per example",
) -> Path | None:
    """Scatter: efficiency x-axis vs accuracy for small@N and large baseline."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    blocks = report.get("datasets") or []
    if dataset is not None:
        blocks = [b for b in blocks if b.get("dataset") == dataset]
    if not blocks:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for block in blocks:
        ds = block.get("dataset")
        for comp in block.get("comparisons") or []:
            small = comp.get("small") or {}
            ax.scatter(
                small.get(x_key, 0.0),
                small.get("accuracy", 0.0),
                marker="o",
                label=None,
            )
            ax.annotate(
                f"S@N={comp.get('small_n')}",
                (small.get(x_key, 0.0), small.get("accuracy", 0.0)),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        large = block.get("large_baseline")
        if large is not None:
            ax.scatter(
                large.get(x_key, 0.0),
                large.get("accuracy", 0.0),
                marker="*",
                s=120,
                label=f"Large greedy ({ds})",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    title = "E4 efficiency: accuracy vs compute"
    if dataset:
        title = f"{title} ({dataset})"
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    # De-duplicate legend labels.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_e4_report(report: dict[str, Any], output_dir: Path, *, make_plots: bool = True) -> dict[str, str]:
    """Write JSON, CSV, and optional plots for an E4 report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e4_comparison.json"
    csv_path = output_dir / "e4_paper_table.csv"
    write_json(json_path, report)
    write_e4_paper_csv(csv_path, report)

    paths = {"json": str(json_path), "csv": str(csv_path)}
    if make_plots:
        p1 = plot_e4_small_vs_large(report, output_dir / "e4_accuracy_vs_n.png")
        p2 = plot_e4_efficiency(report, output_dir / "e4_accuracy_vs_tokens.png")
        p3 = plot_e4_efficiency(
            report,
            output_dir / "e4_accuracy_vs_latency.png",
            x_key="mean_latency_s_per_example",
            x_label="Mean latency per example (s)",
        )
        if p1:
            paths["plot_accuracy_vs_n"] = str(p1)
        if p2:
            paths["plot_accuracy_vs_tokens"] = str(p2)
        if p3:
            paths["plot_accuracy_vs_latency"] = str(p3)

        datasets = [b.get("dataset") for b in report.get("datasets") or [] if b.get("dataset")]
        if len(datasets) > 1:
            for ds in datasets:
                plot_e4_small_vs_large(
                    report, output_dir / f"e4_accuracy_vs_n_{ds}.png", dataset=ds
                )
    return paths
