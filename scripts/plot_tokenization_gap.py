#!/usr/bin/env python3
"""Plot the tokenization cost gap across reasoning strategies (H4 tax).

For each (dataset, model) it aggregates the per-example prompt and completion
token counts from run candidates and draws grouped bars per condition
(``method`` / ``prompt_style``), annotated with the total and the percent change
relative to a reference condition (default ``english_cot``).

Two gaps fall out of the same figure:

- **Input-language tax** - ``english_cot`` on English (``afrimgsm_eng``) vs on
  Yoruba (``afrimgsm_yor``): what it costs to encode Yoruba input.
- **Reasoning-language tax** - ``yoruba_cot`` vs ``english_cot`` on the same
  Yoruba items: what it costs to reason natively.

Use E1/E0 runs (greedy N=1) for the clean figure; nested E2 slices would
double-count. No tokenizer is required - counts come from the runs.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.io_utils import read_jsonl, write_csv
from ttcs_yoruba.metrics import dedupe_candidate_rows, find_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prompt/completion token cost per condition (tokenization tax)."
    )
    parser.add_argument("--runs-dir", type=str, default="runs", help="Run directory root.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Single run directory name (default: all runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tokenization",
        help="Directory for the plot and CSV.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument("--model", type=str, default=None, help="Optional model filter.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Condition name for percent-change annotations (default: first english_cot).",
    )
    parser.add_argument(
        "--include-nested",
        action="store_true",
        help="Include nested/E2 candidates (default: only rows with n == 1).",
    )
    return parser.parse_args()


def condition_name(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("prompt_style") or "unknown")


def token_rows(
    candidates: list[dict[str, Any]],
    *,
    include_nested: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate mean prompt/completion tokens per example for each condition."""
    totals: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"examples": 0.0, "prompt": 0.0, "completion": 0.0}
    )
    for row in candidates:
        if not include_nested and int(row.get("n") or 1) != 1:
            continue
        prompt = row.get("prompt_token_count")
        if prompt is None:
            continue
        prompt_style = str(row.get("prompt_style") or "")
        condition = condition_name(row)
        # Default to standalone E1/E0-style methods (method == prompt_style) so
        # nested E2 slices do not double-count.
        if not include_nested and condition != prompt_style:
            continue
        meta = row.get("metadata") or {}
        dataset = str(row.get("dataset") or meta.get("dataset") or "")
        model = str(row.get("model") or "")
        bucket = totals[(dataset, model, condition, prompt_style)]
        bucket["examples"] += 1
        bucket["prompt"] += float(prompt)
        bucket["completion"] += float(row.get("token_count") or 0)

    rows: list[dict[str, Any]] = []
    for (dataset, model, condition, prompt_style), bucket in totals.items():
        n = bucket["examples"]
        if n <= 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "prompt_style": prompt_style,
                "num_examples": int(n),
                "prompt_tokens_per_example": bucket["prompt"] / n,
                "completion_tokens_per_example": bucket["completion"] / n,
                "total_tokens_per_example": (bucket["prompt"] + bucket["completion"]) / n,
            }
        )
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["condition"]))
    return rows


def _pick_reference(rows: list[dict[str, Any]], reference: str | None) -> dict[str, Any] | None:
    if reference:
        for row in rows:
            if row["condition"] == reference:
                return row
    for row in rows:
        if row["prompt_style"] == "english_cot" and row["dataset"].endswith("_eng"):
            return row
    for row in rows:
        if row["prompt_style"] == "english_cot":
            return row
    return None


def _plot_panel(
    ax: Any,
    rows: list[dict[str, Any]],
    reference: dict[str, Any] | None,
    *,
    multi_dataset: bool,
) -> str:
    rows = sorted(rows, key=lambda r: (r["dataset"], r["condition"]))

    def label(row: dict[str, Any]) -> str:
        return f"{row['dataset']}:{row['condition']}" if multi_dataset else row["condition"]

    labels = [label(r) for r in rows]
    prompt = [r["prompt_tokens_per_example"] for r in rows]
    completion = [r["completion_tokens_per_example"] for r in rows]
    x = list(range(len(rows)))
    width = 0.4

    ax.bar([i - width / 2 for i in x], prompt, width, label="prompt tokens/ex", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], completion, width, label="completion tokens/ex", color="#dd8452")

    ref_total = reference["total_tokens_per_example"] if reference else 0.0
    for i, row in enumerate(rows):
        total = row["total_tokens_per_example"]
        note = f"{total:,.0f}"
        if reference is not None and ref_total > 0 and label(reference) != label(row):
            note += f"\n{total / ref_total - 1:+.0%}"
        ax.annotate(note, (i, max(prompt[i], 0) + completion[i]), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("tokens / example")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return label(reference) if reference is not None else "n/a"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path(args.runs_dir)
    run_dirs = [runs_dir / args.run_id] if args.run_id else find_run_dirs(runs_dir, require_manifest=False)
    run_dirs = [path for path in run_dirs if path.exists()]
    candidates: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / "candidates.jsonl"
        if path.exists():
            candidates.extend(read_jsonl(path))
    candidates = dedupe_candidate_rows(candidates)
    rows = token_rows(candidates, include_nested=args.include_nested)

    if args.dataset:
        rows = [r for r in rows if r["dataset"] == args.dataset]
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if not rows:
        print("No candidate token stats found.", file=sys.stderr)
        sys.exit(1)

    csv_path = output_dir / "tokenization_gap.csv"
    write_csv(
        csv_path,
        rows,
        [
            "dataset",
            "model",
            "condition",
            "prompt_style",
            "num_examples",
            "prompt_tokens_per_example",
            "completion_tokens_per_example",
            "total_tokens_per_example",
        ],
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; wrote {csv_path} only.", file=sys.stderr)
        return

    by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_panel[row["model"]].append(row)

    panels = sorted(by_panel)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4.2 * len(panels)), squeeze=False)
    for ax, model in zip(axes[:, 0], panels):
        panel_rows = by_panel[model]
        multi_dataset = len({r["dataset"] for r in panel_rows}) > 1
        reference = _pick_reference(panel_rows, args.reference)
        ref_label = _plot_panel(ax, panel_rows, reference, multi_dataset=multi_dataset)
        ax.set_title(f"{model}  (token cost; % vs {ref_label})")
    fig.tight_layout()
    out_path = output_dir / "tokenization_gap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_path}")
    for row in rows:
        print(
            f"  {row['dataset']:12s} {row['model']:12s} {row['condition']:16s} "
            f"prompt={row['prompt_tokens_per_example']:7.1f} "
            f"completion={row['completion_tokens_per_example']:7.1f} "
            f"total={row['total_tokens_per_example']:8.1f}"
        )


if __name__ == "__main__":
    main()
