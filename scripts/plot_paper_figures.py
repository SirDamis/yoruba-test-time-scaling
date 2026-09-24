#!/usr/bin/env python3
"""Paper figures that are not covered by the E2/E4 aggregators.

Subcommands:

- ``representation``  E0 English vs E1 Yoruba accuracy per model (H1 / representation gap)
- ``reasoning``       E1 method accuracy with Wilson CIs + McNemar brackets
- ``crosslingual``    language x method accuracy heatmap
- ``selector``        E3 selector comparison (first / majority / verifier) vs k
- ``compute-matched`` E1 method accuracy vs token cost per model
- ``diversity``       pool distinct-answer / degeneracy curves vs k

All read ``runs/*/selections.jsonl`` (and ``candidates.jsonl`` where needed), so
they work on whatever has already been run.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.extraction import is_exact_match
from ttcs_yoruba.io_utils import read_jsonl, write_csv
from ttcs_yoruba.metrics import (
    dedupe_candidate_rows,
    dedupe_selection_rows,
    find_run_dirs,
)
from ttcs_yoruba.selection import select_candidate

E1_METHODS = ["direct", "yoruba_cot", "english_cot", "translate_pivot"]
REASONING_ORDER = {"direct": 0, "yoruba_cot": 1, "english_cot": 2, "translate_pivot": 3}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def suffix(dataset: str) -> str:
    return dataset.rpartition("_")[2]


def family(dataset: str) -> str:
    return dataset.rpartition("_")[0]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def mcnemar(A: dict[str, bool], B: dict[str, bool]) -> tuple[int, int, float]:
    common = set(A) & set(B)
    b = sum(1 for k in common if A[k] and not B[k])
    c = sum(1 for k in common if B[k] and not A[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5**n
    return b, c, min(1.0, p)


def all_run_dirs(runs_dir: str, run_id: str | None) -> list[Path]:
    root = Path(runs_dir)
    dirs = [root / run_id] if run_id else find_run_dirs(root, require_manifest=False)
    return [d for d in dirs if d.exists()]


def load_selections(runs_dir: str, run_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in all_run_dirs(runs_dir, run_id):
        path = run_dir / "selections.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    return dedupe_selection_rows(rows)


def load_candidates(runs_dir: str, run_id: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in all_run_dirs(runs_dir, run_id):
        path = run_dir / "candidates.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    return dedupe_candidate_rows(rows)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(path, rows, fields)


def _filter(rows: list[dict[str, Any]], model: str | None, dataset: str | None):
    if model:
        rows = [r for r in rows if str(r.get("model")) == model]
    if dataset:
        rows = [r for r in rows if str(r.get("dataset")) == dataset]
    return rows


# --------------------------------------------------------------------------- #
# 1. representation gap
# --------------------------------------------------------------------------- #
def fig_representation(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _filter(load_selections(args.runs_dir, args.run_id), args.model, None)
    # model -> family -> language -> list of (example_id, correct, gold)
    data: dict[str, dict[str, dict[str, list[tuple[str, bool, str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if str(row.get("prompt_style")) != "english_cot":
            continue
        model = str(row.get("model"))
        ds = str(row.get("dataset"))
        data[model][family(ds)][suffix(ds)].append(
            (str(row["example_id"]), bool(row.get("is_correct")), str(row.get("gold_answer", "")))
        )

    out: list[dict[str, Any]] = []
    for model, fams in sorted(data.items()):
        for fam, langs in fams.items():
            if "eng" not in langs or "yor" not in langs:
                continue
            eng = sorted(langs["eng"], key=lambda t: t[0])
            yor = sorted(langs["yor"], key=lambda t: t[0])
            n = min(len(eng), len(yor))
            gold_mismatch = sum(1 for i in range(n) if eng[i][2] != yor[i][2])
            A = {str(i): eng[i][1] for i in range(n)}
            B = {str(i): yor[i][1] for i in range(n)}
            eng_k = sum(A.values())
            yor_k = sum(B.values())
            b, c, p = mcnemar(A, B)
            out.append(
                {
                    "model": model,
                    "dataset_family": fam,
                    "n_paired": n,
                    "gold_mismatch": gold_mismatch,
                    "english_input_acc": eng_k / n,
                    "yoruba_input_acc": yor_k / n,
                    "drop_pp": (eng_k - yor_k) / n * 100,
                    "b_eng_only": b,
                    "c_yor_only": c,
                    "mcnemar_p": p,
                }
            )

    plt = _require_matplotlib()
    if plt is None or not out:
        return out
    out_sorted = sorted(out, key=lambda r: r["dataset_family"])
    labels = [f"{r['dataset_family']}\n{r['model']}" for r in out_sorted]
    x = list(range(len(out_sorted)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 5))
    eng = [r["english_input_acc"] for r in out_sorted]
    yor = [r["yoruba_input_acc"] for r in out_sorted]
    engci = [wilson(int(r["english_input_acc"] * r["n_paired"]), r["n_paired"]) for r in out_sorted]
    yorci = [wilson(int(r["yoruba_input_acc"] * r["n_paired"]), r["n_paired"]) for r in out_sorted]
    eng_err = [[a - ci[0] for a, ci in zip(eng, engci)], [ci[1] - a for a, ci in zip(eng, engci)]]
    yor_err = [[a - ci[0] for a, ci in zip(yor, yorci)], [ci[1] - a for a, ci in zip(yor, yorci)]]
    ax.bar([i - width / 2 for i in x], eng, width, yerr=eng_err, capsize=3, label="English input", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], yor, width, yerr=yor_err, capsize=3, label="Yoruba input", color="#c44e52")
    for i, r in enumerate(out_sorted):
        ax.annotate(f"−{r['drop_pp']:.0f}pp\np={r['mcnemar_p']:.1e}", (i, max(r["english_input_acc"], 0.02) + 0.03), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (english_cot)")
    ax.set_title("Representation gap: same items, English vs Yoruba input")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(args.output_dir) / "representation_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# 2. reasoning-language comparison
# --------------------------------------------------------------------------- #
def fig_reasoning(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = args.dataset or "afrimgsm_yor"
    rows = _filter(load_selections(args.runs_dir, args.run_id), args.model, dataset)
    by_model_method: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        method = str(row.get("method") or row.get("prompt_style"))
        if method not in E1_METHODS:
            continue
        by_model_method[(str(row.get("model")), method)][str(row["example_id"])] = bool(row.get("is_correct"))

    out: list[dict[str, Any]] = []
    models = sorted({m for m, _ in by_model_method})
    for model in models:
        for method in E1_METHODS:
            correct = by_model_method.get((model, method))
            if not correct:
                continue
            n = len(correct)
            k = sum(correct.values())
            lo, hi = wilson(k, n)
            out.append({"model": model, "method": method, "n": n, "acc": k / n, "ci_lo": lo, "ci_hi": hi})

    # McNemar pairs within each model
    pairs: list[dict[str, Any]] = []
    for model in models:
        for i, a in enumerate(E1_METHODS):
            for b in E1_METHODS[i + 1 :]:
                A = by_model_method.get((model, a))
                B = by_model_method.get((model, b))
                if not A or not B:
                    continue
                bb, cc, p = mcnemar(A, B)
                pairs.append({"model": model, "a": a, "b": b, "b_only_a": bb, "c_only_b": cc, "p": p})

    plt = _require_matplotlib()
    if plt is None or not out:
        return out
    fig, axes = plt.subplots(len(models), 1, figsize=(7, 4 * len(models)), squeeze=False)
    for ax, model in zip(axes[:, 0], models):
        mrows = [r for r in out if r["model"] == model]
        mrows.sort(key=lambda r: REASONING_ORDER.get(r["method"], 99))
        x = list(range(len(mrows)))
        ax.bar(x, [r["acc"] for r in mrows], yerr=[[r["acc"] - r["ci_lo"] for r in mrows], [r["ci_hi"] - r["acc"] for r in mrows]], capsize=4, color="#55a868")
        for i, r in enumerate(mrows):
            ax.annotate(f"{r['acc']:.1%}\nn={r['n']}", (i, r["ci_hi"] + 0.02), ha="center", fontsize=8)
        # significance brackets over adjacent significant pairs
        for pr in pairs:
            if pr["model"] != model or pr["p"] >= 0.05:
                continue
            ia = [i for i, r in enumerate(mrows) if r["method"] == pr["a"]]
            ib = [i for i, r in enumerate(mrows) if r["method"] == pr["b"]]
            if ia and ib:
                y = max(mrows[ia[0]]["ci_hi"], mrows[ib[0]]["ci_hi"]) + 0.06
                ax.plot([ia[0], ib[0]], [y, y], color="black", lw=1)
                ax.annotate(f"p={pr['p']:.3f}", ((ia[0] + ib[0]) / 2, y + 0.01), ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([r["method"] for r in mrows], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{dataset} · {model} (greedy N=1)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(args.output_dir) / "reasoning_language.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# 3. cross-lingual heatmap
# --------------------------------------------------------------------------- #
def fig_crosslingual(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _filter(load_selections(args.runs_dir, args.run_id), args.model, args.dataset)
    acc: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        ds = str(row.get("dataset"))
        model = str(row.get("model"))
        method = str(row.get("method") or row.get("prompt_style"))
        if method not in E1_METHODS:
            continue
        acc[(f"{family(ds)}|{model}", suffix(ds), method)][str(row["example_id"])] = bool(row.get("is_correct"))

    out: list[dict[str, Any]] = []
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for (grp, lang, method), correct in acc.items():
        groups[grp].append((lang, method, correct))
        out.append({"group": grp, "language": lang, "method": method, "n": len(correct), "acc": sum(correct.values()) / len(correct)})

    plt = _require_matplotlib()
    if plt is None or not groups:
        return out
    for grp, items in sorted(groups.items()):
        fam, model = grp.split("|", 1)
        langs = sorted({lang for lang, _, _ in items})
        methods = [m for m in E1_METHODS if any(m == mm for _, mm, _ in items)]
        methods += sorted({mm for _, mm, _ in items if mm not in methods})
        grid = [[float("nan")] * len(methods) for _ in langs]
        for i, lang in enumerate(langs):
            for j, method in enumerate(methods):
                for lang2, method2, correct in items:
                    if lang2 == lang and method2 == method:
                        grid[i][j] = sum(correct.values()) / len(correct)
        fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 3, 0.6 * len(langs) + 2))
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels(langs)
        for i in range(len(langs)):
            for j in range(len(methods)):
                if not math.isnan(grid[i][j]):
                    ax.text(j, i, f"{grid[i][j]:.0%}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Accuracy")
        ax.set_title(f"{fam} · {model}")
        fig.tight_layout()
        path = Path(args.output_dir) / f"crosslingual_{_safe(fam)}_{_safe(model)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# 4. selector comparison (E3)
# --------------------------------------------------------------------------- #
def _pool_by_example(candidates: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        meta = row.get("metadata") or {}
        if not meta.get("nested") or meta.get("nested_greedy_n1"):
            continue
        pool_n = meta.get("nested_pool_n")
        if pool_n is None or int(row.get("n") or 1) != int(pool_n):
            continue
        ds = str(row.get("dataset") or meta.get("dataset") or "")
        key = (ds, str(row.get("model")), str(row.get("example_id")))
        pools[key].append(row)
    for key in pools:
        pools[key].sort(key=lambda r: int(r.get("sample_index") or 0))
    return pools


def _is_correct(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return is_exact_match(
        str(row.get("extracted_answer", "")),
        str(meta.get("gold_answer", "")),
        str(meta.get("answer_type", "")),
    )


def fig_selector(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = _filter(load_candidates(args.runs_dir, args.run_id), args.model, args.dataset)
    pools = _pool_by_example(candidates)
    if not pools:
        return []
    by_group: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for (ds, model, _eid), pool in pools.items():
        by_group[(ds, model)].append(pool)

    out: list[dict[str, Any]] = []
    for (ds, model), examples in sorted(by_group.items()):
        pool_n = max(len(p) for p in examples)
        ks = [k for k in (1, 2, 3, 4, 8, 16, 32, 64) if k <= pool_n]
        for k in ks:
            first_ok = maj_ok = 0
            for pool in examples:
                subset = pool[:k]
                first = subset[0]
                at = str((first.get("metadata") or {}).get("answer_type", ""))
                if _is_correct(first):
                    first_ok += 1
                res = select_candidate(subset, "majority_vote", answer_type=at)
                if _is_correct(
                    {
                        "extracted_answer": res.selected_answer,
                        "metadata": {"gold_answer": (first.get("metadata") or {}).get("gold_answer"), "answer_type": at},
                    }
                ):
                    maj_ok += 1
            n = len(examples)
            out.append({"dataset": ds, "model": model, "k": k, "n": n, "first": first_ok / n, "majority_vote": maj_ok / n})

    plt = _require_matplotlib()
    if plt is None:
        return out
    for (ds, model) in sorted(by_group):
        rows = [r for r in out if r["dataset"] == ds and r["model"] == model]
        rows.sort(key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ks, [r["first"] for r in rows], marker="o", label="first")
        ax.plot(ks, [r["majority_vote"] for r in rows], marker="s", label="majority_vote")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("N (samples)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"E3 selector comparison · {ds} · {model}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = Path(args.output_dir) / f"selector_comparison_{_safe(ds)}_{_safe(model)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# 6. compute-matched strategy accuracy
# --------------------------------------------------------------------------- #
def fig_compute_matched(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = _filter(load_candidates(args.runs_dir, args.run_id), args.model, args.dataset)
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0.0, "tok": 0.0, "ok": 0.0})
    for row in candidates:
        if int(row.get("n") or 1) != 1:
            continue
        method = str(row.get("method") or row.get("prompt_style"))
        if method not in E1_METHODS:
            continue
        meta = row.get("metadata") or {}
        ds = str(row.get("dataset") or meta.get("dataset") or "")
        key = (ds, method)
        bucket = stats[key]
        bucket["n"] += 1
        bucket["tok"] += float(row.get("prompt_token_count") or 0) + float(row.get("token_count") or 0)
        bucket["ok"] += 1 if _is_correct(row) else 0
    out = [
        {
            "dataset": ds,
            "method": method,
            "n": int(b["n"]),
            "tokens_per_example": b["tok"] / b["n"] if b["n"] else 0.0,
            "accuracy": b["ok"] / b["n"] if b["n"] else 0.0,
        }
        for (ds, method), b in stats.items()
    ]

    plt = _require_matplotlib()
    if plt is None or not out:
        return out
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"direct": "#999999", "yoruba_cot": "#c44e52", "english_cot": "#4c72b0", "translate_pivot": "#55a868"}
    for row in out:
        ax.scatter(row["tokens_per_example"], row["accuracy"], color=colors.get(row["method"], "black"), s=60)
        ax.annotate(f"{row['method']} ({row['dataset']})", (row["tokens_per_example"], row["accuracy"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("tokens / example (prompt + completion)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("E1 strategy accuracy vs token cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(args.output_dir) / "compute_matched_strategies.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return out


# --------------------------------------------------------------------------- #
# 7. pool diversity vs k
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_diversity(
    pools: list[list[dict[str, Any]]],
    k: int,
    *,
    num_subsets: int,
    seed: int,
) -> tuple[float, float]:
    """Mean distinct answers and degenerate rate over random k-subsets of each pool."""
    import random

    rng = random.Random(seed)
    distinct_vals: list[float] = []
    degenerate_vals: list[float] = []
    for pool in pools:
        kk = min(k, len(pool))
        for _ in range(num_subsets):
            subset = rng.sample(pool, kk)
            keys = {str(r.get("extracted_answer", "")).strip() for r in subset}
            keys.discard("")
            distinct_vals.append(float(len(keys)))
            degenerate_vals.append(1.0 if len(keys) <= 1 else 0.0)
    return _mean(distinct_vals), _mean(degenerate_vals)


def fig_diversity(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pool_estimates:
        from ttcs_yoruba.io_utils import read_json

        payload = read_json(Path(args.pool_estimates))
        rows = list(payload.get("estimates", payload))
        if args.dataset:
            rows = [r for r in rows if r["dataset"] == args.dataset]
    else:
        candidates = _filter(load_candidates(args.runs_dir, args.run_id), args.model, args.dataset)
        pools = _pool_by_example(candidates)
        grouped: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = defaultdict(list)
        for (ds, model, _eid), pool in pools.items():
            gid = str((pool[0].get("metadata") or {}).get("nested_group_id") or "group")
            grouped[(ds, model, gid)].append(pool)
        rows = []
        for (ds, model, gid), group_pools in sorted(grouped.items()):
            pool_n = max(len(p) for p in group_pools)
            for k in [k for k in (1, 2, 3, 4, 8, 16, 32, 64) if k <= pool_n]:
                distinct, degenerate = _sample_diversity(
                    group_pools, k, num_subsets=args.num_subsets, seed=args.seed
                )
                rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "nested_group_id": gid,
                        "pool_n": pool_n,
                        "k": k,
                        "num_examples": len(group_pools),
                        "distinct_answers_mean": distinct,
                        "degenerate_pool_rate": degenerate,
                    }
                )

    plt = _require_matplotlib()
    if plt is None or not rows:
        return rows
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["dataset"], row["model"], row["nested_group_id"])].append(row)
    fig, axes = plt.subplots(len(by_group), 1, figsize=(7, 3.5 * len(by_group)), squeeze=False)
    for ax, key in zip(axes[:, 0], sorted(by_group)):
        pts = sorted(by_group[key], key=lambda r: r["k"])
        ks = [p["k"] for p in pts]
        ax.plot(ks, [p["distinct_answers_mean"] for p in pts], marker="o", color="#4c72b0", label="distinct answers (mean)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k (samples)")
        ax.set_ylabel("distinct answers", color="#4c72b0")
        ax.tick_params(axis="y", labelcolor="#4c72b0")
        ax2 = ax.twinx()
        ax2.plot(ks, [p["degenerate_pool_rate"] for p in pts], marker="x", linestyle="--", color="#c44e52", label="degenerate pool rate")
        ax2.set_ylabel("degenerate pool rate", color="#c44e52")
        ax2.set_ylim(0, 1.02)
        ax2.tick_params(axis="y", labelcolor="#c44e52")
        ax.set_title(f"{key[0]} · {key[1]} · {key[2]}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(args.output_dir) / "pool_diversity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
COMMANDS: dict[str, Callable[[argparse.Namespace], list[dict[str, Any]]]] = {
    "representation": fig_representation,
    "reasoning": fig_reasoning,
    "crosslingual": fig_crosslingual,
    "selector": fig_selector,
    "compute-matched": fig_compute_matched,
    "diversity": fig_diversity,
}

CSV_FIELDS = {
    "representation": ["model", "dataset_family", "n_paired", "gold_mismatch", "english_input_acc", "yoruba_input_acc", "drop_pp", "b_eng_only", "c_yor_only", "mcnemar_p"],
    "reasoning": ["model", "method", "n", "acc", "ci_lo", "ci_hi"],
    "crosslingual": ["group", "language", "method", "n", "acc"],
    "selector": ["dataset", "model", "k", "n", "first", "majority_vote"],
    "compute-matched": ["dataset", "method", "n", "tokens_per_example", "accuracy"],
    "diversity": ["dataset", "model", "nested_group_id", "pool_n", "k", "num_examples", "distinct_answers_mean", "degenerate_pool_rate"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        p = sub.add_parser(name, help=COMMANDS[name].__doc__ or name)
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--run-id", default=None)
        p.add_argument("--output-dir", default="results/figures")
        p.add_argument("--model", default=None)
        p.add_argument("--dataset", default=None)
        if name == "diversity":
            p.add_argument("--pool-estimates", default=None)
            p.add_argument("--num-subsets", type=int, default=100)
            p.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = COMMANDS[args.command](args)
    if rows:
        csv_path = output_dir / f"paper_{args.command.replace('-', '_')}.csv"
        _write_csv(csv_path, rows, CSV_FIELDS[args.command])
        print(f"Wrote {csv_path}")
    else:
        print("No data for this figure.", file=sys.stderr)


if __name__ == "__main__":
    main()
