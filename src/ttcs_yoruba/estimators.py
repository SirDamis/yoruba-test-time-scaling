"""Pool-based estimators for test-time compute.

The nested E2 design generates one stochastic pool per example and evaluates
prefix slices. These helpers do the statistically stronger version:

- ``pass_at_k_unbiased``: full-pool pass@k estimator (Chen et al. 2021).
- ``majority_accuracy_at_k``: majority vote averaged over random k-subsets,
  capturing subset sensitivity that a single fixed prefix hides.
- ``pool_diversity``: distinct-answer diagnostic to flag degenerate pools.

None of this requires deterministic generation; it operates on whatever pool
was drawn.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .extraction import is_exact_match, normalize_for_match, numeric_match_key
from .selection import select_candidate


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Probability a random ``k``-subset of ``n`` samples contains >=1 correct.

    ``c`` is the number of correct samples in the pool. Uses the unbiased
    estimator ``1 - C(n-c, k) / C(n, k)`` (Chen et al., 2021).
    """
    if k <= 0 or n <= 0 or c <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def answer_key(answer: str, answer_type: str | None) -> str:
    """Match key used for vote/diversity (numeric-aware for number answers)."""
    if not answer:
        return ""
    if answer_type == "number":
        return numeric_match_key(answer)
    return normalize_for_match(answer)


def _meta(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = row.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def candidate_dataset(row: Mapping[str, Any]) -> str:
    return str(row.get("dataset") or _meta(row).get("dataset") or "")


def _resolve_type_gold(
    pool: Sequence[Mapping[str, Any]], answer_type: str | None
) -> tuple[str, str]:
    if answer_type:
        resolved_type = str(answer_type)
    else:
        resolved_type = str(_meta(pool[0]).get("answer_type") or "")
    gold = str(_meta(pool[0]).get("gold_answer") or "")
    return resolved_type, gold


def pool_diversity(
    pool: Sequence[Mapping[str, Any]], *, answer_type: str | None = None
) -> dict[str, Any]:
    """Distinct non-empty answers and empties in one example's pool."""
    keys = [answer_key(str(r.get("extracted_answer", "")), answer_type) for r in pool]
    non_empty = {key for key in keys if key}
    return {
        "pool_size": len(pool),
        "distinct_answers": len(non_empty),
        "empty_extractions": sum(1 for key in keys if not key),
        "degenerate": len(non_empty) <= 1,
    }


def _count_correct(pool: Sequence[Mapping[str, Any]], answer_type: str, gold: str) -> int:
    return sum(
        1
        for r in pool
        if is_exact_match(str(r.get("extracted_answer", "")), gold, answer_type)
    )


def majority_accuracy_at_k(
    pool: Sequence[Mapping[str, Any]],
    k: int,
    *,
    answer_type: str | None = None,
    num_subsets: int = 20,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Mean and within-pool std of majority-vote accuracy over random k-subsets."""
    if not pool or num_subsets <= 0:
        return 0.0, 0.0
    resolved_type, gold = _resolve_type_gold(pool, answer_type)
    if not gold:
        return 0.0, 0.0
    n = len(pool)
    k = min(max(1, int(k)), n)
    rng = rng or random.Random(0)
    scores: list[float] = []
    for _ in range(num_subsets):
        subset = rng.sample(list(pool), k)
        result = select_candidate(subset, "majority_vote", answer_type=resolved_type)
        scores.append(1.0 if is_exact_match(result.selected_answer, gold, resolved_type) else 0.0)
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return mean, math.sqrt(variance)


def _group_pool_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str], list[Mapping[str, Any]]]:
    """Collect the full (max-N) stochastic pool per (dataset, model, group, example)."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        meta = _meta(row)
        if not meta.get("nested") or meta.get("nested_greedy_n1"):
            continue
        pool_n = meta.get("nested_pool_n")
        if pool_n is None:
            continue
        if int(row.get("n") or 1) != int(pool_n):
            continue
        key = (
            candidate_dataset(row),
            str(row.get("model", "")),
            str(meta.get("nested_group_id") or row.get("method", "")),
            str(row.get("example_id", "")),
        )
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("sample_index") or 0))
    return groups


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def default_ks(pool_n: int) -> list[int]:
    return [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= pool_n]


def nested_pool_estimates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] | None = None,
    num_subsets: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per (dataset, model, group, k): unbiased pass@k, random-subset maj@k, diversity.

    Uses each example's full max-N pool, so the estimate at any k draws on all
    available samples rather than a fixed prefix.
    """
    pools = _group_pool_candidates(candidates)
    if not pools:
        return []

    by_group: dict[tuple[str, str, str], list[tuple[str, list[Mapping[str, Any]]]]] = defaultdict(list)
    for (dataset, model, group, example_id), pool in pools.items():
        by_group[(dataset, model, group)].append((example_id, pool))

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for (dataset, model, group), items in sorted(by_group.items()):
        pool_n = max(len(pool) for _, pool in items)
        ks_for_group = list(ks) if ks is not None else default_ks(pool_n)
        ks_for_group = sorted({int(k) for k in ks_for_group if 1 <= int(k) <= pool_n})

        # Per-sample generation cost, averaged across examples (independent of k).
        pooled = [(example_id, pool) for example_id, pool in items if pool]
        mean_completion_per_sample = _mean(
            [_mean([_as_int(r.get("token_count")) for r in pool]) for _, pool in pooled]
        )
        mean_prompt_per_sample = _mean(
            [_mean([_as_int(r.get("prompt_token_count")) for r in pool]) for _, pool in pooled]
        )

        for k in ks_for_group:
            pass_vals: list[float] = []
            maj_means: list[float] = []
            subset_stds: list[float] = []
            distinct_vals: list[float] = []
            degenerate: list[float] = []
            for _example_id, pool in items:
                if not pool:
                    continue
                resolved_type, gold = _resolve_type_gold(pool, None)
                correct = _count_correct(pool, resolved_type, gold)
                pass_vals.append(pass_at_k_unbiased(len(pool), correct, k))
                subset_mean, subset_std = majority_accuracy_at_k(
                    pool, k, answer_type=resolved_type, num_subsets=num_subsets, rng=rng
                )
                maj_means.append(subset_mean)
                subset_stds.append(subset_std)
                diversity = pool_diversity(pool, answer_type=resolved_type)
                distinct_vals.append(float(diversity["distinct_answers"]))
                degenerate.append(1.0 if diversity["degenerate"] else 0.0)
            if not pass_vals:
                continue
            maj_mean = _mean(maj_means)
            completion_at_k = mean_completion_per_sample * k
            prompt_at_k = mean_prompt_per_sample * k
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "nested_group_id": group,
                    "pool_n": pool_n,
                    "k": k,
                    "num_examples": len(pass_vals),
                    "pass_at_k": _mean(pass_vals),
                    "maj_at_k_mean": maj_mean,
                    "maj_at_k_between_example_std": _std(maj_means),
                    "maj_at_k_subset_std_mean": _mean(subset_stds),
                    "distinct_answers_mean": _mean(distinct_vals),
                    "degenerate_pool_rate": _mean(degenerate),
                    "mean_completion_tokens_per_sample": mean_completion_per_sample,
                    "mean_prompt_tokens_per_sample": mean_prompt_per_sample,
                    "completion_tokens_per_example_at_k": completion_at_k,
                    "prompt_tokens_per_example_at_k": prompt_at_k,
                    "total_tokens_per_example_at_k": completion_at_k + prompt_at_k,
                    "completion_tokens_per_correct": (
                        completion_at_k / maj_mean if maj_mean > 0 else None
                    ),
                }
            )
    return rows
