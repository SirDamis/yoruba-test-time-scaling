"""Tests for example-aware sampling seeds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import (
    DatasetConfig,
    InferenceMethodConfig,
    InferenceModelConfig,
    InferenceRunConfig,
)
from ttcs_yoruba.examples import InferenceExample
from ttcs_yoruba.inference import derive_sample_seed, run_example_candidates
from ttcs_yoruba.schema import BackendOutput


def test_none_base_seed_returns_none() -> None:
    assert derive_sample_seed(None, "ex1", 0) is None


def test_same_example_sample_is_stable() -> None:
    a = derive_sample_seed(20260708, "afrimgsm_all_000001", 0)
    b = derive_sample_seed(20260708, "afrimgsm_all_000001", 0)
    assert a == b
    assert isinstance(a, int)
    assert 0 <= a < 2**31


def test_different_examples_differ_at_same_sample_index() -> None:
    s1 = derive_sample_seed(20260708, "example_a", 0)
    s2 = derive_sample_seed(20260708, "example_b", 0)
    assert s1 != s2


def test_different_sample_indices_differ() -> None:
    s0 = derive_sample_seed(20260708, "example_a", 0)
    s1 = derive_sample_seed(20260708, "example_a", 1)
    assert s0 != s1


def test_not_equal_to_naive_base_plus_index() -> None:
    """Regression: old scheme was base_seed + sample_index for every example."""
    base = 20260708
    for example_id in ("ex1", "ex2", "ex3"):
        for sample_index in range(4):
            derived = derive_sample_seed(base, example_id, sample_index)
            assert derived != base + sample_index or example_id == "ex1"
    # Stronger: sample 0 must not be identical across examples (old bug).
    seeds = {derive_sample_seed(base, f"ex{i}", 0) for i in range(20)}
    assert len(seeds) == 20


class _SeedRecordingBackend:
    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def generate(self, **kwargs):
        self.seeds.append(kwargs.get("seed"))
        return BackendOutput(response="Final answer: 11", token_count=3, latency_s=0.01)


def _candidate_test_inputs(seed: int | None):
    config = InferenceRunConfig(
        run_id="seed_test",
        output_dir=ROOT / "runs",
        datasets=[],
        models=[],
        methods=[],
        seed=seed,
    )
    dataset = DatasetConfig(name="toy", path=ROOT / "toy.jsonl", task="math", source_dataset="toy")
    model = InferenceModelConfig(
        name="fake",
        backend="openai_compatible",
        model="fake",
        size_label="0B",
    )
    method = InferenceMethodConfig(
        name="ttc_n2",
        prompt_style="english_cot",
        selection="majority_vote",
        n=2,
        temperature=0.7,
        max_tokens=64,
    )
    example = InferenceExample(
        id="toy_000001",
        task="math",
        question="Melo ni 5 ati 6?",
        choices=None,
        gold_answer="11",
        answer_type="number",
        source_dataset="toy",
    )
    return config, dataset, model, method, example


def test_candidate_metadata_records_sample_seed() -> None:
    config, dataset, model, method, example = _candidate_test_inputs(seed=20260708)
    backend = _SeedRecordingBackend()

    rows = run_example_candidates(
        config=config,
        dataset=dataset,
        model=model,
        method=method,
        example=example,
        backend=backend,
    )

    expected = [
        derive_sample_seed(config.seed, example.id, sample_index)
        for sample_index in range(method.n)
    ]
    assert backend.seeds == expected
    assert [row["metadata"]["sample_seed"] for row in rows] == expected


def test_candidate_metadata_records_none_sample_seed() -> None:
    config, dataset, model, method, example = _candidate_test_inputs(seed=None)
    backend = _SeedRecordingBackend()

    rows = run_example_candidates(
        config=config,
        dataset=dataset,
        model=model,
        method=method,
        example=example,
        backend=backend,
    )

    assert backend.seeds == [None, None]
    assert [row["metadata"]["sample_seed"] for row in rows] == [None, None]


if __name__ == "__main__":
    test_none_base_seed_returns_none()
    test_same_example_sample_is_stable()
    test_different_examples_differ_at_same_sample_index()
    test_different_sample_indices_differ()
    test_not_equal_to_naive_base_plus_index()
    test_candidate_metadata_records_sample_seed()
    test_candidate_metadata_records_none_sample_seed()
    print("ok")
