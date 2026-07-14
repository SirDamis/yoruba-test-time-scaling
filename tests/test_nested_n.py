"""Tests for nested N sampling (sample once at max N, evaluate prefixes)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import expand_method_configs, load_inference_run_config
from ttcs_yoruba.inference import (
    is_nested_greedy_n1,
    materialize_nested_slice,
    pick_nested_sample_method,
    run_inference_pipeline,
    split_nested_group_methods,
)
from ttcs_yoruba.config import (
    DatasetConfig,
    InferenceMethodConfig,
    InferenceModelConfig,
    InferenceRunConfig,
)


def test_expand_sets_nested_group() -> None:
    methods = expand_method_configs(
        [
            {
                "name": "ttc",
                "prompt_style": "english_cot",
                "selection": "majority_vote",
                "n_values": [1, 4, 8],
                "nested_n": True,
                "temperature": 0.7,
                "greedy_n1": True,
            }
        ]
    )
    assert len(methods) == 3
    assert all(m.nested_group_id == "ttc" for m in methods)
    assert all(m.nested_max_n == 8 for m in methods)
    assert methods[0].n == 1 and methods[0].selection == "first"
    assert methods[0].temperature == 0.0
    assert is_nested_greedy_n1(methods[0])
    assert methods[1].n == 4 and methods[1].selection == "majority_vote"
    assert not is_nested_greedy_n1(methods[1])
    greedy, pool = split_nested_group_methods(methods)
    assert [m.n for m in greedy] == [1]
    assert [m.n for m in pool] == [4, 8]


def test_expand_without_nested_has_no_group() -> None:
    methods = expand_method_configs(
        [
            {
                "name": "ttc",
                "prompt_style": "english_cot",
                "selection": "majority_vote",
                "n_values": [1, 4],
                "temperature": 0.7,
            }
        ]
    )
    assert all(m.nested_group_id is None for m in methods)


def test_e2_config_is_nested() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e2_ttc_scaling.json")
    assert all(m.nested_group_id == "english_cot_ttc" for m in cfg.methods)
    assert max(m.n for m in cfg.methods) == 64


def test_materialize_slice_tags_method_and_n() -> None:
    pool = [
        {
            "sample_index": i,
            "method": "pool",
            "n": 8,
            "selection": "majority_vote",
            "extracted_answer": str(i),
            "metadata": {"dataset": "afrimgsm"},
        }
        for i in range(8)
    ]
    method = InferenceMethodConfig(
        name="ttc_n4",
        prompt_style="english_cot",
        selection="majority_vote",
        n=4,
        nested_group_id="ttc",
        nested_max_n=8,
    )
    sliced = materialize_nested_slice(pool, method=method, nested_group_id="ttc", pool_n=8)
    assert len(sliced) == 4
    assert [r["sample_index"] for r in sliced] == [0, 1, 2, 3]
    assert all(r["method"] == "ttc_n4" and r["n"] == 4 for r in sliced)
    assert all(r["metadata"]["nested"] is True for r in sliced)
    assert sliced[0]["extracted_answer"] == "0"


def test_pick_sample_method_uses_pool_n() -> None:
    group = expand_method_configs(
        [
            {
                "name": "ttc",
                "prompt_style": "english_cot",
                "selection": "majority_vote",
                "n_values": [1, 4, 8],
                "nested_n": True,
                "temperature": 0.7,
                "greedy_n1": True,
            }
        ]
    )
    _, pool_methods = split_nested_group_methods(group)
    sample_m = pick_nested_sample_method(pool_methods, pool_n=8)
    assert sample_m.n == 8
    assert sample_m.temperature == 0.7


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.temperatures: list[float | None] = []

    def generate(self, **kwargs):
        from ttcs_yoruba.schema import BackendOutput

        self.calls += 1
        self.temperatures.append(kwargs.get("temperature"))
        # Alternate answers so majority/pass@k vary with k.
        ans = "11" if self.calls % 2 == 1 else "0"
        return BackendOutput(
            response=f"Final answer: {ans}",
            token_count=5,
            latency_s=0.01,
        )


def test_nested_pipeline_greedy_n1_plus_pool(monkeypatch=None) -> None:
    """Greedy N=1 is a separate decode; pool of 8 is shared by N=4 and N=8.

    Generations per example: 1 (greedy) + 8 (pool) = 9, not 1+4+8=13 and not pool-only 8.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_path = root / "data.jsonl"
        data_path.write_text(
            json.dumps(
                {
                    "question": "Melo?",
                    "gold_answer": "11",
                    "answer_type": "number",
                    "choices": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        methods = expand_method_configs(
            [
                {
                    "name": "ttc",
                    "prompt_style": "english_cot",
                    "selection": "majority_vote",
                    "n_values": [1, 4, 8],
                    "nested_n": True,
                    "temperature": 0.7,
                    "greedy_n1": True,
                    "max_tokens": 64,
                }
            ]
        )
        backend = _CountingBackend()

        # Patch build_backend to return our counter.
        import ttcs_yoruba.inference as inf

        original = inf.build_backend
        inf.build_backend = lambda *a, **k: backend  # type: ignore[assignment]
        try:
            config = InferenceRunConfig(
                run_id="nested_test",
                output_dir=root / "runs",
                datasets=[
                    DatasetConfig(
                        name="toy",
                        path=data_path,
                        task="math",
                        source_dataset="toy",
                    )
                ],
                models=[
                    InferenceModelConfig(
                        name="fake",
                        backend="transformers",
                        model="fake",
                        size_label="0B",
                    )
                ],
                methods=methods,
                seed=0,
            )
            manifest = run_inference_pipeline(config)
        finally:
            inf.build_backend = original  # type: ignore[assignment]

        # One example × (1 greedy + pool_n=8).
        assert backend.calls == 9
        assert manifest["counts"]["model_generations"] == 9
        assert manifest["counts"]["selection_rows"] == 3  # k=1,4,8
        assert backend.temperatures[0] == 0.0
        assert all(t == 0.7 for t in backend.temperatures[1:])
        assert manifest["nested_groups"]["ttc"]["greedy_n1"] is True
        assert manifest["nested_groups"]["ttc"]["pool_n"] == 8

        candidates = [
            json.loads(line)
            for line in (root / "runs" / "nested_test" / "candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        # Materialized rows: 1 + 4 + 8 = 13 stored (9 model calls).
        assert len(candidates) == 13
        by_method = {}
        for row in candidates:
            by_method.setdefault(row["method"], []).append(row)
        assert len(by_method["ttc_n1"]) == 1
        assert len(by_method["ttc_n4"]) == 4
        assert len(by_method["ttc_n8"]) == 8
        # Greedy N=1 is tagged and is not the first stochastic pool sample.
        assert by_method["ttc_n1"][0]["metadata"].get("nested_greedy_n1") is True
        assert by_method["ttc_n4"][0]["metadata"].get("nested_greedy_n1") is not True
        assert by_method["ttc_n1"][0]["response"] != by_method["ttc_n4"][0]["response"]
        # Nested prefixes among stochastic k: first 4 of n8 equal n4.
        assert [r["response"] for r in by_method["ttc_n4"]] == [
            r["response"] for r in by_method["ttc_n8"][:4]
        ]


def test_nested_without_greedy_n1_uses_first_of_pool() -> None:
    """When greedy_n1 is false, N=1 is first-of-pool (no extra decode)."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_path = root / "data.jsonl"
        data_path.write_text(
            json.dumps(
                {
                    "question": "Melo?",
                    "gold_answer": "11",
                    "answer_type": "number",
                    "choices": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        methods = expand_method_configs(
            [
                {
                    "name": "ttc",
                    "prompt_style": "english_cot",
                    "selection": "majority_vote",
                    "n_values": [1, 4],
                    "nested_n": True,
                    "temperature": 0.7,
                    "greedy_n1": False,
                    "max_tokens": 64,
                }
            ]
        )
        assert methods[0].temperature == 0.7
        assert not is_nested_greedy_n1(methods[0])
        backend = _CountingBackend()
        import ttcs_yoruba.inference as inf

        original = inf.build_backend
        inf.build_backend = lambda *a, **k: backend  # type: ignore[assignment]
        try:
            config = InferenceRunConfig(
                run_id="nested_sc1",
                output_dir=root / "runs",
                datasets=[
                    DatasetConfig(
                        name="toy",
                        path=data_path,
                        task="math",
                        source_dataset="toy",
                    )
                ],
                models=[
                    InferenceModelConfig(
                        name="fake",
                        backend="transformers",
                        model="fake",
                        size_label="0B",
                    )
                ],
                methods=methods,
                seed=0,
            )
            manifest = run_inference_pipeline(config)
        finally:
            inf.build_backend = original  # type: ignore[assignment]

        # Only pool of 4; N=1 is first-of-pool (no extra greedy call).
        assert backend.calls == 4
        assert manifest["counts"]["model_generations"] == 4
        assert all(t == 0.7 for t in backend.temperatures)

        candidates = [
            json.loads(line)
            for line in (root / "runs" / "nested_sc1" / "candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        by_method: dict[str, list] = {}
        for row in candidates:
            by_method.setdefault(row["method"], []).append(row)
        assert by_method["ttc_n1"][0]["response"] == by_method["ttc_n4"][0]["response"]
        assert by_method["ttc_n1"][0]["metadata"].get("nested_greedy_n1") is not True


if __name__ == "__main__":
    test_expand_sets_nested_group()
    test_expand_without_nested_has_no_group()
    test_e2_config_is_nested()
    test_materialize_slice_tags_method_and_n()
    test_pick_sample_method_uses_pool_n()
    test_nested_pipeline_greedy_n1_plus_pool()
    test_nested_without_greedy_n1_uses_first_of_pool()
    print("ok")
