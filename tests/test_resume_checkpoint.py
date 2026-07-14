"""Tests for inference resume / checkpoint behavior."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import (
    DatasetConfig,
    InferenceMethodConfig,
    InferenceModelConfig,
    InferenceRunConfig,
    expand_method_configs,
)
from ttcs_yoruba.inference import (
    CHECKPOINT_FILENAME,
    load_completed_units,
    nested_unit_key,
    prepare_resume_artifacts,
    run_inference_pipeline,
    standalone_unit_key,
)
from ttcs_yoruba.metrics import aggregate_run_dir


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs):
        from ttcs_yoruba.schema import BackendOutput

        self.calls += 1
        return BackendOutput(
            response=f"Final answer: {self.calls}",
            token_count=3,
            latency_s=0.001,
        )


def _write_two_examples(path: Path) -> None:
    rows = [
        {
            "id": "ex1",
            "question": "Q1?",
            "gold_answer": "1",
            "answer_type": "number",
            "choices": None,
        },
        {
            "id": "ex2",
            "question": "Q2?",
            "gold_answer": "2",
            "answer_type": "number",
            "choices": None,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_config(root: Path, data_path: Path, methods: list) -> InferenceRunConfig:
    return InferenceRunConfig(
        run_id="resume_test",
        output_dir=root / "runs",
        datasets=[
            DatasetConfig(name="toy", path=data_path, task="math", source_dataset="toy")
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


def test_resume_skips_completed_standalone_units() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_path = root / "data.jsonl"
        _write_two_examples(data_path)
        methods = [
            InferenceMethodConfig(
                name="english_cot",
                prompt_style="english_cot",
                selection="first",
                n=2,
                temperature=0.0,
                max_tokens=32,
            )
        ]
        backend = _CountingBackend()
        import ttcs_yoruba.inference as inf

        original = inf.build_backend
        inf.build_backend = lambda *a, **k: backend  # type: ignore[assignment]
        try:
            config = _make_config(root, data_path, methods)
            m1 = run_inference_pipeline(config, resume=True)
            calls_after_first = backend.calls
            assert calls_after_first == 4  # 2 examples × n=2
            assert m1["counts"]["units_completed_this_run"] == 2
            assert m1["counts"]["units_skipped"] == 0

            m2 = run_inference_pipeline(config, resume=True)
            assert backend.calls == calls_after_first  # no new generations
            assert m2["counts"]["units_skipped"] == 2
            assert m2["counts"]["units_completed_this_run"] == 0
            assert m2["counts"]["model_generations"] == 0
        finally:
            inf.build_backend = original  # type: ignore[assignment]

        ckpt = root / "runs" / "resume_test" / CHECKPOINT_FILENAME
        done = load_completed_units(ckpt)
        assert standalone_unit_key("toy", "fake", "english_cot", "ex1") in done
        assert standalone_unit_key("toy", "fake", "english_cot", "ex2") in done


def test_overwrite_reruns_everything() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_path = root / "data.jsonl"
        _write_two_examples(data_path)
        methods = [
            InferenceMethodConfig(
                name="english_cot",
                prompt_style="english_cot",
                selection="first",
                n=1,
                temperature=0.0,
                max_tokens=32,
            )
        ]
        backend = _CountingBackend()
        import ttcs_yoruba.inference as inf

        original = inf.build_backend
        inf.build_backend = lambda *a, **k: backend  # type: ignore[assignment]
        try:
            config = _make_config(root, data_path, methods)
            run_inference_pipeline(config, resume=True)
            assert backend.calls == 2
            run_inference_pipeline(config, overwrite=True)
            assert backend.calls == 4  # full re-run
        finally:
            inf.build_backend = original  # type: ignore[assignment]


def test_nested_resume_unit_is_whole_group() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_path = root / "data.jsonl"
        _write_two_examples(data_path)
        methods = expand_method_configs(
            [
                {
                    "name": "ttc",
                    "prompt_style": "english_cot",
                    "selection": "majority_vote",
                    "n_values": [1, 4],
                    "nested_n": True,
                    "temperature": 0.7,
                    "greedy_n1": True,
                    "max_tokens": 32,
                }
            ]
        )
        backend = _CountingBackend()
        import ttcs_yoruba.inference as inf

        original = inf.build_backend
        inf.build_backend = lambda *a, **k: backend  # type: ignore[assignment]
        try:
            config = _make_config(root, data_path, methods)
            m1 = run_inference_pipeline(config, resume=True)
            # 2 examples × (1 greedy N=1 + pool_n=4)
            assert backend.calls == 10
            assert m1["counts"]["units_completed_this_run"] == 2
            m2 = run_inference_pipeline(config, resume=True)
            assert backend.calls == 10
            assert m2["counts"]["units_skipped"] == 2
        finally:
            inf.build_backend = original  # type: ignore[assignment]

        done = load_completed_units(root / "runs" / "resume_test" / CHECKPOINT_FILENAME)
        assert nested_unit_key("toy", "fake", "ttc", "ex1") in done


def test_resume_recovers_when_checkpoint_missing_after_write() -> None:
    """Crash window: rows on disk, no checkpoint → resume must not re-generate."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_path = root / "data.jsonl"
        _write_two_examples(data_path)
        methods = [
            InferenceMethodConfig(
                name="english_cot",
                prompt_style="english_cot",
                selection="first",
                n=1,
                temperature=0.0,
                max_tokens=32,
            )
        ]
        backend = _CountingBackend()
        import ttcs_yoruba.inference as inf

        original = inf.build_backend
        inf.build_backend = lambda *a, **k: backend  # type: ignore[assignment]
        try:
            config = _make_config(root, data_path, methods)
            run_inference_pipeline(config, resume=True)
            assert backend.calls == 2

            run_dir = root / "runs" / "resume_test"
            ckpt = run_dir / CHECKPOINT_FILENAME
            assert ckpt.exists()
            # Simulate crash after write but before / without durable checkpoint.
            ckpt.unlink()

            run_inference_pipeline(config, resume=True)
            # Recovered from selections.jsonl — no extra generations.
            assert backend.calls == 2
            candidates = [
                json.loads(line)
                for line in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(candidates) == 2  # not 4
        finally:
            inf.build_backend = original  # type: ignore[assignment]


def test_prepare_resume_dedupes_duplicate_appends() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cand = root / "candidates.jsonl"
        sel = root / "selections.jsonl"
        # Same unit written twice (resume-append bug).
        rows = [
            {
                "dataset": "toy",
                "model": "fake",
                "method": "english_cot",
                "n": 1,
                "example_id": "ex1",
                "sample_index": 0,
                "extracted_answer": "1",
                "token_count": 3,
                "metadata": {"dataset": "toy", "gold_answer": "1", "answer_type": "number"},
            },
            {
                "dataset": "toy",
                "model": "fake",
                "method": "english_cot",
                "n": 1,
                "example_id": "ex1",
                "sample_index": 0,
                "extracted_answer": "1",
                "token_count": 3,
                "metadata": {"dataset": "toy", "gold_answer": "1", "answer_type": "number"},
            },
        ]
        cand.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        sel_rows = [
            {
                "dataset": "toy",
                "model": "fake",
                "method": "english_cot",
                "n": 1,
                "example_id": "ex1",
                "is_correct": True,
                "selected_answer": "1",
            },
            {
                "dataset": "toy",
                "model": "fake",
                "method": "english_cot",
                "n": 1,
                "example_id": "ex1",
                "is_correct": True,
                "selected_answer": "1",
            },
        ]
        sel.write_text("\n".join(json.dumps(r) for r in sel_rows) + "\n", encoding="utf-8")

        done, stats = prepare_resume_artifacts(
            candidates_path=cand,
            selections_path=sel,
            nested_methods_by_group={},
        )
        assert stats["candidate_duplicates_dropped"] == 1
        assert stats["selection_duplicates_dropped"] == 1
        assert standalone_unit_key("toy", "fake", "english_cot", "ex1") in done
        candidates = [
            json.loads(line) for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert len(candidates) == 1


def test_metrics_dedupe_duplicate_candidates() -> None:
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        run_dir.mkdir()
        # Two identical candidate rows for one sample.
        base = {
            "dataset": "toy",
            "model": "fake",
            "method": "english_cot",
            "n": 1,
            "example_id": "ex1",
            "sample_index": 0,
            "extracted_answer": "1",
            "token_count": 10,
            "latency_s": 1.0,
            "estimated_cost": 0.0,
            "prompt_style": "english_cot",
            "reasoning_language": "en",
            "selection": "first",
            "model_size_label": "0B",
            "metadata": {"dataset": "toy", "gold_answer": "1", "answer_type": "number"},
        }
        (run_dir / "candidates.jsonl").write_text(
            json.dumps(base) + "\n" + json.dumps(base) + "\n", encoding="utf-8"
        )
        (run_dir / "selections.jsonl").write_text(
            json.dumps(
                {
                    "dataset": "toy",
                    "model": "fake",
                    "method": "english_cot",
                    "n": 1,
                    "example_id": "ex1",
                    "is_correct": True,
                    "prompt_style": "english_cot",
                    "reasoning_language": "en",
                    "selection": "first",
                    "model_size_label": "0B",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run"}), encoding="utf-8")
        metrics = aggregate_run_dir(run_dir)
        assert len(metrics) == 1
        assert metrics[0].total_candidates == 1
        assert metrics[0].total_tokens == 10


if __name__ == "__main__":
    test_resume_skips_completed_standalone_units()
    test_overwrite_reruns_everything()
    test_nested_resume_unit_is_whole_group()
    test_resume_recovers_when_checkpoint_missing_after_write()
    test_prepare_resume_dedupes_duplicate_appends()
    test_metrics_dedupe_duplicate_candidates()
    print("ok")
