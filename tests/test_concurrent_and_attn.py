"""Tests for concurrent batching helpers and attention resolution."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.backends import resolve_attn_implementation
from ttcs_yoruba.config import InferenceModelConfig, InferenceRunConfig, load_inference_run_config
from ttcs_yoruba.inference import (
    effective_max_concurrent,
    run_concurrent_map,
    run_concurrent_map_tolerant,
)


def test_run_concurrent_map_preserves_order() -> None:
    def work(x: int) -> int:
        # Reverse-ish timing so completion order != input order.
        time.sleep(0.01 * (5 - x))
        return x * 10

    out = run_concurrent_map([1, 2, 3, 4], work, max_workers=4)
    assert out == [10, 20, 30, 40]


def test_run_concurrent_map_serial_when_one_worker() -> None:
    out = run_concurrent_map([1, 2, 3], lambda x: x + 1, max_workers=1)
    assert out == [2, 3, 4]


def test_run_concurrent_map_tolerant_keeps_successes_on_partial_failure() -> None:
    def work(x: int) -> int:
        if x == 2:
            raise RuntimeError("fail-2")
        return x * 10

    results, errors = run_concurrent_map_tolerant([1, 2, 3], work, max_workers=3)
    assert results == [10, None, 30]
    assert len(errors) == 1
    assert errors[0][0] == 1
    assert "fail-2" in str(errors[0][1])


def test_effective_max_concurrent_clamps_transformers() -> None:
    cfg = InferenceRunConfig(
        run_id="t",
        output_dir=Path("runs"),
        datasets=[],
        models=[],
        methods=[],
        max_concurrent=16,
    )
    # Bypass empty validation by constructing partially — use replace-style via object.__new__?
    # Build from dict instead.
    from ttcs_yoruba.config import DatasetConfig, InferenceMethodConfig

    cfg = InferenceRunConfig(
        run_id="t",
        output_dir=Path("runs"),
        datasets=[
            DatasetConfig(name="d", path=Path("x.jsonl"), task="math", source_dataset="d")
        ],
        models=[
            InferenceModelConfig(
                name="m", backend="transformers", model="x", size_label="4B"
            )
        ],
        methods=[
            InferenceMethodConfig(
                name="english_cot",
                prompt_style="english_cot",
                selection="first",
                n=1,
            )
        ],
        max_concurrent=16,
    )
    assert effective_max_concurrent(cfg, cfg.models[0]) == 1
    oai = InferenceModelConfig(
        name="m",
        backend="openai_compatible",
        model="x",
        size_label="4B",
        base_url_env="OPENAI_COMPATIBLE_BASE_URL",
        api_key_env="OPENAI_COMPATIBLE_API_KEY",
    )
    assert effective_max_concurrent(cfg, oai) == 16


def test_resolve_attn_auto_cpu_or_fallback() -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    assert resolve_attn_implementation("auto", _Torch()) == "sdpa"
    assert resolve_attn_implementation("flash_attention_2", _Torch()) == "flash_attention_2"
    assert resolve_attn_implementation("sdpa", _Torch()) == "sdpa"


def test_resolve_attn_auto_turing_like() -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(_idx: int = 0) -> tuple[int, int]:
            return (7, 5)  # T4

    class _Torch:
        cuda = _Cuda()

    assert resolve_attn_implementation("auto", _Torch()) == "sdpa"


def test_resolve_attn_auto_ada_without_flash_pkg() -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(_idx: int = 0) -> tuple[int, int]:
            return (8, 9)  # L4

    class _Torch:
        cuda = _Cuda()

    # Without flash_attn installed, should fall back to sdpa.
    impl = resolve_attn_implementation("auto", _Torch())
    assert impl in {"flash_attention_2", "sdpa"}


def test_e1_vllm_config_has_concurrency_and_512() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language_vllm.json")
    assert cfg.max_concurrent == 8
    assert all(m.max_tokens == 512 for m in cfg.methods)


def test_e1_hf_config_has_auto_attn_and_512() -> None:
    cfg = load_inference_run_config(ROOT / "configs" / "e1_reasoning_language.json")
    assert cfg.max_concurrent == 1
    assert all(m.max_tokens == 512 for m in cfg.methods)
    for model in cfg.models:
        assert model.backend_kwargs.get("attn_implementation") == "auto"


if __name__ == "__main__":
    test_run_concurrent_map_preserves_order()
    test_run_concurrent_map_serial_when_one_worker()
    test_run_concurrent_map_tolerant_keeps_successes_on_partial_failure()
    test_effective_max_concurrent_clamps_transformers()
    test_resolve_attn_auto_cpu_or_fallback()
    test_resolve_attn_auto_turing_like()
    test_resolve_attn_auto_ada_without_flash_pkg()
    test_e1_vllm_config_has_concurrency_and_512()
    test_e1_hf_config_has_auto_attn_and_512()
    print("ok")
