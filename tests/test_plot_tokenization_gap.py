"""Tests for the tokenization-gap plot helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_tokenization_gap", ROOT / "scripts" / "plot_tokenization_gap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_token_rows_aggregates_and_skips_nested_slices() -> None:
    module = _load_module()
    candidates = [
        {
            "dataset": "d",
            "model": "m",
            "method": "english_cot",
            "prompt_style": "english_cot",
            "n": 1,
            "prompt_token_count": 100,
            "token_count": 50,
        },
        {
            "dataset": "d",
            "model": "m",
            "method": "english_cot",
            "prompt_style": "english_cot",
            "n": 1,
            "prompt_token_count": 200,
            "token_count": 70,
        },
        {
            "dataset": "d",
            "model": "m",
            "method": "translate_pivot_ttc_n1",
            "prompt_style": "translate_pivot",
            "n": 1,
            "prompt_token_count": 999,
            "token_count": 1,
        },
    ]
    rows = module.token_rows(candidates)
    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == "english_cot"
    assert row["num_examples"] == 2
    assert row["prompt_tokens_per_example"] == 150.0
    assert row["completion_tokens_per_example"] == 60.0


if __name__ == "__main__":
    test_token_rows_aggregates_and_skips_nested_slices()
    print("ok")
