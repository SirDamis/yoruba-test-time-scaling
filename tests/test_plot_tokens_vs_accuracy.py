"""Tests for the accuracy-vs-tokens frontier plot helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_tokens_vs_accuracy", ROOT / "scripts" / "plot_tokens_vs_accuracy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frontier_rows_uses_requested_token_field() -> None:
    module = _load_module()
    estimates = [
        {
            "dataset": "d",
            "model": "m",
            "nested_group_id": "g",
            "pool_n": 8,
            "k": 4,
            "num_examples": 3,
            "pass_at_k": 0.5,
            "maj_at_k_mean": 0.2,
            "total_tokens_per_example_at_k": 4000.0,
            "completion_tokens_per_example_at_k": 1500.0,
            "prompt_tokens_per_example_at_k": 2500.0,
        }
    ]
    rows = module.frontier_rows(estimates, x_field="completion_tokens_per_example_at_k")
    assert len(rows) == 1
    assert rows[0]["k"] == 4
    assert rows[0]["tokens_per_example"] == 1500.0
    assert rows[0]["pass_at_k"] == 0.5


if __name__ == "__main__":
    test_frontier_rows_uses_requested_token_field()
    print("ok")
