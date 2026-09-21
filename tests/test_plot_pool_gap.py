"""Tests for the pass@k vs maj@k gap plot helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_pool_gap", ROOT / "scripts" / "plot_pool_gap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gap_rows_computes_gap() -> None:
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
        }
    ]
    rows = module.gap_rows(estimates)
    assert len(rows) == 1
    assert rows[0]["k"] == 4
    assert abs(rows[0]["gap"] - 0.3) < 1e-9


if __name__ == "__main__":
    test_gap_rows_computes_gap()
    print("ok")
