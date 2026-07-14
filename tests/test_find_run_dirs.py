"""Tests for find_run_dirs filtering."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.metrics import find_run_dirs


def _make_run(path: Path, *, with_manifest: bool = True, candidates: str = "{}\n") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidates.jsonl").write_text(candidates, encoding="utf-8")
    if with_manifest:
        (path / "manifest.json").write_text(
            json.dumps({"run_id": path.name}), encoding="utf-8"
        )


def test_find_run_dirs_requires_manifest_by_default() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_run(root / "good_run", with_manifest=True)
        _make_run(root / "no_manifest", with_manifest=False)
        found = find_run_dirs(root)
        assert [p.name for p in found] == ["good_run"]


def test_find_run_dirs_excludes_smoke_and_sample_names() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_run(root / "e2_ttc_scaling")
        _make_run(root / "sample_prompts")
        _make_run(root / "_smoke")
        _make_run(root / "synthetic_e2")
        _make_run(root / "my_smoke_test")
        found = {p.name for p in find_run_dirs(root)}
        assert found == {"e2_ttc_scaling"}


def test_find_run_dirs_skips_empty_candidates() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_run(root / "empty", candidates="")
        _make_run(root / "ok", candidates='{"x":1}\n')
        found = [p.name for p in find_run_dirs(root)]
        assert found == ["ok"]


if __name__ == "__main__":
    test_find_run_dirs_requires_manifest_by_default()
    test_find_run_dirs_excludes_smoke_and_sample_names()
    test_find_run_dirs_skips_empty_candidates()
    print("ok")
