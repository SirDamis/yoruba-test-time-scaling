from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

THINKPRM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = THINKPRM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(THINKPRM_ROOT.parent / "prm" / "src"))
sys.path.insert(0, str(THINKPRM_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from thinkprm_yoruba.config import CollectConfig, load_json
from thinkprm_yoruba.data import build_dataset, split_rows, write_built_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesise ThinkPRM verification chains with a reasoning teacher and "
            "keep only chains whose step judgements match the gold process labels."
        )
    )
    parser.add_argument("--config", default="thinkprm/configs/collect_data.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--samples-per-example", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = CollectConfig.from_dict(load_json(args.config))
    overrides = {}
    for key, value in {
        "output_dir": args.output_dir,
        "eval_fraction": args.eval_fraction,
        "samples_per_example": args.samples_per_example,
        "max_examples": args.max_examples,
        "seed": args.seed,
    }.items():
        if value is not None:
            overrides[key] = value
    if overrides:
        config = replace(config, **overrides)

    if not config.prm_data:
        raise SystemExit(
            "No inputs configured. Run prm/scripts/build_data.py first, then point "
            "thinkprm/configs/collect_data.json at prm/data/processed/all.jsonl."
        )

    rows = build_dataset(config)
    if not rows:
        raise SystemExit(
            "No verification chains survived filtering. Check the teacher endpoint "
            "and that the gold process labels/instruction are well-formed."
        )

    train, evaluate = split_rows(rows, eval_fraction=config.eval_fraction, seed=config.seed)
    manifest = write_built_dataset(train, evaluate, rows, config.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
