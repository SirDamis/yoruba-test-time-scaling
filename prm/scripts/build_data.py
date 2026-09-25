from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PRM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PRM_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from prm_yoruba.config import BuildConfig, load_json
from prm_yoruba.data import build_dataset, split_rows, write_built_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build discriminative-PRM data from PRM800K: released English step "
            "labels + a supplied Yoruba translation."
        )
    )
    parser.add_argument("--config", default="prm/configs/build_data.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = BuildConfig.from_dict(load_json(args.config))
    overrides = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.eval_fraction is not None:
        overrides["eval_fraction"] = args.eval_fraction
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        config = replace(config, **overrides)

    rows = build_dataset(config)
    if not rows:
        raise SystemExit(
            "No PRM training rows built. Check the prm800k languages / paths and "
            "that every requested language has either a released file or paths entry."
        )

    train, evaluate = split_rows(rows, eval_fraction=config.eval_fraction, seed=config.seed)
    manifest = write_built_dataset(train, evaluate, rows, config.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
