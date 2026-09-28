from __future__ import annotations

import argparse
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

from thinkprm_yoruba.config import TrainConfig, load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a ThinkPRM generative verifier on verification CoTs."
    )
    parser.add_argument("--config", default="thinkprm/configs/train_thinkprm.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--full-finetune", action="store_true", help="Disable LoRA.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable QLoRA.")
    args = parser.parse_args()

    config = TrainConfig.from_dict(load_json(args.config))
    overrides = {}
    for key, value in {
        "run_name": args.run_name,
        "model": args.model,
        "train_data": args.train_data,
        "eval_data": args.eval_data,
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
    }.items():
        if value is not None:
            overrides[key] = value
    if args.full_finetune:
        overrides["use_lora"] = False
    if args.load_in_4bit:
        overrides["load_in_4bit"] = True
    if overrides:
        config = replace(config, **overrides)

    from thinkprm_yoruba.train import run_training

    output_dir = run_training(config)
    print(f"Saved ThinkPRM to {output_dir}")


if __name__ == "__main__":
    main()
