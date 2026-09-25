from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PRM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRM_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PRM_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from prm_yoruba.config import TrainConfig, load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a discriminative PRM (Qwen2.5-Math-PRM) on mined traces."
    )
    parser.add_argument("--config", default="prm/configs/train_prm.json")
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

    from prm_yoruba.train import run_training

    output_dir = run_training(config)
    print(f"Saved PRM to {output_dir}")


if __name__ == "__main__":
    main()
