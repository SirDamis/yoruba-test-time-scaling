from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.config import load_inference_run_config
from ttcs_yoruba.inference import run_inference_pipeline


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Yoruba TTC inference from a cloud model endpoint.")
    parser.add_argument("--config", default="configs/inference.json", help="Inference config JSON path.")
    parser.add_argument("--run-id", default=None, help="Optional run_id override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names to run.")
    parser.add_argument("--models", default=None, help="Comma-separated model names to run.")
    parser.add_argument("--methods", default=None, help="Comma-separated method names to run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset example limit for cloud smoke runs.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from completed_units.jsonl and append (default: true). Use --no-resume to rewrite outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior candidates/selections/checkpoint for this run_id and start clean.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-example progress to stderr (default: true). Use --no-progress to silence.",
    )
    args = parser.parse_args()

    config = load_inference_run_config(args.config)
    if args.run_id is not None:
        config = replace(config, run_id=args.run_id)
    if args.output_dir is not None:
        config = replace(config, output_dir=Path(args.output_dir))

    manifest = run_inference_pipeline(
        config,
        dataset_names=parse_csv_set(args.datasets),
        model_names=parse_csv_set(args.models),
        method_names=parse_csv_set(args.methods),
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
