from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from ttcs_yoruba.config import InferenceRunConfig, load_inference_run_config
from ttcs_yoruba.inference import run_inference_pipeline


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_language_filter(
    config: InferenceRunConfig,
    dataset_names: set[str] | None,
    languages: set[str] | None,
) -> set[str] | None:
    """Restrict the run to datasets whose name ends with ``_<language>``.

    ``--language yor`` keeps ``afrimgsm_yor`` / ``afrimmlu_yor``;
    ``--language hau,ibo`` keeps both suffixes. ``all`` disables the filter.
    Raises when a requested language has no matching dataset in the config.
    """
    if not languages or "all" in languages:
        return dataset_names

    available = {d.name for d in config.datasets}
    available_langs = {name.rpartition("_")[2] for name in available}
    unknown = languages - available_langs
    if unknown:
        raise SystemExit(
            f"--language {','.join(sorted(unknown))}: no datasets with these suffixes in the config. "
            f"Available language suffixes: {sorted(available_langs)}"
        )

    base = dataset_names if dataset_names is not None else available
    return {name for name in base if name.rpartition("_")[2] in languages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Yoruba TTC inference from a cloud model endpoint.")
    parser.add_argument("--config", default="configs/inference.json", help="Inference config JSON path.")
    parser.add_argument("--run-id", default=None, help="Optional run_id override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names to run.")
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Comma-separated language suffixes to run (e.g. yor or hau,ibo). "
            "Filters datasets named <dataset>_<language>. 'all' disables the filter."
        ),
    )
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
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Max in-flight generations for openai_compatible/vLLM (overrides config). "
            "Transformers backend always uses 1."
        ),
    )
    args = parser.parse_args()

    config = load_inference_run_config(args.config)
    if args.run_id is not None:
        config = replace(config, run_id=args.run_id)
    if args.output_dir is not None:
        config = replace(config, output_dir=Path(args.output_dir))

    dataset_names = resolve_language_filter(
        config,
        parse_csv_set(args.datasets),
        parse_csv_set(args.language),
    )

    manifest = run_inference_pipeline(
        config,
        dataset_names=dataset_names,
        model_names=parse_csv_set(args.models),
        method_names=parse_csv_set(args.methods),
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
        progress=args.progress,
        max_concurrent=args.max_concurrent,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
