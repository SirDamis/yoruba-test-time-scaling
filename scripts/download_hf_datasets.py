from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.datasets import (
    ALL_LANGUAGE_CODES,
    available_dataset_keys,
    download_yoruba_hf_dataset,
)


def parse_splits(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [split.strip() for split in value.split(",") if split.strip()]


def parse_languages(value: str | None) -> list[str]:
    if not value or not value.strip():
        return ["yor"]
    langs = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [lang for lang in langs if lang not in ALL_LANGUAGE_CODES]
    if invalid:
        raise SystemExit(
            f"Unsupported language(s): {invalid}. Expected codes from {sorted(ALL_LANGUAGE_CODES)}"
        )
    return langs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face benchmarks and write compact per-language JSONL exports."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=available_dataset_keys() + ["all"],
        default=None,
        help="Dataset key to download (e.g. afrimgsm_yor, afrimmlu_hau). Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Comma-separated language codes to download for template datasets "
            f"(codes: {','.join(ALL_LANGUAGE_CODES)}). Default: yor. "
            "Ignored for fully-qualified keys like afrimgsm_translate."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/normalized",
        help="Root directory for normalized outputs.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Optional comma-separated split override, for example train,test.",
    )
    parser.add_argument(
        "--hf-id",
        default=None,
        help="Optional Hugging Face dataset ID override. Only valid when downloading one dataset.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hugging Face config override. Only valid when downloading one dataset.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "stdlib", "datasets"],
        default="auto",
        help="Download backend. auto uses dependency-free registered URLs for the canonical datasets.",
    )
    args = parser.parse_args()
    languages = parse_languages(args.language)

    selected = args.dataset or ["all"]
    if "all" in selected:
        selected = available_dataset_keys()

    # Expand template keys (afrimgsm, afrimmlu) across requested languages.
    expanded: list[str] = []
    for key in selected:
        if key in {"afrimgsm", "afrimmlu"}:
            expanded.extend(f"{key}_{lang}" for lang in languages)
        else:
            expanded.append(key)

    if (args.hf_id or args.config) and len(expanded) != 1:
        raise SystemExit("--hf-id and --config overrides can only be used with exactly one --dataset value")

    for dataset_key in expanded:
        manifest = download_yoruba_hf_dataset(
            dataset_key=dataset_key,
            output_root=args.output_root,
            splits=parse_splits(args.splits),
            hf_id_override=args.hf_id,
            config_override=args.config,
            backend=args.backend,
        )
        print(
            f"{manifest['dataset']}: retained {manifest['total_retained_yoruba_rows']} rows -> {manifest['all_path']}"
        )


if __name__ == "__main__":
    main()
