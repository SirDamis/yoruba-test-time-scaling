from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.datasets import HF_DATASET_REGISTRY, download_yoruba_hf_dataset


def parse_splits(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [split.strip() for split in value.split(",") if split.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets and write compact Yoruba JSONL exports."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(HF_DATASET_REGISTRY) + ["all"],
        default=None,
        help="Dataset key to download. Repeatable. Defaults to all.",
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

    selected = args.dataset or ["all"]
    if "all" in selected:
        selected = sorted(HF_DATASET_REGISTRY)
    if (args.hf_id or args.config) and len(selected) != 1:
        raise SystemExit("--hf-id and --config overrides can only be used with exactly one --dataset value")

    for dataset_key in selected:
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
