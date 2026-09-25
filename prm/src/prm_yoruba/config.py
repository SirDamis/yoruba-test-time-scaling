from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ttcs_yoruba.io_utils import read_json

# prm/src/prm_yoruba/config.py -> [0] pkg, [1] src, [2] prm, [3] repo root
PRM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PRM_ROOT.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    return read_json(resolve_path(path))


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "prm/data/processed"
    sources: list[dict[str, Any]] = field(default_factory=list)
    eval_fraction: float = 0.1
    seed: int = 1234

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BuildConfig":
        return cls(
            output_dir=str(row.get("output_dir", "prm/data/processed")),
            sources=list(row.get("sources", [])),
            eval_fraction=float(row.get("eval_fraction", 0.1)),
            seed=int(row.get("seed", 1234)),
        )


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    model: str
    train_data: str
    output_dir: str = "prm/checkpoints"
    eval_data: str | None = None
    max_seq_length: int = 1024
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 1234
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    prm_interface: str = "qwen2.5-math-prm"
    step_sep_token: str = "<extra_0>"
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TrainConfig":
        return cls(
            run_name=str(row["run_name"]),
            model=str(row["model"]),
            train_data=str(row["train_data"]),
            output_dir=str(row.get("output_dir", "prm/checkpoints")),
            eval_data=None if row.get("eval_data") is None else str(row["eval_data"]),
            max_seq_length=int(row.get("max_seq_length", 1024)),
            num_train_epochs=float(row.get("num_train_epochs", 2.0)),
            per_device_train_batch_size=int(row.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(row.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(row.get("gradient_accumulation_steps", 16)),
            learning_rate=float(row.get("learning_rate", 1e-5)),
            lr_scheduler_type=str(row.get("lr_scheduler_type", "linear")),
            warmup_ratio=float(row.get("warmup_ratio", 0.1)),
            weight_decay=float(row.get("weight_decay", 0.01)),
            logging_steps=int(row.get("logging_steps", 10)),
            save_steps=int(row.get("save_steps", 200)),
            save_total_limit=int(row.get("save_total_limit", 3)),
            bf16=bool(row.get("bf16", True)),
            gradient_checkpointing=bool(row.get("gradient_checkpointing", True)),
            seed=int(row.get("seed", 1234)),
            use_lora=bool(row.get("use_lora", True)),
            lora_r=int(row.get("lora_r", 16)),
            lora_alpha=int(row.get("lora_alpha", 32)),
            lora_dropout=float(row.get("lora_dropout", 0.05)),
            lora_target_modules=list(row.get("lora_target_modules", [])),
            load_in_4bit=bool(row.get("load_in_4bit", False)),
            trust_remote_code=bool(row.get("trust_remote_code", True)),
            prm_interface=str(row.get("prm_interface", "qwen2.5-math-prm")),
            step_sep_token=str(row.get("step_sep_token", "<extra_0>")),
            system_prompt=str(
                row.get(
                    "system_prompt",
                    "Please reason step by step, and put your final answer within \\boxed{}.",
                )
            ),
        )


@dataclass(frozen=True)
class ScoreConfig:
    name: str
    model: str
    run_ids: list[str] = field(default_factory=list)
    runs_dir: str = "runs"
    adapter_path: str | None = None
    output_dir: str = "prm/results/prm_selection"
    aggregation: str = "last"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    prm_interface: str = "qwen2.5-math-prm"
    step_sep_token: str = "<extra_0>"
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    max_steps: int = 256
    max_input_tokens: int | None = None
    limit_groups: int | None = None
    batch_size: int = 1

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScoreConfig":
        return cls(
            name=str(row["name"]),
            model=str(row["model"]),
            run_ids=list(row.get("run_ids", [])),
            runs_dir=str(row.get("runs_dir", "runs")),
            adapter_path=None if row.get("adapter_path") is None else str(row["adapter_path"]),
            output_dir=str(row.get("output_dir", "prm/results/prm_selection")),
            aggregation=str(row.get("aggregation", "last")),
            device_map=str(row.get("device_map", "auto")),
            torch_dtype=str(row.get("torch_dtype", "auto")),
            load_in_4bit=bool(row.get("load_in_4bit", False)),
            trust_remote_code=bool(row.get("trust_remote_code", True)),
            prm_interface=str(row.get("prm_interface", "qwen2.5-math-prm")),
            step_sep_token=str(row.get("step_sep_token", "<extra_0>")),
            system_prompt=str(
                row.get(
                    "system_prompt",
                    "Please reason step by step, and put your final answer within \\boxed{}.",
                )
            ),
            max_steps=int(row.get("max_steps", 256)),
            max_input_tokens=None
            if row.get("max_input_tokens") is None
            else int(row["max_input_tokens"]),
            limit_groups=None if row.get("limit_groups") is None else int(row["limit_groups"]),
            batch_size=int(row.get("batch_size", 1)),
        )
