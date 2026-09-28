from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ttcs_yoruba.io_utils import read_json

from .prompts import DEFAULT_VERIFICATION_INSTRUCTION, PREDECISION_STRING

# thinkprm/src/thinkprm_yoruba/config.py -> [0] pkg, [1] src, [2] thinkprm, [3] repo root
THINKPRM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = THINKPRM_ROOT.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    return read_json(resolve_path(path))


@dataclass(frozen=True)
class CollectConfig:
    """Synthesise verification chains-of-thought for a generative PRM.

    ``prm_data`` points at step-labelled JSONL produced by
    ``prm/scripts/build_data.py`` (or any file with ``question`` + ``steps`` +
    ``label``). A strong reasoning teacher (e.g. QwQ-32B-Preview) critiques each
    step and only chains whose step judgements match the gold labels are kept.
    """

    output_dir: str = "thinkprm/data/processed"
    prm_data: list[str] = field(default_factory=list)
    teacher: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    instruction: str = DEFAULT_VERIFICATION_INSTRUCTION
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    samples_per_example: int = 4
    max_cots_per_example: int = 1
    max_cot_tokens: int | None = None
    max_examples: int | None = None
    eval_fraction: float = 0.1
    request_timeout_s: float = 300.0
    seed: int = 1234

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CollectConfig":
        return cls(
            output_dir=str(row.get("output_dir", "thinkprm/data/processed")),
            prm_data=[str(path) for path in row.get("prm_data", [])],
            teacher=dict(row.get("teacher", {})),
            system_prompt=str(row.get("system_prompt", "")),
            instruction=str(row.get("instruction", DEFAULT_VERIFICATION_INSTRUCTION)),
            temperature=float(row.get("temperature", 0.7)),
            top_p=float(row.get("top_p", 0.95)),
            max_tokens=int(row.get("max_tokens", 4096)),
            samples_per_example=int(row.get("samples_per_example", 4)),
            max_cots_per_example=int(row.get("max_cots_per_example", 1)),
            max_cot_tokens=None
            if row.get("max_cot_tokens") is None
            else int(row["max_cot_tokens"]),
            max_examples=None if row.get("max_examples") is None else int(row["max_examples"]),
            eval_fraction=float(row.get("eval_fraction", 0.1)),
            request_timeout_s=float(row.get("request_timeout_s", 300.0)),
            seed=int(row.get("seed", 1234)),
        )


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    model: str
    train_data: str
    output_dir: str = "thinkprm/checkpoints"
    eval_data: str | None = None
    max_seq_length: int = 4096
    num_train_epochs: float = 1.0
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
    instruction: str = DEFAULT_VERIFICATION_INSTRUCTION
    predecision_string: str = PREDECISION_STRING
    add_think_token: bool = True

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TrainConfig":
        return cls(
            run_name=str(row["run_name"]),
            model=str(row["model"]),
            train_data=str(row["train_data"]),
            output_dir=str(row.get("output_dir", "thinkprm/checkpoints")),
            eval_data=None if row.get("eval_data") is None else str(row["eval_data"]),
            max_seq_length=int(row.get("max_seq_length", 4096)),
            num_train_epochs=float(row.get("num_train_epochs", 1.0)),
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
            instruction=str(row.get("instruction", DEFAULT_VERIFICATION_INSTRUCTION)),
            predecision_string=str(row.get("predecision_string", PREDECISION_STRING)),
            add_think_token=bool(row.get("add_think_token", True)),
        )


@dataclass(frozen=True)
class ScoreConfig:
    name: str
    model: str
    run_ids: list[str] = field(default_factory=list)
    runs_dir: str = "runs"
    adapter_path: str | None = None
    output_dir: str = "thinkprm/results/thinkprm_selection"
    aggregation: str = "prefix"
    instruction: str = DEFAULT_VERIFICATION_INSTRUCTION
    predecision_string: str = PREDECISION_STRING
    max_steps: int = 256
    max_input_tokens: int | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    n_verifications: int = 1
    decision_temperature: float = 1.0
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    limit_groups: int | None = None
    batch_size: int = 4
    # N conditions to select/report. Empty means "all present". N=1 and N=3 are
    # dropped by default (greedy baseline covers N=1).
    n_values: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ScoreConfig":
        return cls(
            name=str(row["name"]),
            model=str(row["model"]),
            run_ids=list(row.get("run_ids", [])),
            runs_dir=str(row.get("runs_dir", "runs")),
            adapter_path=None if row.get("adapter_path") is None else str(row["adapter_path"]),
            output_dir=str(row.get("output_dir", "thinkprm/results/thinkprm_selection")),
            aggregation=str(row.get("aggregation", "prefix")),
            instruction=str(row.get("instruction", DEFAULT_VERIFICATION_INSTRUCTION)),
            predecision_string=str(row.get("predecision_string", PREDECISION_STRING)),
            max_steps=int(row.get("max_steps", 256)),
            max_input_tokens=None
            if row.get("max_input_tokens") is None
            else int(row["max_input_tokens"]),
            max_new_tokens=int(row.get("max_new_tokens", 2048)),
            temperature=float(row.get("temperature", 0.7)),
            top_p=float(row.get("top_p", 0.95)),
            n_verifications=int(row.get("n_verifications", 1)),
            decision_temperature=float(row.get("decision_temperature", 1.0)),
            device_map=str(row.get("device_map", "auto")),
            torch_dtype=str(row.get("torch_dtype", "auto")),
            load_in_4bit=bool(row.get("load_in_4bit", False)),
            trust_remote_code=bool(row.get("trust_remote_code", False)),
            limit_groups=None if row.get("limit_groups") is None else int(row["limit_groups"]),
            batch_size=int(row.get("batch_size", 4)),
            n_values=(
                [int(value) for value in row["n_values"]]
                if row.get("n_values") is not None
                else [2, 4, 8, 16, 32, 64]
            ),
        )
