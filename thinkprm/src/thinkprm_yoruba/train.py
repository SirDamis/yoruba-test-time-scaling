"""LoRA fine-tuning of a ThinkPRM generative verifier.

The loss is a standard causal-LM loss applied only to the supervised assistant
turn (the verification chain-of-thought plus the final ``Yes``/``No`` decision);
the prompt is masked out. This is the lightweight, config-driven analogue of the
paper's full fine-tune.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from .config import TrainConfig, resolve_path
from .parse import DECISION_STRING
from .prompts import build_assistant_target, build_verification_user_message

DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _filter_kwargs(cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        allowed = {field.name for field in dataclass_fields(cls)}
    except TypeError:
        return values
    return {key: value for key, value in values.items() if key in allowed}


def _load_rows(path: str) -> list[dict[str, Any]]:
    from ttcs_yoruba.io_utils import read_jsonl

    return read_jsonl(resolve_path(path))


def decision_for(row: dict[str, Any]) -> str:
    """Prefix decision string for a training row."""
    if "is_correct" in row:
        return "Yes" if bool(row["is_correct"]) else "No"
    labels = [int(label) for label in row.get("labels", [])]
    return "Yes" if labels and labels[-1] == 1 else "No"


def build_prompt_prefix(tokenizer: Any, row: dict[str, Any], config: TrainConfig) -> str:
    """The inference-time prompt, including the generation prompt.

    For the released ThinkPRM base this ends with ``<|Assistant|><think>\\n``.
    """
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": build_verification_user_message(
                    row["problem"], row["solution"], config.instruction
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _prefix_opens_think(prefix: str, tokenizer: Any, config: TrainConfig) -> bool:
    if not config.add_think_token:
        return False
    ids = tokenizer.encode("<think>", add_special_tokens=False)
    if not ids:
        return False
    return prefix.rstrip().endswith(tokenizer.decode(ids[-1]))


def build_training_text(tokenizer: Any, row: dict[str, Any], config: TrainConfig) -> str:
    """Prompt prefix + supervised completion.

    We deliberately do **not** render the assistant turn through
    ``apply_chat_template``: the ThinkPRM chat template drops everything before
    ``</think>`` in an assistant message, which would erase the verification
    critique. Instead we append the completion to the generation-prompt prefix,
    exactly as it is generated at inference time.
    """
    prefix = build_prompt_prefix(tokenizer, row, config)
    target = build_assistant_target(
        row["cot"],
        decision_for(row),
        predecision_string=config.predecision_string,
        opens_think=_prefix_opens_think(prefix, tokenizer, config),
    )
    return prefix + target


def tokenize_example(
    tokenizer: Any,
    row: dict[str, Any],
    config: TrainConfig,
) -> dict[str, list[int]]:
    """Tokenize one row, masking everything before the assistant turn."""
    text = build_training_text(tokenizer, row, config)
    prefix = build_prompt_prefix(tokenizer, row, config)
    prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=config.max_seq_length,
        add_special_tokens=False,
    )
    input_ids = list(tokenized["input_ids"])
    labels = list(input_ids)
    for index in range(min(prefix_length, len(labels))):
        labels[index] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": list(tokenized["attention_mask"]),
        "labels": labels,
    }


def _make_collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        width = max(len(feature["input_ids"]) for feature in features)

        def pad(values: list[int], filler: int) -> list[int]:
            return values + [filler] * (width - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(f["input_ids"], pad_id) for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [pad(f["attention_mask"], 0) for f in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [pad(f["labels"], -100) for f in features], dtype=torch.long
            ),
        }

    return collate


def run_training(config: TrainConfig) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)

    if config.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.lora_target_modules or DEFAULT_LORA_TARGETS,
        )
        model = get_peft_model(model, lora_config)

    def to_tokens(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenize_example(tokenizer, batch, config)

    train_rows = _load_rows(config.train_data)
    eval_rows = _load_rows(config.eval_data) if config.eval_data else []
    train_dataset = Dataset.from_list(train_rows).map(
        to_tokens, remove_columns=list(train_rows[0].keys())
    )
    eval_dataset = (
        Dataset.from_list(eval_rows).map(to_tokens, remove_columns=list(eval_rows[0].keys()))
        if eval_rows
        else None
    )

    output_dir = resolve_path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    arguments_values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "bf16": config.bf16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "seed": config.seed,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if eval_dataset is not None:
        arguments_values["eval_strategy"] = "steps"
        arguments_values["evaluation_strategy"] = "steps"
        arguments_values["eval_steps"] = config.save_steps
    arguments = TrainingArguments(**_filter_kwargs(TrainingArguments, arguments_values))

    trainer_values: dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": _make_collator(tokenizer),
    }
    trainer_kwargs = _filter_kwargs(Trainer, trainer_values)
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)


__all__ = [
    "DECISION_STRING",
    "build_prompt_prefix",
    "build_training_text",
    "decision_for",
    "run_training",
    "tokenize_example",
]
