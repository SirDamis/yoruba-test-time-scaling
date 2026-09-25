"""LoRA fine-tuning of a discriminative PRM (Qwen2.5-Math-PRM).

Adapted from ``prm/reference/ft.py`` with three changes:
  * LoRA by default instead of full fine-tuning;
  * paths are config-driven rather than hard-coded to ``/data/...``;
  * no DeepSpeed / W&B / Hugging Face Hub side effects by default.

Only the ``+``/``-`` candidate tokens at step boundaries contribute to the loss,
exactly as in the reference.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from .config import TrainConfig, resolve_path
from .model import resolve_prm_tokens
from .steps import QWEN_STEP_TAG

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


def _build_tokenize_fn(tokenizer: Any, tokens: Any, max_seq_length: int):
    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        text = f"{row['question']} {row['process']}"
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )
        input_ids = list(tokenized["input_ids"])
        length = len(input_ids)
        indices = [i for i, token in enumerate(input_ids) if token == tokens.step_tag_id]
        labels = list(row["label"])
        if len(indices) != len(labels):
            labels = labels[: len(indices)]
            indices = indices[: len(labels)]

        out_labels = [-100] * length
        attention_mask = [1] * length
        for position, label in zip(indices, labels):
            if label == "+" or label == 1:
                out_labels[position] = tokens.good_id
            elif label == "-" or label == 0:
                out_labels[position] = tokens.bad_id
            else:
                raise ValueError(f"Unexpected step label {label!r}")
            attention_mask[position] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": out_labels}

    return tokenize


def _make_collator(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        width = max(len(feature["input_ids"]) for feature in features)

        def pad(values: list[int], filler: int) -> list[int]:
            return values + [filler] * (width - len(values))

        input_ids = torch.tensor([pad(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor(
            [pad(f["attention_mask"], 0) for f in features], dtype=torch.long
        )
        labels = torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def _compute_metrics():
    try:
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    except ImportError:
        return None

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        probabilities, gold = eval_pred
        return {
            "auc": float(roc_auc_score(gold, probabilities)),
            "ll": float(log_loss(gold, probabilities)),
            "acc": float(accuracy_score(gold, probabilities > 0.5)),
        }

    return compute_metrics


def _preprocess_logits_for_metrics(tokens: Any):
    def preprocess(logits: Any, labels: Any) -> Any:
        import torch

        positions = torch.argwhere(
            torch.bitwise_or(labels == tokens.good_id, labels == tokens.bad_id)
        )
        gold = torch.where(labels[positions[:, 0], positions[:, 1]] == tokens.bad_id, 0, 1)
        selected = logits[positions[:, 0], positions[:, 1]][:, [tokens.bad_id, tokens.good_id]]
        probabilities = torch.softmax(selected, dim=-1)
        return probabilities[:, 1], gold

    return preprocess


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
    tokens = resolve_prm_tokens(tokenizer, config.model)

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

    tokenize = _build_tokenize_fn(tokenizer, tokens, config.max_seq_length)

    def prm_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Sources carry different extra columns (runs vs prm800k); keep only the
        # three the tokenizer + loss need so Dataset.from_list has a uniform schema.
        return [
            {"question": row["question"], "process": row["process"], "label": row["label"]}
            for row in rows
        ]

    train_rows = prm_columns(_load_rows(config.train_data))
    eval_rows = prm_columns(_load_rows(config.eval_data)) if config.eval_data else []
    train_dataset = Dataset.from_list(train_rows).map(
        tokenize, remove_columns=["question", "process", "label"]
    )
    eval_dataset = (
        Dataset.from_list(eval_rows).map(tokenize, remove_columns=["question", "process", "label"])
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
        # Transformers renamed evaluation_strategy -> eval_strategy; include both
        # and let _filter_kwargs drop whichever the installed version lacks.
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
        "compute_metrics": _compute_metrics(),
        "preprocess_logits_for_metrics": _preprocess_logits_for_metrics(tokens),
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


# Keep the module importable without ``transformers`` installed.
__all__ = ["run_training", "QWEN_STEP_TAG"]
