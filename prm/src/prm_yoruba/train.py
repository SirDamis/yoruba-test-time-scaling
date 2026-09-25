"""LoRA fine-tuning of a process reward model.

Two interfaces are supported:

* ``qwen2.5-math-prm`` (default): the official ``Qwen2ForProcessRewardModel``
  checkpoints. Steps are joined with ``<extra_0>`` and the 2-class reward head is
  trained with cross-entropy at those positions.
* ``token_classifier``: the reference repo's ``+``/``-`` LM-token objective used
  with ``Qwen2.5-Math-7B-Instruct``.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from .config import TrainConfig, resolve_path
from .model import INTERFACES, load_official_model, resolve_prm_tokens, resolve_sep_id
from .steps import QWEN_STEP_TAG, render_sep_joined, split_tagged_process

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


def _steps_and_labels(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        steps = split_tagged_process(str(row.get("process", "")), QWEN_STEP_TAG)
    labels = [str(label) for label in row.get("label", [])]
    n = min(len(steps), len(labels))
    return [str(step) for step in steps[:n]], labels[:n]


def _build_token_classifier_tokenize(tokenizer: Any, tokens: Any, max_seq_length: int):
    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        text = f"{row['question']} {row['process']}"
        tokenized = tokenizer(
            text, truncation=True, padding="max_length", max_length=max_seq_length
        )
        input_ids = list(tokenized["input_ids"])
        length = len(input_ids)
        indices = [i for i, token in enumerate(input_ids) if token == tokens.step_tag_id]
        labels = [str(label) for label in row["label"]]
        if len(indices) != len(labels):
            labels = labels[: len(indices)]
            indices = indices[: len(labels)]

        out_labels = [-100] * length
        attention_mask = [1] * length
        for position, label in zip(indices, labels):
            if label in ("+", "1", "True"):
                out_labels[position] = tokens.good_id
            elif label in ("-", "0", "False"):
                out_labels[position] = tokens.bad_id
            else:
                raise ValueError(f"Unexpected step label {label!r}")
            attention_mask[position] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": out_labels}

    return tokenize


def _build_official_tokenize(
    tokenizer: Any, sep_id: int, sep: str, system_prompt: str, max_seq_length: int
):
    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        steps, labels = _steps_and_labels(row)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(row["question"])},
            {"role": "assistant", "content": render_sep_joined(steps, sep)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text, truncation=True, padding="max_length", max_length=max_seq_length
        )
        input_ids = list(tokenized["input_ids"])
        positions = [i for i, token in enumerate(input_ids) if token == sep_id]
        if len(positions) != len(labels):
            labels = labels[: len(positions)]
            positions = positions[: len(labels)]

        out_labels = [-100] * len(input_ids)
        for position, label in zip(positions, labels):
            out_labels[position] = 1 if label in ("+", "1", "True") else 0
        return {
            "input_ids": input_ids,
            "attention_mask": list(tokenized["attention_mask"]),
            "labels": out_labels,
        }

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


def _preprocess_token_classifier(tokens: Any):
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


def _preprocess_official():
    def preprocess(logits: Any, labels: Any) -> Any:
        import torch

        mask = labels != -100
        selected = logits[mask]  # [n, 2]
        probabilities = torch.softmax(selected, dim=-1)[:, 1]
        return probabilities, labels[mask]

    return preprocess


def run_training(config: TrainConfig) -> str:
    if config.prm_interface not in INTERFACES:
        raise ValueError(
            f"Unknown prm_interface {config.prm_interface!r}. Expected one of {sorted(INTERFACES)}"
        )

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments

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

    if config.prm_interface == "qwen2.5-math-prm":
        model = load_official_model(
            config.model,
            model_kwargs,
            tokenizer,
            trust_remote_code=config.trust_remote_code,
        )
        sep_id = resolve_sep_id(tokenizer, config.step_sep_token)
        tokenize = _build_official_tokenize(
            tokenizer,
            sep_id,
            config.step_sep_token,
            config.system_prompt,
            config.max_seq_length,
        )
        preprocess_logits = _preprocess_official()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
        tokens = resolve_prm_tokens(tokenizer, config.model)
        tokenize = _build_token_classifier_tokenize(tokenizer, tokens, config.max_seq_length)
        preprocess_logits = _preprocess_token_classifier(tokens)

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

    def keep_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "question": row["question"],
                "process": row.get("process", ""),
                "steps": row.get("steps", []),
                "label": row["label"],
            }
            for row in rows
        ]

    remove_columns = ["question", "process", "steps", "label"]
    train_rows = keep_columns(_load_rows(config.train_data))
    eval_rows = keep_columns(_load_rows(config.eval_data)) if config.eval_data else []
    train_dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=remove_columns)
    eval_dataset = (
        Dataset.from_list(eval_rows).map(tokenize, remove_columns=remove_columns)
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
        "compute_metrics": _compute_metrics(),
        "preprocess_logits_for_metrics": preprocess_logits,
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


__all__ = ["run_training", "QWEN_STEP_TAG"]
