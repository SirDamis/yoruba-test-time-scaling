from __future__ import annotations

from .config import (
    THINKPRM_ROOT,
    REPO_ROOT,
    CollectConfig,
    ScoreConfig,
    TrainConfig,
    load_json,
    resolve_path,
)
from .data import (
    build_dataset,
    build_teacher,
    build_training_example,
    collect_for_row,
    labels_until_first_error,
    load_prm_rows,
    split_rows,
    write_built_dataset,
)
from .model import AGGREGATIONS, ThinkPrmScorer, aggregate_step_scores
from .parse import (
    extract_boxed_decisions,
    extract_decision,
    extract_step_labels,
    format_step_scores,
    pad_step_labels,
    truncate_after_last_decision,
)
from .prompts import (
    COLLECTION_INSTRUCTION,
    DEFAULT_VERIFICATION_INSTRUCTION,
    build_assistant_target,
    build_verification_user_message,
    format_verification_prompt,
)
from .select import attach_thinkprm_scores
from .steps import render_numbered_solution, split_into_steps, strip_step_prefix

__all__ = [
    "AGGREGATIONS",
    "COLLECTION_INSTRUCTION",
    "CollectConfig",
    "DEFAULT_VERIFICATION_INSTRUCTION",
    "REPO_ROOT",
    "ScoreConfig",
    "THINKPRM_ROOT",
    "ThinkPrmScorer",
    "TrainConfig",
    "aggregate_step_scores",
    "attach_thinkprm_scores",
    "build_assistant_target",
    "build_dataset",
    "build_teacher",
    "build_training_example",
    "build_verification_user_message",
    "collect_for_row",
    "extract_boxed_decisions",
    "extract_decision",
    "extract_step_labels",
    "format_step_scores",
    "format_verification_prompt",
    "labels_until_first_error",
    "load_json",
    "load_prm_rows",
    "pad_step_labels",
    "render_numbered_solution",
    "resolve_path",
    "split_into_steps",
    "split_rows",
    "strip_step_prefix",
    "truncate_after_last_decision",
    "write_built_dataset",
]
