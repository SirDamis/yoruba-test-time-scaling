"""ThinkPRM verification prompt templates.

Adapted from the official ThinkPRM release (``utils/helper.py``), which formats
a math problem plus a proposed step-by-step solution and asks the verifier to
critique every step before emitting a final ``Yes``/``No`` decision.

The exact templates are reproduced here so training and inference stay in sync:
``format_verification_prompt`` is the inference prompt, while
``build_assistant_target`` builds the supervised completion used for SFT.
"""

from __future__ import annotations

from typing import Any

PREDECISION_STRING = "Is the solution correct?"

DEFAULT_VERIFICATION_INSTRUCTION = (
    "Review and critique each step in the proposed solution to determine whether "
    "each step is correct. If the solution is incomplete, only verify the provided "
    "steps."
)

# Used only when synthesising training chains with a teacher model: it makes the
# per-step judgement format explicit so the outputs can be filtered against the
# gold PRM800K labels.
COLLECTION_INSTRUCTION = (
    "Review and critique each step in the proposed solution. For every step, "
    "first explain whether the step is valid, then end that step's critique with "
    "a judgement of \\boxed{correct} or \\boxed{incorrect}. If the solution is "
    "incomplete, only verify the provided steps."
)


def _render_template(problem: str, solution: str, instruction: str | None) -> str:
    # Avoid ``str.format`` here: the instruction can legitimately contain braces
    # (e.g. ``\\boxed{correct}``), which would be parsed as format fields.
    return (
        "You are given a math problem and a proposed step-by-step solution:\n\n"
        "[Math Problem]\n\n"
        f"{str(problem).strip()}\n\n"
        "[Solution]\n\n"
        f"{str(solution).strip()}\n\n"
        f"{(instruction or DEFAULT_VERIFICATION_INSTRUCTION).strip()}"
    )


def build_verification_user_message(
    problem: str,
    solution: str,
    instruction: str | None = None,
) -> str:
    """Render the user turn asking for a step-by-step verification."""
    return _render_template(problem, solution, instruction)


def format_verification_prompt(
    tokenizer: Any,
    problem: str,
    solution: str,
    instruction: str | None = None,
) -> str:
    """Apply the model chat template with a generation prompt (inference)."""
    message = build_verification_user_message(problem, solution, instruction)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_assistant_target(
    cot: str,
    decision: str,
    *,
    predecision_string: str = PREDECISION_STRING,
    opens_think: bool = True,
) -> str:
    """Build the supervised completion that follows the generation-prompt prefix.

    The ThinkPRM chat template already opens a thinking block on
    ``add_generation_prompt=True`` (``...<|Assistant|><think>\\n``), and it *drops*
    assistant content before ``</think>`` when rendering a chat. To stay faithful
    we therefore emit the continuation manually: the critique body, the closing
    ``</think>`` (only when the prefix opened one), then the decision line.

    ``cot`` is the verification chain (one ``\\boxed{correct}``/``\\boxed{incorrect}``
    judgement per step). ``decision`` is ``"Yes"`` or ``"No"``.
    """
    body = str(cot).strip()
    if opens_think:
        if body.startswith("<think>"):
            body = body[len("<think>") :].lstrip("\n")
        if "</think>" not in body:
            body = f"{body}\n</think>"
    return f"{body}\n{predecision_string} {str(decision).strip()}"
