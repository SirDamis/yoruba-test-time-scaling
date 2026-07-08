from __future__ import annotations

from dataclasses import dataclass

from .examples import InferenceExample


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str

    def as_text(self) -> str:
        return f"System:\n{self.system}\n\nUser:\n{self.user}"


def render_prompt(example: InferenceExample, prompt_style: str) -> PromptBundle:
    if prompt_style == "english_cot":
        return render_english_cot_prompt(example)
    if prompt_style == "yoruba_cot":
        return render_yoruba_cot_prompt(example)
    if prompt_style == "best_of_n_cot":
        return render_english_cot_prompt(example)
    raise ValueError(f"Unsupported prompt_style: {prompt_style!r}")


def render_english_cot_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "You are solving questions from a Yoruba benchmark. "
        "Think carefully through the problem in English. Provide the correct final answer."
    )
    user = "\n\n".join(
        [
            render_problem_block(example),
            render_answer_format(example),
            "Think through the problem step by step in English.",
            "At the end, output: Final answer: <answer>"
        ]
    )
    return PromptBundle(system=system, user=user)


def render_yoruba_cot_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "O n yanju awon ibeere lati idanwo Yorùbá.",
        "Ronu daradara nipa ibeere naa ni èdè Yorùbá.",
        "Fun idahun ikẹhin ti o tọ."
    )
    user = "\n\n".join(
        [
            render_problem_block(example),
            render_answer_format(example),
            "Ronu nipa ibeere naa ni igbese-nipasẹ-igbesẹ ni Yorùbá."
            "Ni ipari, kọ: Final answer: <answer>",
        ]
    )
    return PromptBundle(system=system, user=user)



def render_problem_block(example: InferenceExample) -> str:
    parts = [f"Question:\n{example.question}"]
    if example.choices:
        parts.append("Choices:\n" + "\n".join(format_choices(example.choices)))
    return "\n\n".join(parts)


def render_answer_format(example: InferenceExample) -> str:
    if example.answer_type == "choice":
        return "Answer format: output only the option letter, such as A, B, C, or D."
    if example.answer_type == "number":
        return "Answer format: output only the final number, without extra explanation."
    return "Answer format: output a concise final answer in Yoruba."


def format_choices(choices: list[str]) -> list[str]:
    formatted = []
    for index, choice in enumerate(choices):
        label = chr(ord("A") + index)
        stripped = choice.strip()
        if len(stripped) >= 2 and stripped[0].upper() == label and stripped[1] in {".", ")"}:
            formatted.append(stripped)
        else:
            formatted.append(f"{label}. {stripped}")
    return formatted
