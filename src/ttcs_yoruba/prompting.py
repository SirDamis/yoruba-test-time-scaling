from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .examples import InferenceExample


@dataclass(frozen=True)
class Exemplar:
    question: str
    choices: list[str] | None
    reasoning: str
    answer: str


TASK_EXEMPLARS: dict[str, list[Exemplar]] = {
    "math": [
        Exemplar(
            question="Bàbá kan ra àpótí mẹwa (10) ti eso osan. Àpótí kọọkan ni eso osan 12. Bàbá naa fi eso osan 15 fun awon aladugbo. Awon eso osan melo ni o ku?",
            choices=None,
            reasoning="Start with 10 boxes * 12 oranges per box = 120 total oranges. He gives away 15 oranges. Remaining oranges = 120 - 15 = 105.",
            answer="105",
        ),
        Exemplar(
            question="Mo ti ka oju-iwe 30 ninu iwe ti o ni oju-iwe 250. Mo ni lati ka oju-iwe melo ni iyoku lati pari iwe naa?",
            choices=None,
            reasoning="Total pages = 250. Pages read = 30. Remaining pages = Total pages - Pages read = 250 - 30 = 220.",
            answer="220",
        ),
        Exemplar(
            question="Ọkọ̀ kan gbe àwọn apoti 50. Àpoti kọọkan wọn 8 kilo. Ọkọ̀ naa yọ apoti 20 silẹ ni ibudo akọkọ. Kí ni ìwọ̀n tí ó ṣẹ́kù lórí ọkọ̀ naa ni kilo?",
            choices=None,
            reasoning="Initial weight = 50 boxes * 8 kg = 400 kg. Weight removed at first stop = 20 boxes * 8 kg = 160 kg. Remaining weight = 400 - 160 = 240 kg.",
            answer="240",
        ),
        Exemplar(
            question="Ade bi owo 2,500 naira lo ose. Ti o ba na 700 naira fun ounje ati 350 naira fun ọkọ, iye owo ni o ku fun awon nkan miiran?",
            choices=None,
            reasoning="Total allowance = 2500. Total spent on food and transport = 700 + 350 = 1050. Remaining money = Total - Spent = 2500 - 1050 = 1450.",
            answer="1450",
        ),
    ],
    "qa": [
        Exemplar(
            question="Kínni iyì p nínú 24 = 2p?",
            choices=["p = 4", "p = 8", "p = 12", "p = 24"],
            reasoning="The equation is 24 = 2p. Divide both sides by 2: p = 12. This matches option C.",
            answer="C",
        ),
        Exemplar(
            question="Ms. Perez wa àpapọ̀ máílì 40 ní ọjọ́ 5. Ó wa iye máílì yìí kan náà ní ọjọ́ kọ̀ọ̀kan. Iye máílì mélòó ni Ms. Perez wà ní ọjọ́ kọ̀ọ̀kan?",
            choices=["5", "7", "8", "9"],
            reasoning="Total miles = 40, total days = 5. Same distance each day: 40 ÷ 5 = 8 miles per day. The answer is C.",
            answer="C",
        ),
        Exemplar(
            question="Ṣàwarí òpó iye −40 ÷ (−8).",
            choices=["1 lórí 5", "-5", "−1 lórí 5", "5"],
            reasoning="−40 ÷ (−8) = 5. Negative divided by negative equals positive. 40 ÷ 8 = 5, so the answer is D.",
            answer="D",
        ),
        Exemplar(
            question="Báwo ni 1/4 + 2/4 ṣe dọ́gba?",
            choices=["1/4", "4/3", "3/4", "1/2"],
            reasoning="When adding fractions with the same denominator, add the numerators: 1/4 + 2/4 = (1+2)/4 = 3/4. The answer is C.",
            answer="C",
        ),
    ],
    "reading_comprehension": [
        Exemplar(
            question="Àyọkà:\nỌ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́ ní ilé ẹ̀kọ́ alákọ̀ọ́bẹ̀rẹ̀. Ó ní ọmọ ọdún mẹ́tàlélógún (23) nínú kíláàsì rẹ̀.\n\nÌbéèrè:\nKí ni iṣẹ́ Ọ̀jẹ̀ẹ́ Adéjùmọ́?",
            choices=["Oníṣẹ̀-ọ̀gbìn", "Olùkọ́", "Dókítà", "Awẹ̀"],
            reasoning="The passage states 'Ọ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́', meaning she teaches students. So she is a teacher, which is option B (Olùkọ́).",
            answer="B",
        ),
        Exemplar(
            question="Àyọkà:\nỌjà ńlá kan wà ní àárín ìlú. Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá. Wọ́n máa ń ta ẹ̀fọ́, ẹja, àti ẹran.\n\nÌbéèrè:\nỌjọ́ wo ni àwọn olùtajà máa ń kó ọjà wá?",
            choices=["Ọjọ́ Ajé", "Ọjọ́ Ìṣẹ́gun", "Ọjọ́bọ̀", "Ọjọ́ Ẹtì"],
            reasoning="The passage says 'Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá' (On Thursday, traders bring goods). The answer is C (Ọjọ́bọ̀).",
            answer="C",
        ),
        Exemplar(
            question="Àyọkà:\nÓ jẹ́ ọmọ ọdún mẹ́wàá (10). Ó ní ìwé mẹ́rin: ìwé ìròyìn, ìwé àròsọ, ìwé ẹ̀kọ́ ìtàn, àti ìwé ẹ̀kọ́ ìmọ̀ ìjìnlẹ̀.\n\nÌbéèrè:\nÌwé mélòó ni ọmọ náà ní?",
            choices=["Ìwé méjì", "Ìwé mẹ́rin", "Ìwé mẹ́fà", "Ìwé mẹ́jọ"],
            reasoning="The passage says 'Ó ní ìwé mẹ́rin' (He has 4 books), so the answer is B (Ìwé mẹ́rin).",
            answer="B",
        ),
        Exemplar(
            question="Àyọkà:\nAdìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù. Ariwo rẹ̀ máa ń jí gbogbo agbárílé.\n\nÌbéèrè:\nKí ni Adìẹ àgbààgbà máa ń ṣe ní àárọ̀ kùtùkùtù?",
            choices=["Ó máa ń sùn", "Ó máa ń pariwo", "Ó máa ń jẹun", "Ó máa ń lọ sí ọjà"],
            reasoning="The passage says 'Adìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù' (The old rooster crows early in the morning). The answer is B.",
            answer="B",
        ),
    ],
}


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
    if prompt_style == "english_direct":
        return render_english_direct_prompt(example)
    if prompt_style == "yoruba_direct":
        return render_yoruba_direct_prompt(example)
    if prompt_style == "best_of_n_direct":
        return render_best_of_n_direct_prompt(example)
    raise ValueError(f"Unsupported prompt_style: {prompt_style!r}")


def render_exemplar_block(task: str) -> str:
    exemplars = TASK_EXEMPLARS.get(task)
    if not exemplars:
        return ""

    blocks = ["Examples:"]
    for index, ex in enumerate(exemplars, start=1):
        parts = [f"Question:\n{ex.question}"]
        if ex.choices:
            parts.append("Choices:\n" + "\n".join(format_choices(ex.choices)))
        parts.append(f"Reasoning: {ex.reasoning}")
        parts.append(f"Final answer: {ex.answer}")
        blocks.append("\n\n".join(parts))

    blocks.append("Now answer the following question.")
    return "\n\n---\n\n".join(blocks) + "\n\n"


def render_english_cot_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "You are solving questions from a Yoruba benchmark. "
        "Think carefully through the problem in English. Provide the correct final answer."
    )
    exemplar_block = render_exemplar_block(example.task)
    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            "Think through the problem step by step in English.",
            "At the end, output: Final answer: <answer>",
        ]
    )
    return PromptBundle(system=system, user=user)


def render_yoruba_cot_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "O n yanju awon ibeere lati idanwo Yorùbá. "
        "Ronu daradara nipa ibeere naa ni èdè Yorùbá. "
        "Fun idahun ikẹhin ti o tọ."
    )
    exemplar_block = render_exemplar_block(example.task)
    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            "Ronu nipa ibeere naa ni igbese-nipasẹ-igbesẹ ni Yorùbá."
            "Ni ipari, kọ: Final answer: <answer>",
        ]
    )
    return PromptBundle(system=system, user=user)


def render_english_direct_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "You are solving questions from a Yoruba benchmark. "
        "Provide the correct final answer directly, without step-by-step reasoning."
    )
    exemplar_block = render_exemplar_block(example.task)
    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            "Output only the final answer without reasoning.",
            "End with exactly one final line in this format:\nFinal answer: <answer>",
        ]
    )
    return PromptBundle(system=system, user=user)


def render_yoruba_direct_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "O n yanju awon ibeere lati idanwo Yorùbá. "
        "Fun idahun ikẹhin taara, laisi ironu-igbesẹ-nipase-igbesẹ."
    )
    exemplar_block = render_exemplar_block(example.task)
    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            "Ko idahun ikẹhin nikan jade laisi ironu-igbesẹ-nipase-igbesẹ.",
            "Pari pelu ila ikeyin kan pere ni ona yi:\nFinal answer: <idahun>",
        ]
    )
    return PromptBundle(system=system, user=user)


def render_best_of_n_direct_prompt(example: InferenceExample) -> PromptBundle:
    system = (
        "You are solving questions from a Yoruba benchmark. This is one sampled candidate "
        "in a Best-of-N run. Provide the correct final answer directly, "
        "without step-by-step reasoning."
    )
    exemplar_block = render_exemplar_block(example.task)
    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            "Output only the final answer without reasoning.",
            "End with exactly one final line in this format:\nFinal answer: <answer>",
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
        return "Answer format: output only the final number."
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
