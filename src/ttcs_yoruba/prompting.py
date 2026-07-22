from __future__ import annotations

from dataclasses import dataclass

from .examples import InferenceExample


@dataclass(frozen=True)
class Exemplar:
    """Few-shot demonstration for a Yoruba benchmark item.

    ``reasoning_en`` / ``reasoning_yo`` support E1 language-matched CoT.
    ``translated_question`` supports the translate-pivot strategy.
    """

    question: str
    choices: list[str] | None
    reasoning_en: str
    reasoning_yo: str
    answer: str
    translated_question: str


TASK_EXEMPLARS: dict[str, list[Exemplar]] = {
    "math": [
        Exemplar(
            question=(
                "Bàbá kan ra àpótí mẹwa (10) ti eso osan. Àpótí kọọkan ni eso osan 12. "
                "Bàbá naa fi eso osan 15 fun awon aladugbo. Awon eso osan melo ni o ku?"
            ),
            choices=None,
            reasoning_en=(
                "Start with 10 boxes * 12 oranges per box = 120 total oranges. "
                "He gives away 15 oranges. Remaining oranges = 120 - 15 = 105."
            ),
            reasoning_yo=(
                "Bẹ̀rẹ̀ pẹ̀lú àpótí 10 × ẹṣọ ọsàn 12 nínú ọ̀kọ̀ọ̀kan = 120 ẹṣọ ọsàn lápapọ̀. "
                "Ó fún àwọn aládùúgbò ní ẹṣọ ọsàn 15. Ẹṣọ ọsàn tó kù = 120 - 15 = 105."
            ),
            answer="105",
            translated_question=(
                "A father bought 10 boxes of oranges. Each box has 12 oranges. "
                "He gave 15 oranges to the neighbors. How many oranges remain?"
            ),
        ),
        Exemplar(
            question=(
                "Mo ti ka oju-iwe 30 ninu iwe ti o ni oju-iwe 250. "
                "Mo ni lati ka oju-iwe melo ni iyoku lati pari iwe naa?"
            ),
            choices=None,
            reasoning_en=(
                "Total pages = 250. Pages read = 30. "
                "Remaining pages = Total pages - Pages read = 250 - 30 = 220."
            ),
            reasoning_yo=(
                "Àpapọ̀ ojú-ìwé = 250. Ojú-ìwé tí a ti kà = 30. "
                "Ojú-ìwé tó kù = Àpapọ̀ - Èyí tí a ti kà = 250 - 30 = 220."
            ),
            answer="220",
            translated_question=(
                "I have read 30 pages of a book that has 250 pages. "
                "How many pages do I still need to read to finish the book?"
            ),
        ),
        Exemplar(
            question=(
                "Ọkọ̀ kan gbe àwọn apoti 50. Àpoti kọọkan wọn 8 kilo. "
                "Ọkọ̀ naa yọ apoti 20 silẹ ni ibudo akọkọ. "
                "Kí ni ìwọ̀n tí ó ṣẹ́kù lórí ọkọ̀ naa ni kilo?"
            ),
            choices=None,
            reasoning_en=(
                "Initial weight = 50 boxes * 8 kg = 400 kg. "
                "Weight removed at first stop = 20 boxes * 8 kg = 160 kg. "
                "Remaining weight = 400 - 160 = 240 kg."
            ),
            reasoning_yo=(
                "Ìwọ̀n àkọ́kọ́ = àpótí 50 × 8 kg = 400 kg. "
                "Ìwọ̀n tí a yọ kúrò ní ibùdó àkọ́kọ́ = àpótí 20 × 8 kg = 160 kg. "
                "Ìwọ̀n tó kù = 400 - 160 = 240 kg."
            ),
            answer="240",
            translated_question=(
                "A truck carries 50 boxes. Each box weighs 8 kg. "
                "The truck drops 20 boxes at the first stop. "
                "What weight remains on the truck in kilograms?"
            ),
        ),
        Exemplar(
            question=(
                "Ade bi owo 2,500 naira lo ose. Ti o ba na 700 naira fun ounje "
                "ati 350 naira fun ọkọ, iye owo ni o ku fun awon nkan miiran?"
            ),
            choices=None,
            reasoning_en=(
                "Total allowance = 2500. Total spent on food and transport = 700 + 350 = 1050. "
                "Remaining money = Total - Spent = 2500 - 1050 = 1450."
            ),
            reasoning_yo=(
                "Àpapọ̀ owó = 2500. Owó tí a ná lórí oúnjẹ àti ọkọ̀ = 700 + 350 = 1050. "
                "Owó tó kù = Àpapọ̀ - Èyí tí a ná = 2500 - 1050 = 1450."
            ),
            answer="1450",
            translated_question=(
                "Ade received 2,500 naira for the week. If he spends 700 naira on food "
                "and 350 naira on transport, how much money remains for other things?"
            ),
        ),
    ],
    "math_en": [
        Exemplar(
            question="Tom has 5 bags of apples. Each bag contains 8 apples. He gives 12 apples to his friend. How many apples does Tom have left?",
            choices=None,
            reasoning_en=(
                "Start with 5 bags * 8 apples per bag = 40 total apples. "
                "He gives away 12 apples. Remaining apples = 40 - 12 = 28."
            ),
            reasoning_yo=(
                "Bẹ̀rẹ̀ pẹ̀lú àpò 5 × èso ápù 8 nínú ọ̀kọ̀ọ̀kan = 40 èso lápapọ̀. "
                "Ó fún ọ̀rẹ́ rẹ̀ ní èso 12. Èso tó kù = 40 - 12 = 28."
            ),
            answer="28",
            translated_question="Tom has 5 bags of apples. Each bag contains 8 apples. He gives 12 apples to his friend. How many apples does Tom have left?",
        ),
        Exemplar(
            question="Sarah reads 25 pages of a book on Monday and 30 pages on Tuesday. The book has 120 pages. How many pages does she still need to read?",
            choices=None,
            reasoning_en=(
                "Pages read so far = 25 + 30 = 55. "
                "Remaining pages = Total pages - Pages read = 120 - 55 = 65."
            ),
            reasoning_yo=(
                "Ojú-ìwé tí a ti kà = 25 + 30 = 55. "
                "Ojú-ìwé tó kù = Àpapọ̀ - Èyí tí a ti kà = 120 - 55 = 65."
            ),
            answer="65",
            translated_question="Sarah reads 25 pages of a book on Monday and 30 pages on Tuesday. The book has 120 pages. How many pages does she still need to read?",
        ),
        Exemplar(
            question="A baker makes 6 trays of cookies. Each tray holds 24 cookies. He sells 85 cookies. How many cookies are left?",
            choices=None,
            reasoning_en=(
                "Total cookies = 6 trays * 24 cookies per tray = 144. "
                "Cookies left = 144 - 85 = 59."
            ),
            reasoning_yo=(
                "Àpapọ̀ kúkì = àtẹ̀ 6 × kúkì 24 nínú ọ̀kọ̀ọ̀kan = 144. "
                "Kúkì tó kù = 144 - 85 = 59."
            ),
            answer="59",
            translated_question="A baker makes 6 trays of cookies. Each tray holds 24 cookies. He sells 85 cookies. How many cookies are left?",
        ),
        Exemplar(
            question="James bought a bike for $320. He had $280 saved and earned the rest by doing chores. If he did chores for 5 weeks earning the same amount each week, how much did he earn per week from chores?",
            choices=None,
            reasoning_en=(
                "Amount needed from chores = 320 - 280 = 40. "
                "Per week earnings = 40 / 5 = 8."
            ),
            reasoning_yo=(
                "Owó tó nílò láti ọwọ́ iṣẹ́ = 320 - 280 = 40. "
                "Owó ọ̀sọ̀ọ̀sẹ̀ = 40 / 5 = 8."
            ),
            answer="8",
            translated_question="James bought a bike for $320. He had $280 saved and earned the rest by doing chores. If he did chores for 5 weeks earning the same amount each week, how much did he earn per week from chores?",
        ),
    ],
    "qa": [
        Exemplar(
            question="Kínni iyì p nínú 24 = 2p?",
            choices=["p = 4", "p = 8", "p = 12", "p = 24"],
            reasoning_en=(
                "The equation is 24 = 2p. Divide both sides by 2: p = 12. This matches option C."
            ),
            reasoning_yo=(
                "Ìṣirò náà ni 24 = 2p. Pín ẹ̀gbẹ́ méjèèjì pẹ̀lú 2: p = 12. Èyí jẹ́ àṣàyàn C."
            ),
            answer="C",
            translated_question="What is the value of p in 24 = 2p?",
        ),
        Exemplar(
            question=(
                "Ms. Perez wa àpapọ̀ máílì 40 ní ọjọ́ 5. Ó wa iye máílì yìí kan náà "
                "ní ọjọ́ kọ̀ọ̀kan. Iye máílì mélòó ni Ms. Perez wà ní ọjọ́ kọ̀ọ̀kan?"
            ),
            choices=["5", "7", "8", "9"],
            reasoning_en=(
                "Total miles = 40, total days = 5. Same distance each day: 40 ÷ 5 = 8 miles per day. "
                "The answer is C."
            ),
            reasoning_yo=(
                "Àpapọ̀ máílì = 40, àpapọ̀ ọjọ́ = 5. Ó rìn iye kan náà lójúmọ́: 40 ÷ 5 = 8 máílì lójúmọ́. "
                "Ìdáhùn ni C."
            ),
            answer="C",
            translated_question=(
                "Ms. Perez drove a total of 40 miles in 5 days. She drove the same number of miles "
                "each day. How many miles did Ms. Perez drive each day?"
            ),
        ),
        Exemplar(
            question="Ṣàwarí òpó iye −40 ÷ (−8).",
            choices=["1 lórí 5", "-5", "−1 lórí 5", "5"],
            reasoning_en=(
                "−40 ÷ (−8) = 5. Negative divided by negative equals positive. "
                "40 ÷ 8 = 5, so the answer is D."
            ),
            reasoning_yo=(
                "−40 ÷ (−8) = 5. Àìdára pín sí àìdára jẹ́ dídára. "
                "40 ÷ 8 = 5, nítorí náà ìdáhùn ni D."
            ),
            answer="D",
            translated_question="Find the value of −40 ÷ (−8).",
        ),
        Exemplar(
            question="Báwo ni 1/4 + 2/4 ṣe dọ́gba?",
            choices=["1/4", "4/3", "3/4", "1/2"],
            reasoning_en=(
                "When adding fractions with the same denominator, add the numerators: "
                "1/4 + 2/4 = (1+2)/4 = 3/4. The answer is C."
            ),
            reasoning_yo=(
                "Nígbà tí a ń ṣàfikún ìdá pẹ̀lú oníṣirò abẹ́ kan náà, ṣàfikún àwọn oníṣirò òkè: "
                "1/4 + 2/4 = (1+2)/4 = 3/4. Ìdáhùn ni C."
            ),
            answer="C",
            translated_question="What is 1/4 + 2/4 equal to?",
        ),
    ],
    "reading_comprehension": [
        Exemplar(
            question=(
                "Àyọkà:\nỌ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́ ní ilé ẹ̀kọ́ alákọ̀ọ́bẹ̀rẹ̀. "
                "Ó ní ọmọ ọdún mẹ́tàlélógún (23) nínú kíláàsì rẹ̀.\n\n"
                "Ìbéèrè:\nKí ni iṣẹ́ Ọ̀jẹ̀ẹ́ Adéjùmọ́?"
            ),
            choices=["Oníṣẹ̀-ọ̀gbìn", "Olùkọ́", "Dókítà", "Awẹ̀"],
            reasoning_en=(
                "The passage states 'Ọ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́', meaning she teaches "
                "students. So she is a teacher, which is option B (Olùkọ́)."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Ọ̀jẹ̀ẹ́ Adéjùmọ́ máa ń kọ́ àwọn akẹ́kọ̀ọ́', èyí túmọ̀ sí pé ó ń kọ́ni. "
                "Nítorí náà ó jẹ́ olùkọ́, àṣàyàn B (Olùkọ́)."
            ),
            answer="B",
            translated_question=(
                "Passage:\nMrs. Adejumo teaches students at a primary school. "
                "She has 23 students in her class.\n\nQuestion:\nWhat is Mrs. Adejumo's job?"
            ),
        ),
        Exemplar(
            question=(
                "Àyọkà:\nỌjà ńlá kan wà ní àárín ìlú. Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá. "
                "Wọ́n máa ń ta ẹ̀fọ́, ẹja, àti ẹran.\n\n"
                "Ìbéèrè:\nỌjọ́ wo ni àwọn olùtajà máa ń kó ọjà wá?"
            ),
            choices=["Ọjọ́ Ajé", "Ọjọ́ Ìṣẹ́gun", "Ọjọ́bọ̀", "Ọjọ́ Ẹtì"],
            reasoning_en=(
                "The passage says 'Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá' "
                "(On Thursday, traders bring goods). The answer is C (Ọjọ́bọ̀)."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Lọ́jọ́ Ọjọ́bọ̀, àwọn olùtajà máa ń kó ọjà wá'. "
                "Nítorí náà ìdáhùn ni C (Ọjọ́bọ̀)."
            ),
            answer="C",
            translated_question=(
                "Passage:\nThere is a large market in the center of town. On Thursday, traders "
                "bring goods. They sell vegetables, fish, and meat.\n\n"
                "Question:\nOn which day do the traders bring goods?"
            ),
        ),
        Exemplar(
            question=(
                "Àyọkà:\nÓ jẹ́ ọmọ ọdún mẹ́wàá (10). Ó ní ìwé mẹ́rin: ìwé ìròyìn, ìwé àròsọ, "
                "ìwé ẹ̀kọ́ ìtàn, àti ìwé ẹ̀kọ́ ìmọ̀ ìjìnlẹ̀.\n\n"
                "Ìbéèrè:\nÌwé mélòó ni ọmọ náà ní?"
            ),
            choices=["Ìwé méjì", "Ìwé mẹ́rin", "Ìwé mẹ́fà", "Ìwé mẹ́jọ"],
            reasoning_en=(
                "The passage says 'Ó ní ìwé mẹ́rin' (He has 4 books), so the answer is B (Ìwé mẹ́rin)."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Ó ní ìwé mẹ́rin', nítorí náà ìdáhùn ni B (Ìwé mẹ́rin)."
            ),
            answer="B",
            translated_question=(
                "Passage:\nHe is 10 years old. He has four books: a newspaper, a storybook, "
                "a history textbook, and a science textbook.\n\n"
                "Question:\nHow many books does the child have?"
            ),
        ),
        Exemplar(
            question=(
                "Àyọkà:\nAdìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù. "
                "Ariwo rẹ̀ máa ń jí gbogbo agbárílé.\n\n"
                "Ìbéèrè:\nKí ni Adìẹ àgbààgbà máa ń ṣe ní àárọ̀ kùtùkùtù?"
            ),
            choices=["Ó máa ń sùn", "Ó máa ń pariwo", "Ó máa ń jẹun", "Ó máa ń lọ sí ọjà"],
            reasoning_en=(
                "The passage says 'Adìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù' "
                "(The old rooster crows early in the morning). The answer is B."
            ),
            reasoning_yo=(
                "Àyọkà sọ pé 'Adìẹ àgbààgbà máa ń pariwo ní àárọ̀ kùtùkùtù'. "
                "Nítorí náà ìdáhùn ni B."
            ),
            answer="B",
            translated_question=(
                "Passage:\nThe old rooster crows early in the morning. "
                "Its noise wakes the whole household.\n\n"
                "Question:\nWhat does the old rooster do early in the morning?"
            ),
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
    """Render a prompt for an E1/E2 strategy.

    BoN / TTC sampling reuses the same E1 prompt styles with ``n > 1`` and
    majority-vote selection. ``best_of_n_cot`` is kept as a legacy alias for
    English CoT.
    """
    if prompt_style in {"english_cot", "best_of_n_cot"}:
        return render_english_cot_prompt(example)
    if prompt_style == "yoruba_cot":
        return render_yoruba_cot_prompt(example)
    if prompt_style == "translate_pivot":
        return render_translate_pivot_prompt(example)
    raise ValueError(f"Unsupported prompt_style: {prompt_style!r}")


def render_exemplar_block(task: str, *, reasoning_mode: str) -> str:
    """Render few-shot demonstrations."""

    exemplars = TASK_EXEMPLARS.get(task)
    if not exemplars:
        return ""

    blocks = ["Examples:"]

    for ex in exemplars:
        parts = [
            f"Question:\n{ex.question}",
        ]

        if ex.choices:
            parts.append(
                "Choices:\n" + "\n".join(format_choices(ex.choices))
            )

        if reasoning_mode == "translate":
            parts.extend(
                [
                    f"English translation:\n{ex.translated_question}",
                    f"Reasoning:\n{ex.reasoning_en}",
                    f"Final answer: {ex.answer}",
                ]
            )
        else:
            reasoning = (
                ex.reasoning_yo
                if reasoning_mode == "yo"
                else ex.reasoning_en
            )

            parts.extend(
                [
                    f"Reasoning:\n{reasoning}",
                    f"Final answer: {ex.answer}",
                ]
            )

        blocks.append("\n\n".join(parts))

    blocks.append("Now answer the following question using the same format.")

    return "\n\n---\n\n".join(blocks) + "\n\n"

COMMON_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant.\n\n"
    "Solve the user's problem accurately.\n"
    "Return your response using the same format as the examples.\n"
    "Return the final answer in the requested format."
)


def render_english_cot_prompt(example: InferenceExample) -> PromptBundle:
    task_for_exemplars = (
        "math_en" if example.source_dataset == "afrimgsm_translate" else example.task
    )
    exemplar_block = render_exemplar_block(
        task_for_exemplars,
        reasoning_mode="en",
    )

    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            (
                "Instructions:\n"
                "- Follow the same format as the examples.\n"
                "- Reason step by step in English.\n"
                "- Use only the information provided in the question.\n"
                "- Write your reasoning after 'Reasoning:'.\n"
                "- Finish with exactly one line:\n"
                "Final answer: <answer>"
            ),
        ]
    )

    return PromptBundle(
        system=COMMON_SYSTEM_PROMPT,
        user=user,
    )

def render_yoruba_cot_prompt(example: InferenceExample) -> PromptBundle:
    exemplar_block = render_exemplar_block(
        example.task,
        reasoning_mode="yo",
    )

    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            (
                "Instructions:\n"
                "- Follow the same format as the examples.\n"
                "- Reason step by step in Yoruba.\n"
                "- Use only the information provided in the question.\n"
                "- Write your reasoning after 'Reasoning:'.\n"
                "- Finish with exactly one line:\n"
                "Final answer: <answer>"
            ),
        ]
    )

    return PromptBundle(
        system=COMMON_SYSTEM_PROMPT,
        user=user,
    )

def render_translate_pivot_prompt(example: InferenceExample) -> PromptBundle:
    exemplar_block = render_exemplar_block(
        example.task,
        reasoning_mode="translate",
    )

    user = "\n\n".join(
        [
            f"{exemplar_block}{render_problem_block(example)}",
            render_answer_format(example),
            (
                "Instructions:\n"
                "- Follow the same format as the examples.\n"
                "- Translate the question into English.\n"
                "- Solve the translated question by reasoning step by step in English.\n"
                "- Write your reasoning after 'Reasoning:'.\n"
                "- Finish with exactly one line:\n"
                "Final answer: <answer>"
            ),
        ]
    )

    return PromptBundle(
        system=COMMON_SYSTEM_PROMPT,
        user=user,
    )

def render_problem_block(
    example: InferenceExample,
    *,
    question_label: str = "Question",
) -> str:
    parts = [f"{question_label}:\n{example.question}"]

    if example.choices:
        parts.append(
            "Choices:\n" + "\n".join(format_choices(example.choices))
        )

    return "\n\n".join(parts)

def render_answer_format(example: InferenceExample) -> str:
    if example.answer_type == "choice":
        return (
            "Answer format:\n"
            "Final answer: <option letter>"
        )

    if example.answer_type == "number":
        return (
            "Answer format:\n"
            "Final answer: <number>"
        )

    return (
        "Answer format:\n"
        "Final answer: <concise Yoruba answer>"
    )

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
