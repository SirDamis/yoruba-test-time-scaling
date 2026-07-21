"""Dump sample E1 prompts into per-task folders, one file per CoT style.

Layout::

    runs/sample_prompts/
      math/
        english_cot.txt
        yoruba_cot.txt
        translate_pivot.txt
        best_of_n_cot.txt
      qa/
        ...
      reading_comprehension/
        ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttcs_yoruba.examples import InferenceExample
from ttcs_yoruba.prompting import render_prompt


EXAMPLES: list[InferenceExample] = [
    InferenceExample(
        id="math_001",
        task="math",
        question=(
            "Ibéèrè: Roger ní bọ́ọ́lù aláfajọ̀ 5. Ó ra agolo bọ́ọ́lù aláfajọ̀gbá 2 kún-un. "
            "Agolo kọ̀ọ̀kan ní bọ́ọ́lù aláfajọ̀gbá 3. Bọ́ọ́lù aláfajọ̀gbá mélòó ni ó ní báyìí?"
        ),
        choices=None,
        gold_answer="11",
        answer_type="number",
        source_dataset="afrimgsm",
    ),
    InferenceExample(
        id="qa_001",
        task="qa",
        question="Kínni iyì p nínú 24 = 2p?",
        choices=["p = 4", "p = 8", "p = 12", "p = 24"],
        gold_answer="C",
        answer_type="choice",
        source_dataset="afrimmlu",
    ),
    InferenceExample(
        id="rc_001",
        task="reading_comprehension",
        question=(
            "Àyọkà:\nAjírọ́lá àti àwọn àbúrò rẹ̀ pa ẹnu pò láti sé ọjọ́ ìbí fún bàbá wọn.\n\n"
            "Ìbéèrè:\nTa ni wọ́n ṣe ayẹyẹ fún?"
        ),
        choices=["Ajírọ́lá", "Àbúrò", "Bàbá", "Ìyá"],
        gold_answer="C",
        answer_type="choice",
        source_dataset="naijarc",
    ),
]

STYLES = [
    ("english_cot", "E1: Chain-of-Thought (English)"),
    ("yoruba_cot", "E1: Chain-of-Thought (Yoruba)"),
    ("translate_pivot", "E1: Translate Pivot (Yo→En reason)"),
    ("best_of_n_cot", "Best-of-N CoT (English)"),
]

OUTPUT_DIR = Path("runs") / "sample_prompts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _example_header(example: InferenceExample) -> list[str]:
    lines = [
        f"TASK: {example.task}",
        f"EXAMPLE_ID: {example.id}",
        f"SOURCE: {example.source_dataset}",
        f"QUESTION: {example.question}",
        f"GOLD ANSWER: {example.gold_answer}",
        f"ANSWER TYPE: {example.answer_type}",
    ]
    if example.choices:
        lines.append(f"CHOICES: {example.choices}")
    lines.append("=" * 72)
    lines.append("")
    return lines


def main() -> None:
    for example in EXAMPLES:
        task_dir = OUTPUT_DIR / example.task
        task_dir.mkdir(parents=True, exist_ok=True)
        header = _example_header(example)

        for style_key, style_label in STYLES:
            prompt = render_prompt(example, style_key)
            lines = list(header)
            lines.append(f"PROMPT STYLE: {style_key}")
            lines.append(f"LABEL: {style_label}")
            lines.append("-" * 72)
            lines.append("")
            lines.append(prompt.as_text())
            lines.append("")

            out_path = task_dir / f"{style_key}.txt"
            out_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Written {out_path}")


if __name__ == "__main__":
    main()
