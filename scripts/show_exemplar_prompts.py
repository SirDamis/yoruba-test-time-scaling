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
        question="Ibéèrè: Roger ní bọ́ọ́lù aláfajọ̀ 5. Ó ra agolo bọ́ọ́lù aláfajọ̀gbá 2 kún-un. Agolo kọ̀ọ̀kan ní bọ́ọ́lù aláfajọ̀gbá 3. Bọ́ọ́lù aláfajọ̀gbá mélòó ni ó ní báyìí?",
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
        question="Àyọkà:\nAjírọ́lá àti àwọn àbúrò rẹ̀ pa ẹnu pò láti sé ọjọ́ ìbí fún bàbá wọn.\n\nÌbéèrè:\nTa ni wọ́n ṣe ayẹyẹ fún?",
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

for example in EXAMPLES:
    task = example.task
    file_path = OUTPUT_DIR / f"{task}.txt"
    lines: list[str] = []
    lines.append(f"TASK: {task}")
    lines.append(f"QUESTION: {example.question}")
    lines.append(f"GOLD ANSWER: {example.gold_answer}")
    lines.append(f"ANSWER TYPE: {example.answer_type}")
    if example.choices:
        lines.append(f"CHOICES: {example.choices}")
    lines.append("=" * 72)
    lines.append("")
    for style_key, style_label in STYLES:
        prompt = render_prompt(example, style_key)
        lines.append(f"--- {style_label} ---")
        lines.append(prompt.as_text())
        lines.append("")
    file_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {len(lines)} lines to {file_path}")
