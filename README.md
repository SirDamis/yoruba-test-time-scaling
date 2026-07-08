# Yoruba Test-Time Compute Scaling

This repository scaffolds the reproducible experiment pipeline for **When More Thinking Is Not Enough: Test-Time Compute Scaling for Yoruba Reasoning**.

The project evaluates **Yoruba tasks only**. English appears only inside inference interventions such as English chain-of-thought, translate-to-English pivoting, or verifier prompts. There are no standalone English benchmark runs in Paper 1.

## What Is Implemented

- Compact Yoruba JSONL benchmark exports.
- Dataset validation and generic raw-to-normalized adapter tooling.
- Prompt templates for direct answering, Yoruba CoT, English CoT, translation pivoting, mixed reasoning, and budget forcing.
- Config-driven experiment runner with mock, Hugging Face, and OpenAI-compatible/vLLM-ready backends.
- Test-time compute methods: greedy, self-consistency, Best-of-N answer verifier, Best-of-N trace verifier, and oracle/pass@N analysis.
- Evaluation metrics: accuracy, exact/F1, pass@N, select@N, language compliance, token counts, estimated cost, and tokens per correct answer.
- Analysis scripts for aggregation, validation sampling, and paper-ready plots.
- A five-example Yoruba smoke dataset that runs without GPU dependencies.



## Repository Layout

```text
configs/                  Experiment configs
data/normalized/          Normalized Yoruba JSONL datasets
data/raw/                 Local raw datasets, ignored by git
data/manual_validation/   Human review templates and sampled review files
docs/                     Protocol and dataset notes
scripts/                  CLI entrypoints
src/ttcs_yoruba/          Project core 
tests/                    Unit 
```

## Setup

This project is configured for `uv`:

```bash
uv lock
uv run python scripts/download_hf_datasets.py --dataset all
```

The default downloader uses registered Hugging Face/GitHub file URLs and does not require the Hugging Face `datasets` package. To force the `datasets` backend for a dataset, run:

```bash
uv run --with datasets python scripts/download_hf_datasets.py --dataset afrimgsm --backend datasets
```

## Downloaded Dataset Schema

Downloaded JSONL rows retain only the fields needed for evaluation:

```json
{
  "answer_type": "choice|number|text|freeform|instruction",
  "choices": ["A. ...", "B. ..."],
  "gold_answer": "B",
  "question": "Yoruba prompt",
}
```

`choices` is `null` for datasets that do not provide multiple-choice options. The internal normalizer still validates Yoruba-only rows before writing the compact export.

## Main Experiment Config

`configs/main_yoruba_ttc.json` defines the intended full study:

- Yoruba-only benchmark path.
- Open-model backend placeholders for Qwen, Llama, Gemma, and optional African-adapted models.
- TTC sweeps over `N = 1, 4, 8, 16, 32, 64`.
- Reasoning-language interventions and budget-forcing variants.
The smoke config uses the deterministic mock backend so the pipeline can be verified before cloud runs.
