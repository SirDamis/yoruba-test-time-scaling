# Yoruba Test-Time Compute Scaling

This repository scaffolds the reproducible experiment pipeline for **When More Thinking Is Not Enough: Test-Time Compute Scaling for Yoruba Reasoning**.

The project evaluates **Yoruba tasks only**. English appears only inside inference interventions such as English chain-of-thought, translate-to-English pivoting, or verifier prompts. There are no standalone English benchmark runs in Paper 1.

## What Is Implemented

- Compact Yoruba JSONL benchmark exports.
- Dataset validation and generic raw-to-normalized adapter tooling.
- Prompt templates for Yoruba CoT and English CoT on Yoruba tasks.
- Config-driven cloud inference runner with a Hugging Face Transformers backend by default.
- Secondary OpenAI-compatible/vLLM-ready chat endpoint backend.
- Best-of-N candidate sampling with majority-vote selection.
- Candidate and selected-answer JSONL artifacts with exact-match correctness flags.



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
uv venv
uv pip install -r requirements.txt
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
  "question": "Yoruba prompt"
}
```

`choices` is `null` for datasets that do not provide multiple-choice options. The internal normalizer still validates Yoruba-only rows before writing the compact export.

## Cloud Inference Pipeline

`configs/inference.json` defines the first cloud-ready inference sweep:

- Yoruba-only datasets from the compact JSONL downloads.
- Hugging Face `transformers` as the main backend.
- Three inference options: English CoT, Yoruba CoT, and English Best-of-N CoT.
- Best-of-N CoT uses English reasoning with majority-vote selection over `N = 4, 6, 8, 10, 12, 16, 24, 32`.
- Model entries across size labels so small, medium, and larger models can be compared.

On the cloud machine, first install dependencies and create the normalized data files:

```bash
uv venv
uv pip install -r requirements.txt
uv run python scripts/download_hf_datasets.py --dataset all
```

Then run the Transformers-backed sweep. Set `HF_TOKEN` if a gated model such as Llama or Gemma needs authentication.

```bash
export HF_TOKEN="..."
uv run python scripts/run_inference.py --config configs/inference.json
```

Useful cloud smoke-run filters:

```bash
uv run python scripts/run_inference.py \
  --config configs/inference.json \
  --datasets afrimgsm \
  --models qwen2.5-7b \
  --methods best_of_n_cot_n4 \
  --limit 5
```

The secondary backend is OpenAI-compatible chat completions, for example a vLLM server. Use `configs/openai_compatible_inference.json` for that path:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
uv run python scripts/run_inference.py --config configs/openai_compatible_inference.json
```

Each run writes:

- `runs/<run_id>/candidates.jsonl`: every sampled candidate response.
- `runs/<run_id>/selections.jsonl`: one selected answer per dataset/model/method/example.
- `runs/<run_id>/manifest.json`: run metadata and row counts.

The runner keeps all experiment artifacts in `runs/`, which is ignored by git.
