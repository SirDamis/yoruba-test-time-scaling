# Yoruba Test-Time Compute Scaling

This repository contains the code, configs, and analysis scripts for evaluating **test-time compute (TTC) scaling for Yoruba reasoning**. The central question is whether spending more inference compute — more samples, self-consistency, best-of-N — can compensate for the weak Yoruba representations of current open models.

We evaluate on **AfriMGSM** (grade-school math) and **AfriMMLU** (multiple-choice QA) in Yoruba, with Hausa, Igbo, Swahili, Amharic and English arms for cross-lingual comparison. English is also used *inside* two of the reasoning strategies (English CoT and translate-to-English pivoting) to test whether the bottleneck is language understanding or reasoning itself.

All inference runs against an OpenAI-compatible model endpoint: a local **vLLM** server, **OpenRouter**, or **Ramp Router**.

## Research questions

| RQ | Question |
|----|----------|
| **RQ1** | Which reasoning language works best for Yoruba tasks (Yoruba CoT, English CoT, or translate-to-English pivot)? |
| **RQ2** | Does TTC improve performance on a low-resource language like Yoruba? |
| **RQ3** | Where is the bottleneck: candidate generation or candidate selection? |
| **RQ4** | Can a small model plus TTC match a larger model without TTC? |

## Setup

```bash
uv lock && uv venv
uv pip install -r requirements.txt
```

Local serving requires vLLM on the GPU machine (installed separately from the pinned requirements):

```bash
uv pip install vllm
```

### Model endpoints

Pick whichever endpoint you are running against and set its environment variables. A root `.env` is loaded automatically, so the keys can live there.

```bash
# Local vLLM (serve one model at a time; see "Local vLLM" below)
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."

# Ramp Router
export ROUTER_KEY="..."
```

## Data

All experiments use test splits only. Download and normalize the benchmarks first:

```bash
uv run python scripts/download_hf_datasets.py --dataset all
# or a subset / language filter
uv run python scripts/download_hf_datasets.py --dataset afrimgsm --language yor,hau
```

Normalized JSONL rows:
`{"answer_type": "choice|number", "choices": [...] | null, "gold_answer": "...", "question": "..."}`.

| Dataset | Task | Languages | Rows | Path |
|---------|------|-----------|------|------|
| `afrimgsm_*` | Math | yor, hau, ibo, swa, amh, eng | 250 | `data/normalized/math-reasoning/<name>/test.jsonl` |
| `afrimmlu_*` | QA | yor, hau, ibo, swa, amh, eng | 500 | `data/normalized/question-answering/<name>/test.jsonl` |
| `afrimgsm_translate`, `afrimmlu_translate` | Math / QA | English input | 250 / 500 | per-language path |
| `afriqa` | QA | Yoruba | 326 | `data/normalized/question-answering/afriqa/test.jsonl` |
| `naijarc` | Reading comprehension | Yoruba | 191 | `data/normalized/question-answering/naijarc/test.jsonl` |

The experiment configs currently cover the `afrimgsm_*` and `afrimmlu_*` families (E0 is English-only; the E2 OpenRouter config is Yoruba-only). `afriqa` and `naijarc` are downloaded but not yet attached to a config.

## Models

Model `name` is what you pass to `--models` and must match the config, the vLLM served id, or the router slug.

| Config `name` | vLLM | OpenRouter | Ramp Router |
|---------------|------|------------|-------------|
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | — | `qwen/qwen3.5-4b` |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | `qwen/qwen3.5-9b` | `qwen/qwen3.5-9b` |
| `qwen3-14b` | `Qwen/Qwen3-14B` | — | — |
| `qwen3-32b` | `Qwen/Qwen3-32B` | — | — |
| `gemma3-4b` | `google/gemma-3-4b-it` | `google/gemma-3-4b-it` | `google/gemma-3-4b-it` |
| `gemma3-12b`, `gemma3-27b` | `google/gemma-3-12b-it`, `google/gemma-3-27b-it` | — | — |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/llama-3.2-3b-instruct` | `meta-llama/llama-3.2-3b-instruct` |
| `llama3.2-11b-vision` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | — | — |
| `deepseek-v4-flash` | — | `deepseek-v4-flash-0731` | `accounts/fireworks/models/deepseek-v4-flash-0731` |
| `deepseek-v4.1-flash` | — | `deepseek/deepseek-v4.1-flash` | `accounts/fireworks/models/deepseek-v4.1-flash` |

Qwen3.5 (`Qwen3_5ForConditionalGeneration`) needs a vLLM build whose model registry includes that architecture; verify with `vllm serve Qwen/Qwen3.5-9B`. It is not offered on OpenRouter at 4B.

## Experiments

Run order is **E0 → E1 → E2 → E3 → E4**. Each experiment is provided as one config per endpoint, e.g. `configs/e1_reasoning_language_{vllm,openrouter,ramp_router}.json`.

**Design note.** Every arm is *prompted* chain-of-thought, so each model's native thinking mode is disabled to keep the comparison controlled — Qwen3/Qwen3.5 via `enable_thinking=false`, OpenRouter via `reasoning.enabled=false`, Ramp Router via `reasoning_effort=none`. E1 has four reasoning strategies: `yoruba_cot`, `english_cot`, `translate_pivot`, and `direct` (no-CoT control). Only `translate_pivot` demonstrates an explicit translation step; the other arms reason directly in their target language.

### E0 — English baseline (multilingual reasoning gap)

English question → English CoT → answer on the native `eng` splits. Compared against each language's `english_cot` rows from E1 to measure the reasoning gap.

```bash
uv run python scripts/run_inference.py --config configs/e0_english_baseline_vllm.json
uv run python scripts/run_inference.py --config configs/e0_english_baseline_openrouter.json
uv run python scripts/run_inference.py --config configs/e0_english_baseline_ramp_router.json
```

### E1 — Reasoning language (greedy N=1)

Which strategy maximizes accuracy: `yoruba_cot`, `english_cot`, `translate_pivot`, or `direct`?

```bash
uv run python scripts/run_inference.py --config configs/e1_reasoning_language_vllm.json
uv run python scripts/run_inference.py --config configs/e1_reasoning_language_openrouter.json
uv run python scripts/run_inference.py --config configs/e1_reasoning_language_ramp_router.json
```

### E2 — TTC scaling

The winning E1 strategy (currently `english_cot_ttc`) is expanded to `_n1..._n64`: N ∈ {1, 4, 8, 16, 32, 64} with `nested_n: true`. N=1 is a true greedy decode (temp 0, `selection=first`); N≥4 are nested prefixes of one stochastic pool sampled at max N (temp 0.7, top-p 0.95, `max_tokens=4096`).

```bash
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling_vllm.json
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling_openrouter.json

uv run python scripts/aggregate_ttc_metrics.py --runs-dir runs --run-id e2_ttc_scaling_vllm
```

Outputs land in `results/ttc_scaling/`: metrics JSON/CSV, `pool_estimates.{json,csv}` (unbiased `pass@k` + random-subset `maj@k` + diversity), `accuracy_vs_n.png`, `accuracy_vs_tokens.png`. The aggregate warns when any condition exceeds 2% truncation.

Report the pool `maj@k` (random k-subsets) as the self-consistency metric — the prefix-based `select@N` in the main table is order-sensitive for majority vote.

### E3 — Generation vs selection

Reuses E2 traces. Per N: `pass@N` (generation ceiling) versus `first` / `majority_vote` selection. High pass@N with low select@N indicates a selection bottleneck; low pass@N indicates a generation bottleneck.

```bash
uv run python scripts/reselect_candidates.py \
  --run-id e2_ttc_scaling_vllm --strategies first,majority_vote
```

Writes `results/e3_reselection/<run_id>/` (`selections_*.jsonl`, `e3_report.json`).

### E4 — Small model + TTC vs large greedy (offline)

Can Qwen 4B with TTC match Qwen 32B greedy (N=1)? This stage only reads saved E2 metrics; no model is loaded.

```bash
uv run python scripts/compare_e4.py --runs-dir runs --run-id e2_ttc_scaling_vllm
# or: uv run python scripts/compare_e4.py --metrics-json results/ttc_scaling/metrics.json
```

Writes `results/e4_comparison/` (JSON, CSV table, plots).

## Local vLLM

Serve one model at a time in a long-lived process (use `tmux` so an SSH drop does not kill it):

```bash
chmod +x scripts/serve_vllm.sh   # once
./scripts/serve_vllm.sh Qwen/Qwen3.5-9B
# equivalent:
# vllm serve Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 \
#   --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.90

curl -s http://localhost:8000/v1/models | head   # health check
```

The served id must equal the config's `model` field (see the [Models](#models) table). Gated checkpoints (`gemma-3-*`, `Llama-3.2-*`) need `export HF_TOKEN="hf_..."` for the download. Resume a crashed run by re-running without `--overwrite`.

## Inference runner

Entrypoint: `scripts/run_inference.py`.

| Flag | Meaning |
|------|---------|
| `--config PATH` | Inference config JSON |
| `--run-id ID` | Override the config `run_id` (names the `runs/<id>/` directory) |
| `--output-dir DIR` | Override the output root (default `runs`) |
| `--datasets NAMES` | Comma-separated **exact** dataset names (e.g. `afrimgsm_yor`) |
| `--language CODES` | Keep datasets whose name ends in `_<code>` (e.g. `yor`, `yor,hau`; `all` disables) |
| `--models NAMES` | Comma-separated **exact** model names (e.g. `qwen3.5-9b`) |
| `--methods NAMES` | Comma-separated **exact** method names (E2: `english_cot_ttc_n4`, ...) |
| `--limit N` | Per-dataset example cap (smoke runs) |
| `--resume` / `--no-resume` | Resume from checkpoint (default `--resume`) |
| `--overwrite` | Delete prior artifacts for this `run_id` and start clean |
| `--max-concurrent N` | Override in-flight generations |
| `--progress` / `--no-progress` | Per-example progress to stderr (default on) |

Example: E1 with Qwen3.5-9B on Yoruba AfriMGSM via OpenRouter.

```bash
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_openrouter.json \
  --models qwen3.5-9b \
  --datasets afrimgsm_yor \
  --run-id e1_afrimgsm_yor_qwen3.5-9b_openrouter \
  --overwrite
```

Each run writes under `runs/<run_id>/`:

| Artifact | Contents |
|----------|----------|
| `candidates.jsonl` | Every sampled candidate (incl. `finish_reason`) |
| `selections.jsonl` | One selected answer per condition × example |
| `manifest.json` | Run metadata and counts |
| `completed_units.jsonl` | Resume checkpoint |

## Evaluation and analysis

```bash
uv run python scripts/evaluate_runs.py                     # pass@N / select@N for all runs
uv run python scripts/evaluate_runs.py --run-id <run_id>   # one run
```

- **pass@N** — any of N candidates correct (generation ceiling)
- **select@N** — selected answer correct
- **Trunc%** — fraction of generations that hit the token limit (reported by `aggregate_ttc_metrics.py`; check it before trusting scaling curves)

Additional analysis: `scripts/e1_accuracy.py` prints a per-condition accuracy table for one run, and `scripts/estimate_pool_ttc.py` computes pool-based `pass@k` / `maj@k` / diversity from nested E2 pools.

## Repository layout

```text
configs/           Experiment configs, one per endpoint (<exp>_<endpoint>.json)
data/normalized/   Yoruba JSONL datasets (gitignored; download locally)
scripts/           CLI entrypoints
src/ttcs_yoruba/   Library code (prompting, backends, inference, metrics, selection)
tests/             Unit tests (.venv/bin/python -m pytest tests -q)
runs/              Inference artifacts (gitignored)
results/           Evaluation outputs (gitignored)
```

## Scripts

| Script | Role |
|--------|------|
| `scripts/download_hf_datasets.py` | Download / normalize benchmarks |
| `scripts/run_inference.py` | E0/E1/E2 inference |
| `scripts/serve_vllm.sh` | Start a local vLLM OpenAI-compatible server |
| `scripts/evaluate_runs.py` | pass@N / select@N summary |
| `scripts/e1_accuracy.py` | Per-condition accuracy table for one run |
| `scripts/aggregate_ttc_metrics.py` | E2 metrics + plots |
| `scripts/estimate_pool_ttc.py` | Pool-based pass@k / maj@k / diversity |
| `scripts/reselect_candidates.py` | E3 offline re-selection |
| `scripts/compare_e4.py` | E4 small+TTC vs large greedy |
| `scripts/extract_truncated.py` | Pull truncated generations for inspection |
| `scripts/show_exemplar_prompts.py` | Dump sample prompts per style |
