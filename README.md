# Yoruba Test-Time Compute Scaling

Reproducible experiment pipeline for **When Is More Thinking Enough? Evaluation of Test-Time Compute Scaling for Yoruba Language Reasoning**.

**Core idea:** Can test-time compute (TTC) scaling compensate for weak low-resource language representations?

The project evaluates **Yoruba tasks only**. English appears only inside inference interventions (English chain-of-thought, translate-to-English pivoting). There are no standalone English benchmark runs.

## Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Which reasoning language works best for Yoruba tasks? |
| **RQ2** | Does TTC improve language model performance on low-resource languages like Yoruba? |
| **RQ3** | Where is the bottleneck: candidate generation or candidate selection? |
| **RQ4** | Can small models plus TTC match larger models without TTC? |

## Experiment Plan

### E1: Reasoning Language

**Question:** Which reasoning strategy maximizes performance on Yoruba tasks?

| Field | Value |
|-------|--------|
| **Models** | Qwen 4B, Gemma 4B, Llama 3B |
| **Strategies** | Yoruba CoT, English CoT, Translate Pivot |
| **Run count** | 3 models × 3 strategies × dataset sample size |

| Strategy | Pipeline |
|----------|----------|
| **Yoruba CoT** | Yoruba question → Yoruba reasoning → Yoruba answer |
| **English CoT** | Yoruba question → English reasoning → Yoruba answer |
| **Translate Pivot** | Translate question → English reasoning → Yoruba answer |

The best-performing reasoning strategy from E1 is fixed for later experiments.

**Config:** `configs/e1_reasoning_language.json` (3 models × 3 strategies, greedy `N=1`).

```bash
# Full E1 sweep
uv run python scripts/run_inference.py --config configs/e1_reasoning_language.json

# Smoke test
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods translate_pivot \
  --limit 5
```

### E2: TTC Scaling

**Question:** How does test-time compute scale on Yoruba tasks?

| Field | Value |
|-------|--------|
| **Models** | Qwen 4B, 14B, 32B; optional: Gemma 4B, 12B, 27B; Llama 3B, 11B |
| **Reasoning** | Best strategy from E1 (set `prompt_style` in the E2 config) |
| **N sweep** | 1, 4, 8, 16, 32, 64 — **nested** by default (`nested_n: true`) |
| **Compute** | **True greedy** for *N*=1 (`greedy_n1`); sample **once** at max *N* (64) for TTC and score prefixes for each *k* ≥ 4 |
| **Run count** | models × (1 greedy + max_N pool) × dataset sample size (generations); metrics still reported per *k* |

**Configs**

- Primary: `configs/e2_ttc_scaling.json` (Qwen 4B / 14B / 32B)
- Optional families: `configs/e2_ttc_scaling_optional.json`

After E1, change `methods[0].prompt_style` to the winning style (`english_cot`, `yoruba_cot`, or `translate_pivot`). BoN reuses that same prompt for every sample.

```bash
# Full E2 (Qwen ladder)
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling.json

# Smoke test
uv run python scripts/run_inference.py \
  --config configs/e2_ttc_scaling.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods english_cot_ttc_n4 \
  --limit 5

# Aggregate accuracy / tokens / latency + plots
uv run python scripts/aggregate_ttc_metrics.py --runs-dir runs
# optional: uv pip install matplotlib
```

**Metrics** (written to `results/ttc_scaling/`)

- Accuracy (select@N), pass@N, gap
- Token usage (total and mean per example)
- Inference latency (total and mean per example)
- Tokens per correct answer

**Plots**

- `accuracy_vs_n.png` — Accuracy vs *N*
- `accuracy_vs_tokens.png` — Accuracy vs total tokens
- Per-dataset variants when multiple datasets are present

### E3: Generation vs Selection

**Question:** Where is the primary bottleneck for Yoruba reasoning — candidate generation or candidate selection?

Reuse E2 candidate traces **without regenerating**. For each *N*:

| Axis | Metrics |
|------|---------|
| **Generation quality** | `pass@N` (any of *N* samples is correct) |
| **Selection quality** | `first`, `majority_vote` |

**Interpretation**

- High `pass@N`, low selected accuracy → selection is the bottleneck
- Low `pass@N` → candidate generation is the bottleneck

`pass@N` is the selection ceiling (no gold-label selector). LLM-as-judge / reward-model verifiers are deferred.

```bash
# Offline re-selection (first + majority vote)
uv run python scripts/reselect_candidates.py \
  --run-id e2_ttc_scaling \
  --strategies first,majority_vote
```

Outputs under `results/e3_reselection/<run_id>/`:

- `selections_<strategy>.jsonl` — one selected answer per example per strategy
- `e3_report.json` — pass@N vs select@N and gap per condition

### E4: Small Model + TTC vs Larger Model

**Question:** Can a small model with TTC match a larger model without TTC?

Primary comparison (from E2 Qwen runs):

- **Qwen 4B** with sweeping *N* (BoN / self-consistency)
- vs **Qwen 32B** greedy (`N=1`)
- Optional ladder context: Qwen 14B greedy (`configs/e4_comparison.json`)

**Config:** `configs/e4_comparison.json`

```bash
# From aggregated E2 metrics
uv run python scripts/compare_e4.py \
  --metrics-json results/ttc_scaling/metrics.json

# Or directly from run artifacts
uv run python scripts/compare_e4.py --runs-dir runs --run-id e2_ttc_scaling

# Override models
uv run python scripts/compare_e4.py \
  --metrics-json results/ttc_scaling/metrics.json \
  --small-model qwen3-4b \
  --large-model qwen3-32b \
  --large-n 1
```

**Outputs** under `results/e4_comparison/`:

| File | Contents |
|------|----------|
| `e4_comparison.json` | Full comparison + headline (min *N* to match large) |
| `e4_paper_table.csv` | Flat table: accuracy, Δacc, tokens/correct, latency, ratios |
| `e4_accuracy_vs_n.png` | Small accuracy vs *N* with large greedy reference line |
| `e4_accuracy_vs_tokens.png` | Accuracy vs mean tokens/example |
| `e4_accuracy_vs_latency.png` | Accuracy vs mean latency/example |

**Metrics**

- Accuracy delta (small@*N* − large greedy)
- Whether small matches/beats large
- Tokens per correct, latency per correct
- Accuracy per 1k tokens (and per cost if `cost_per_1k_tokens` is set)
- Token/latency ratios vs large baseline
- Per-dataset **min *N*** where small accuracy ≥ large greedy

### Ablations

| Ablation | Design |
|----------|--------|
| **Diacritics** | 100 samples with vs without diacritic marks under TTC |
| **Adapted model** | AfriqueQwen vs same-size base Qwen |
| **Zero-shot CoT** | Smallest and largest models; compare to greedy for the same family at *N* = 1 (optional *N* = 4, 8, 16) |
| **Few-shot language** | Yoruba few-shot exemplars vs English few-shot exemplars |

## What Is Implemented

- Compact Yoruba JSONL benchmark exports.
- Dataset validation and generic raw-to-normalized adapter tooling.
- Prompt templates for E1 reasoning strategies: Yoruba CoT, English CoT, and Translate Pivot.
- Language-matched few-shot exemplars (Yoruba reasoning for Yoruba CoT; English + translation demos for pivot).
- Config-driven cloud inference runner with a Hugging Face Transformers backend by default.
- Secondary OpenAI-compatible / vLLM-ready chat endpoint backend.
- Best-of-N / TTC sampling over any E1 prompt style with majority-vote selection (`N = 1,4,8,16,32,64`), with optional **nested** sampling (`nested_n`): true greedy *N*=1 plus one stochastic pool at max *N* for prefixes.
- Candidate and selected-answer JSONL artifacts with exact-match correctness flags.
- `pass@N` vs `select@N` evaluation for generation-vs-selection analysis.
- E2 metrics aggregation (accuracy, tokens, latency) and scaling plots.
- E3 offline re-selection (`first`, `majority_vote`) and pass@N vs select@N reports.
- Sample prompt dumps for generation styles.
- E4 small+TTC vs large greedy comparison tables and efficiency plots.

## Repository Layout

```text
configs/                  Experiment configs
data/normalized/          Normalized Yoruba JSONL datasets
data/raw/                 Local raw datasets, ignored by git
data/manual_validation/   Human review templates and sampled review files
docs/                     Protocol and dataset notes
scripts/                  CLI entrypoints
src/ttcs_yoruba/          Project core
tests/                    Unit tests
runs/                     Inference artifacts (gitignored)
results/                  Evaluation outputs
```

## Setup

This project is configured for `uv`:

```bash
uv lock
uv venv
uv pip install -r requirements.txt
uv run python scripts/download_hf_datasets.py --dataset all
```

The default downloader uses registered Hugging Face / GitHub file URLs and does not require the Hugging Face `datasets` package. To force the `datasets` backend for a dataset:

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

`choices` is `null` for datasets that do not provide multiple-choice options. The internal normalizer validates Yoruba-only rows before writing the compact export.

## Inference Pipeline

`configs/inference.json` defines a cloud-ready inference sweep:

- Yoruba-only datasets from the compact JSONL downloads.
- Hugging Face `transformers` as the main backend.
- Methods include English CoT, Yoruba CoT, and English Best-of-N CoT.
- Best-of-N uses majority-vote selection over a configurable *N* sweep.
- Model entries across size labels for small-vs-large comparisons (E4).

On the cloud machine:

```bash
uv venv
uv pip install -r requirements.txt
uv run python scripts/download_hf_datasets.py --dataset all
```

Then run the Transformers-backed sweep. Set `HF_TOKEN` if a gated model (Llama, Gemma) needs authentication:

```bash
export HF_TOKEN="..."
uv run python scripts/run_inference.py --config configs/inference.json
```

**Resume / checkpoint (default on):** progress is stored in `runs/<run_id>/completed_units.jsonl`. Re-running the same `run_id` skips finished units and appends. Use `--overwrite` to delete prior artifacts and start clean, or `--no-resume` to rewrite outputs without using the checkpoint.

```bash
# Continue a crashed / partial run (default)
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling.json --run-id e2_ttc_scaling

# Start clean for that run_id
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling.json --run-id e2_ttc_scaling --overwrite
```

Smoke-run filters:

```bash
uv run python scripts/run_inference.py \
  --config configs/inference.json \
  --datasets afrimgsm \
  --models qwen2.5-7b \
  --methods best_of_n_cot_n4 \
  --limit 5
```

OpenAI-compatible / vLLM path (`configs/openai_compatible_inference.json`):

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
uv run python scripts/run_inference.py --config configs/openai_compatible_inference.json
```

Each run writes:

| Artifact | Contents |
|----------|----------|
| `runs/<run_id>/candidates.jsonl` | Every sampled candidate response |
| `runs/<run_id>/selections.jsonl` | One selected answer per dataset / model / method / example |
| `runs/<run_id>/manifest.json` | Run metadata and row counts |

`runs/` is gitignored.

## Evaluation Pipeline

```bash
# Evaluate all runs (writes per-run JSON to results/pass_vs_select/)
uv run python scripts/evaluate_runs.py

# Evaluate a specific run
uv run python scripts/evaluate_runs.py --run-id 00000

# Write aggregated results into a single file
uv run python scripts/evaluate_runs.py --run-id 00000 --output results/aggregated.json
```

The script reads `candidates.jsonl` and `selections.jsonl` and computes (E3):

- **pass@N** — whether *any* of the *N* sampled candidates is correct (generation quality / selection ceiling).
- **select@N** — whether the selection method (e.g. majority vote) picked the correct answer.
- **gap** — `pass@N − select@N`: correct answer generated but selector missed it.
