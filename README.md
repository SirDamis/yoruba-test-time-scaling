# Yoruba Test-Time Compute Scaling

Experiment pipeline for **When Is More Thinking Enough? Evaluation of Test-Time Compute Scaling for Yoruba Language Reasoning**.

**Core idea:** Can test-time compute (TTC) scaling compensate for weak low-resource language representations?

Evaluation is **Yoruba-only** (AfriMGSM + AfriMMLU test splits). English appears only as an intervention (English CoT, translate-to-English pivot) and via the English-input `*_translate` variants used for baseline comparison. Local vLLM setup: **[README_VLLM.md](README_VLLM.md)**.

## Research questions

| RQ | Question |
|----|----------|
| **RQ1** | Which reasoning language works best for Yoruba tasks? |
| **RQ2** | Does TTC improve performance on low-resource languages like Yoruba? |
| **RQ3** | Where is the bottleneck: candidate generation or selection? |
| **RQ4** | Can a small model plus TTC match a larger model without TTC? |

## Datasets

All runs use **test splits only** (`data/normalized/.../test.jsonl`):

| Name | Task | Rows | Path |
|------|------|------|------|
| `afrimgsm` | Math (Yoruba) | 250 | `math-reasoning/afrimgsm/test.jsonl` |
| `afrimmlu` | QA (Yoruba) | 500 | `question-answering/afrimmlu/test.jsonl` |
| `afrimgsm_translate` | Math (English input, baseline) | 250 | `math-reasoning/afrimgsm_translate/test.jsonl` |
| `afrimmlu_translate` | QA (English input, baseline) | 500 | `question-answering/afrimmlu_translate/test.jsonl` |

Normalized JSONL rows: `{"answer_type": "choice|number", "choices": [...] | null, "gold_answer": "...", "question": "..."}`

```bash
uv run python scripts/download_hf_datasets.py --dataset all
```

## Setup

```bash
uv lock && uv venv
uv pip install -r requirements.txt   # cloud GPU image: install CUDA torch first, then vllm if needed
```

Set `HF_TOKEN` for gated models. Qwen3 native thinking is disabled in both backends (`enable_thinking=False`) — this is a prompted-CoT experiment.

## Experiment plan

Run order: **E0 → E1 → E2 → E3 → E4**. After E1, set the E2 config's `prompt_style` to the winning strategy.

### E0 — English baseline (multilingual reasoning gap)

English question → English CoT → answer on the native `eng` splits of the same benchmarks; compare against each language's `english_cot` rows from E1 to measure the reasoning gap.

```bash
uv run python scripts/run_inference.py --config configs/e0_english_baseline.json            # HF
uv run python scripts/run_inference.py --config configs/e0_english_baseline_vllm.json       # vLLM
uv run python scripts/run_inference.py --config configs/e0_english_baseline_openrouter.json  # OpenRouter
uv run python scripts/run_inference.py --config configs/e0_english_baseline_ramp_router.json # Ramp Router
```

### E1 — Reasoning language (greedy N=1)

Which strategy maximizes performance: `yoruba_cot`, `english_cot`, or `translate_pivot`?

```bash
# HF Transformers
uv run python scripts/run_inference.py --config configs/e1_reasoning_language.json

# Or local vLLM (see README_VLLM.md)
uv run python scripts/run_inference.py --config configs/e1_reasoning_language_vllm.json

```

Models: `qwen3-4b`, `gemma3-4b`, `llama3.2-3b`. Cloud-router variants exist (`configs/e1_reasoning_language_openrouter.json`, `configs/e1_reasoning_language_ramp_router.json`); both read their API key from env (`OPENROUTER_API_KEY` / `ROUTER_KEY`, auto-loaded from `.env`).

### E2 — TTC scaling

Method `english_cot_ttc` expanded to `_n1…_n64`: N ∈ {1, 4, 8, 16, 32, 64}, `nested_n: true`.
N=1 is a true greedy decode (temp 0, `selection=first`); N≥4 are nested prefixes of one stochastic pool sampled at max N (temp 0.7, top-p 0.7).

```bash
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling.json          # Qwen 4B/14B/32B ladder
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling_vllm.json     # vLLM variant

# Aggregate accuracy / tokens / latency / truncation + plots
uv run python scripts/aggregate_ttc_metrics.py --runs-dir runs --run-id e2_ttc_scaling
```

Outputs under `results/ttc_scaling/`: metrics JSON/CSV, `accuracy_vs_n.png`, `accuracy_vs_tokens.png`.

### E3 — Generation vs selection 

Reuses E2 traces. Per N: `pass@N` (generation ceiling) vs `first` / `majority_vote` selection.
High pass@N + low select@N → selection bottleneck; low pass@N → generation bottleneck.

```bash
uv run python scripts/reselect_candidates.py \
  --run-id e2_ttc_scaling --strategies first,majority_vote
```

Writes `results/e3_reselection/<run_id>/` (`selections_*.jsonl`, `e3_report.json`).

### E4 — Small model + TTC vs large greedy (offline)

Can Qwen 4B with TTC match Qwen 32B greedy (N=1)? Compares saved E2 metrics only — no model load.

```bash
uv run python scripts/compare_e4.py --runs-dir runs --run-id e2_ttc_scaling
# or: uv run python scripts/compare_e4.py --metrics-json results/ttc_scaling/metrics.json
```

Writes `results/e4_comparison/` (JSON, CSV table, plots).

## Inference runner

Primary entrypoint: `scripts/run_inference.py`.

```bash
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language.json \
  --datasets afrimgsm,afrimmlu \
  --models qwen3-4b \
  --methods yoruba_cot,english_cot \
  --limit 5 \
  --run-id my_run \
  --overwrite          # wipe prior artifacts for this run_id
# default: --resume (skip finished units via completed_units.jsonl)
```

Each run writes under `runs/<run_id>/`:

| Artifact | Contents |
|----------|----------|
| `candidates.jsonl` | Every sampled candidate (incl. `finish_reason`) |
| `selections.jsonl` | One selected answer per condition × example |
| `manifest.json` | Run metadata and counts |
| `completed_units.jsonl` | Resume checkpoint |

## Evaluation

```bash
uv run python scripts/evaluate_runs.py                 # pass@N / select@N for finished runs
uv run python scripts/evaluate_runs.py --run-id e1_reasoning_language
```

- **pass@N** — any of *N* candidates correct (generation ceiling)
- **select@N** — selected answer correct
- **Trunc%** — fraction of generations that hit the token limit (in `aggregate_ttc_metrics.py`; check it before trusting scaling curves)

## Repository layout

```text
configs/           Experiment and backend configs
data/normalized/   Yoruba JSONL datasets (gitignored; download locally)
scripts/           CLI entrypoints
src/ttcs_yoruba/   Library code (prompting, inference, metrics, selection)
tests/             Unit tests (.venv/bin/python -m pytest tests -q)
runs/              Inference artifacts (gitignored)
results/           Evaluation outputs (gitignored)
```

## Scripts

| Script | Role |
|--------|------|
| `scripts/download_hf_datasets.py` | Download / normalize benchmarks |
| `scripts/run_inference.py` | E1/E2 inference |
| `scripts/evaluate_runs.py` | pass@N / select@N summary |
| `scripts/aggregate_ttc_metrics.py` | E2 metrics + plots |
| `scripts/reselect_candidates.py` | E3 offline re-selection |
| `scripts/compare_e4.py` | E4 small+TTC vs large greedy |
| `scripts/show_exemplar_prompts.py` | Dump sample prompts |
