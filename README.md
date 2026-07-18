# Yoruba Test-Time Compute Scaling

Experiment pipeline for **When Is More Thinking Enough? Evaluation of Test-Time Compute Scaling for Yoruba Language Reasoning**.

**Core idea:** Can test-time compute (TTC) scaling compensate for weak low-resource language representations?

Evaluation is **Yoruba-only**. English appears only as an inference intervention (English CoT, translate-to-English pivot). There are no standalone English benchmark runs.

## Research questions

| RQ | Question |
|----|----------|
| **RQ1** | Which reasoning language works best for Yoruba tasks? |
| **RQ2** | Does TTC improve performance on low-resource languages like Yoruba? |
| **RQ3** | Where is the bottleneck: candidate generation or selection? |
| **RQ4** | Can a small model plus TTC match a larger model without TTC? |

## Datasets

| Name | Task | Path |
|------|------|------|
| `afrimgsm` | Math | `data/normalized/math-reasoning/afrimgsm/` |
| `afrimmlu` | QA | `data/normalized/question-answering/afrimmlu/` |
| `afriqa` | QA | `data/normalized/question-answering/afriqa/` |
| `naijarc` | Reading comprehension | `data/normalized/question-answering/naijarc/` |

Normalized JSONL rows look like:

```json
{
  "answer_type": "choice|number|text|freeform|instruction",
  "choices": ["A. ...", "B. ..."],
  "gold_answer": "B",
  "question": "Yoruba prompt"
}
```

`choices` is `null` when the dataset is not multiple-choice.

## Setup

```bash
uv lock
uv venv
uv pip install -r requirements.txt
uv run python scripts/download_hf_datasets.py --dataset all
```

The default downloader uses registered file URLs (no Hugging Face `datasets` package required). To force that backend:

```bash
uv run --with datasets python scripts/download_hf_datasets.py --dataset afrimgsm --backend datasets
```

Set `HF_TOKEN` when loading gated models (Llama, Gemma):

```bash
export HF_TOKEN="..."
```

## Experiment plan

Run order: **E1 → E2 → E3 → E4**. After E1, set E2 `prompt_style` to the winning strategy.

### E1 — Reasoning language

**Question:** Which strategy maximizes performance on Yoruba tasks?

| Field | Value |
|-------|--------|
| **Config (HF Transformers)** | `configs/e1_reasoning_language.json` |
| **Config (local vLLM)** | `configs/e1_reasoning_language_vllm.json` |
| **Models** | `qwen3-4b`, `gemma3-4b`, `llama3.2-3b` |
| **Methods** | `yoruba_cot`, `english_cot`, `translate_pivot` (greedy `N=1`) |

| Strategy | Pipeline |
|----------|----------|
| **Yoruba CoT** | Yoruba question → Yoruba reasoning → Yoruba answer |
| **English CoT** | Yoruba question → English reasoning → Yoruba answer |
| **Translate Pivot** | Translate question → English reasoning → Yoruba answer |

```bash
# Full E1 (in-process Transformers)
uv run python scripts/run_inference.py --config configs/e1_reasoning_language.json

# One dataset × one model (all 3 strategies), fresh start
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --overwrite

# Smoke test
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods english_cot \
  --limit 5
```

**Local vLLM (same E1 protocol, faster serving; no paid API):** start the server in one terminal, then run E1 against it.

E1/E2 configs use **`max_tokens: 512`** (good for AfriMGSM). vLLM configs set **`max_concurrent: 8`** so the client issues multiple in-flight requests and vLLM continuous-batches them. Transformers configs stay at concurrency 1 (HF `generate` is not thread-safe) and use **`attn_implementation: auto`** (FlashAttention-2 on L4/SM≥8 when `flash-attn` is installed, else SDPA; T4 → SDPA).

```bash
# Terminal A — L4-friendly helper (or plain vllm serve)
./scripts/serve_vllm.sh Qwen/Qwen3-4B
# equivalent:
# vllm serve Qwen/Qwen3-4B --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 4096

# Terminal B — point the client at local vLLM
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"

uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm \
  --models qwen3-4b \
  --max-concurrent 8 \
  --overwrite
```

Optional: `pip install flash-attn` on L4 for FA2 with the Transformers backend. vLLM enables efficient attention kernels itself on supported GPUs.

Serve Gemma or Llama the same way (one model per vLLM process), then pass the matching `--models` name.

### E2 — TTC scaling

**Question:** How does test-time compute scale on Yoruba tasks?

| Field | Value |
|-------|--------|
| **Primary config (HF)** | `configs/e2_ttc_scaling.json` (Qwen 4B / 14B / 32B) |
| **Primary config (vLLM)** | `configs/e2_ttc_scaling_vllm.json` |
| **Optional families (HF)** | `configs/e2_ttc_scaling_optional.json` (Gemma, Llama) |
| **Optional families (vLLM)** | `configs/e2_ttc_scaling_optional_vllm.json` |
| **Method** | `english_cot_ttc` (expanded to `english_cot_ttc_n1` … `_n64`) |
| **N** | 1, 4, 8, 16, 32, 64 with `nested_n: true` |
| **N=1** | True greedy (`greedy_n1`, temp 0, `selection=first`) |
| **N≥4** | One stochastic pool at max *N*; prefixes scored for each *k* |

After E1, set `methods[0].prompt_style` in the E2 config to the winner (`english_cot`, `yoruba_cot`, or `translate_pivot`).

```bash
# Full E2 (Qwen ladder, Transformers)
uv run python scripts/run_inference.py --config configs/e2_ttc_scaling.json

# Smoke test (filter to one expanded method name)
uv run python scripts/run_inference.py \
  --config configs/e2_ttc_scaling.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods english_cot_ttc_n4 \
  --limit 5

# E2 via local vLLM (serve matching model first)
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
uv run python scripts/run_inference.py \
  --config configs/e2_ttc_scaling_vllm.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --overwrite

# Aggregate accuracy / tokens / latency + plots
uv run python scripts/aggregate_ttc_metrics.py --runs-dir runs
# With a vLLM run-id:
# uv run python scripts/aggregate_ttc_metrics.py --runs-dir runs --run-id e2_ttc_scaling_vllm
```

Outputs under `results/ttc_scaling/`: metrics JSON/CSV, `accuracy_vs_n.png`, `accuracy_vs_tokens.png`.

### E3 — Generation vs selection

**Question:** Is the bottleneck candidate generation or selection?

Reuses E2 traces (no new generations). For each *N*:

| Axis | Metric |
|------|--------|
| Generation quality | `pass@N` (any of *N* samples correct) |
| Selection quality | `first`, `majority_vote` |

- High `pass@N`, low select accuracy → selection bottleneck  
- Low `pass@N` → generation bottleneck  

No separate inference config — offline on whatever E2 `run_id` you produced (HF or vLLM).

```bash
# HF E2 traces
uv run python scripts/reselect_candidates.py \
  --run-id e2_ttc_scaling \
  --strategies first,majority_vote

# Local-vLLM E2 traces
uv run python scripts/reselect_candidates.py \
  --run-id e2_ttc_scaling_vllm \
  --strategies first,majority_vote
```

Writes `results/e3_reselection/<run_id>/` (`selections_*.jsonl`, `e3_report.json`).

### E4 — Small model + TTC vs large greedy

**Question:** Can Qwen 4B with TTC match Qwen 32B greedy (`N=1`)?

| Field | Value |
|-------|--------|
| **Config (HF E2 traces)** | `configs/e4_comparison.json` |
| **Config (vLLM E2 traces)** | `configs/e4_comparison_vllm.json` |
| **Small** | `qwen3-4b` over the TTC *N* sweep |
| **Large** | `qwen3-32b` at `N=1` |

E4 only compares saved metrics/runs (no model load). Use the vLLM comparison config’s notes and the matching E2 `run_id`.

```bash
# From aggregated E2 metrics
uv run python scripts/compare_e4.py \
  --metrics-json results/ttc_scaling/metrics.json

# Or from run artifacts (HF)
uv run python scripts/compare_e4.py --runs-dir runs --run-id e2_ttc_scaling

# Or from local-vLLM E2 artifacts
uv run python scripts/compare_e4.py --runs-dir runs --run-id e2_ttc_scaling_vllm
```

Writes `results/e4_comparison/` (JSON, CSV table, accuracy vs *N* / tokens / latency plots).

### Planned ablations

Not implemented as dedicated configs yet:

| Ablation | Design |
|----------|--------|
| Diacritics | With vs without diacritic marks under TTC |
| Adapted model | AfriqueQwen vs same-size base Qwen |
| Zero-shot CoT | No few-shot; vs greedy few-shot CoT |
| Few-shot language | Yoruba exemplars vs English exemplars |

## Inference runner

Primary entrypoint: `scripts/run_inference.py`.

Experiment configs (`e1_*`, `e2_*`) are preferred. Generic kitchen-sink configs also exist:

- `configs/inference.json` — Hugging Face Transformers backend  
- `configs/*_vllm.json` — same E1/E2 experiment protocols over local vLLM (`openai_compatible`)  
- `configs/e4_comparison_vllm.json` — E4 filters for vLLM E2 `run_id`s (offline compare only)  
- `configs/openai_compatible_inference.json` — generic multi-model kitchen sink (not a full experiment protocol)  

```bash
# Common flags
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language.json \
  --datasets afrimgsm,afrimmlu \
  --models qwen3-4b \
  --methods yoruba_cot,english_cot \
  --limit 5 \
  --run-id my_run \
  --overwrite          # wipe prior artifacts for this run_id
# default: --resume (skip finished units via completed_units.jsonl)
#          --no-resume to rewrite without using the checkpoint
```

**vLLM / OpenAI-compatible:**

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
uv run python scripts/run_inference.py \
  --config configs/openai_compatible_inference.json \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods english_cot \
  --limit 5
```

Each run writes under `runs/<run_id>/` (gitignored):

| Artifact | Contents |
|----------|----------|
| `candidates.jsonl` | Every sampled candidate |
| `selections.jsonl` | One selected answer per condition × example |
| `manifest.json` | Run metadata and counts |
| `completed_units.jsonl` | Resume checkpoint |

## Evaluation

```bash
# pass@N vs select@N for finished runs
uv run python scripts/evaluate_runs.py
uv run python scripts/evaluate_runs.py --run-id e1_reasoning_language
uv run python scripts/evaluate_runs.py \
  --run-id e2_ttc_scaling \
  --output results/aggregated.json
```

- **pass@N** — any of *N* candidates is correct (generation ceiling)  
- **select@N** — selected answer is correct  
- **gap** — `pass@N − select@N`  

For E2 scaling curves and E4 tables, use `aggregate_ttc_metrics.py` and `compare_e4.py` (above).

## Repository layout

```text
configs/           Experiment and backend configs
data/normalized/   Yoruba JSONL datasets (gitignored; download locally)
scripts/           CLI entrypoints
src/ttcs_yoruba/   Library code (prompting, inference, metrics, selection)
tests/             Unit tests
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
