# Running experiments with local vLLM

Ops guide for this repo using a **local vLLM** server with GPU.

Experiment design lives in [README.md](README.md).

---

## 1. Install

```bash
git clone https://github.com/SirDamis/yoruba-test-time-scaling.git
cd yoruba-test-time-scaling

curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
uv lock && uv venv && source .venv/bin/activate

uv pip install -r requirements.txt

# optional (gated models):
export HF_TOKEN="hf_..."
```

## 2. Download data

```bash
uv run python scripts/download_hf_datasets.py --dataset all
```

## 3. Serve a model (Terminal A)

Keep this process running; use `tmux` so SSH drop doesn't kill it.

```bash
chmod +x scripts/serve_vllm.sh    # once
./scripts/serve_vllm.sh Qwen/Qwen3-4B
# equivalent: vllm serve Qwen/Qwen3-4B --host 0.0.0.0 --port 8000 \
#   --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.90

curl -s http://localhost:8000/v1/models | head   # health check
```

Serve **one** HF model id at a time; the client `--models` filter must match what vLLM loaded (`Qwen/Qwen3-4B` ↔ config name `qwen3-4b`). Model map:

| CLI `--models` | Serve id |
|----------------|----------|
| `qwen3-4b` | `Qwen/Qwen3-4B` |
| `qwen3-14b` | `Qwen/Qwen3-14B` |
| `qwen3-32b` | `Qwen/Qwen3-32B` |
| `gemma3-4b` | `google/gemma-3-4b-it` |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` |

## 4. Client environment (Terminal B)

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
```

Shared config defaults: `backend: openai_compatible`, `max_tokens: 1024`, `max_concurrent: 8`, timeout 300s. Qwen3 thinking is disabled via `chat_template_kwargs.enable_thinking=false`.

| Experiment | Config | Default `run_id` |
|------------|--------|------------------|
| E1 reasoning language | `configs/e1_reasoning_language_vllm.json` | `e1_reasoning_language_vllm` |
| E2 TTC (Qwen ladder) | `configs/e2_ttc_scaling_vllm.json` | `e2_ttc_scaling_vllm` |
| E2 optional families | `configs/e2_ttc_scaling_optional_vllm.json` | `e2_ttc_scaling_optional_vllm` |
| E4 compare (offline) | `configs/e4_comparison_vllm.json` | — |


## 5. Full runs

**E1** (RQ1: `yoruba_cot` vs `english_cot` vs `translate_pivot`; greedy N=1):

```bash
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm --models qwen3-4b \
  --max-concurrent 8 --overwrite
```

Resume after a crash by re-running without `--overwrite`.

**E2** (after picking the E1-winning `prompt_style` into the config):

```bash
uv run python scripts/run_inference.py \
  --config configs/e2_ttc_scaling_vllm.json \
  --run-id e2_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm --models qwen3-4b \
  --max-concurrent 8 --overwrite
```

**Evaluate / aggregate:**

```bash
uv run python scripts/evaluate_runs.py --run-id e1_afrimgsm_qwen3-4b_vllm
uv run python scripts/aggregate_ttc_metrics.py --runs-dir runs --run-id e2_afrimgsm_qwen3-4b_vllm
```

## 7. E3 and E4 (offline — no GPU needed)

```bash
uv run python scripts/reselect_candidates.py \
  --run-id e2_afrimgsm_qwen3-4b_vllm --strategies first,majority_vote

uv run python scripts/compare_e4.py --runs-dir runs --run-id e2_ttc_scaling_vllm
```
