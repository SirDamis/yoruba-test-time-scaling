# Running experiments with local vLLM

End-to-end guide for this repo using a **local vLLM** server (no paid API).  
Recommended GPU: **NVIDIA L4** (24 GB). A T4 can run **4B** models more slowly; **14B/32B** need more VRAM or quantization.

**You do not need `flash-attn`.** vLLM ships its own efficient attention kernels. Compiling `flash-attn` is unnecessary for any `configs/*_vllm.json` run and can exhaust RAM.

Related experiment design lives in the main [README.md](README.md). This file is **ops-focused**: install → serve → run → evaluate.

---

## Architecture

```text
Terminal A                    Terminal B (this repo)
──────────                    ─────────────────────
vLLM serves model  ──HTTP──►  scripts/run_inference.py
  :8000/v1                    backend: openai_compatible
                              configs/*_vllm.json
```

| Role | Component |
|------|-----------|
| Server | `vllm serve …` or `./scripts/serve_vllm.sh` |
| Client | `scripts/run_inference.py` + `*_vllm.json` configs |
| Env | `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY` |

---

## 1. Prerequisites

```bash
nvidia-smi
python3 --version   # need Python >= 3.12
```

- NVIDIA driver + CUDA working (`nvidia-smi` lists the GPU).
- Enough disk for model weights (Qwen3-4B is several GB).
- Two terminals (or `tmux`/`screen`): one for vLLM, one for the client.

---

## 2. Clone and create the project environment

```bash
git clone https://github.com/SirDamis/yoruba-test-time-scaling.git
cd yoruba-test-time-scaling
git pull origin main   # if already cloned

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv lock
uv venv
source .venv/bin/activate
```

---

## 3. Install dependencies

Install **CUDA PyTorch first**, then project packages, then **vLLM**.  
**Do not install `flash-attn`.**

```bash
source .venv/bin/activate

# CUDA torch (adjust index if your CUDA stack needs cu121)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

uv pip install -r requirements.txt
uv pip install vllm
```

Verify:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import vllm
print("vllm ok")
PY
```

Optional Hugging Face token (gated models later; Qwen often works without):

```bash
export HF_TOKEN="hf_..."
```

---

## 4. Download data

AfriMGSM only (typical pilot):

```bash
uv run python scripts/download_hf_datasets.py --dataset afrimgsm
wc -l data/normalized/math-reasoning/afrimgsm/*.jsonl
# train 8, test 250, all 258
```

All benchmarks:

```bash
uv run python scripts/download_hf_datasets.py --dataset all
```

---

## 5. Serve a model with vLLM (Terminal A)

Keep this process running for the whole experiment.

```bash
source .venv/bin/activate
chmod +x scripts/serve_vllm.sh   # once

# Default: Qwen/Qwen3-4B on port 8000, max-model-len 4096
./scripts/serve_vllm.sh Qwen/Qwen3-4B

# Custom port / context length
# ./scripts/serve_vllm.sh Qwen/Qwen3-4B 8000 4096
```

Equivalent manual command:

```bash
vllm serve Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

Health check (from any terminal):

```bash
curl -s http://localhost:8000/v1/models | head
```

**Important:** serve **one** HF model id at a time. The client `--models` filter and config `model` field must match what vLLM loaded (e.g. `Qwen/Qwen3-4B` ↔ config name `qwen3-4b`).

Use `tmux` so SSH drop does not kill the server:

```bash
tmux new -s vllm
./scripts/serve_vllm.sh Qwen/Qwen3-4B
# detach: Ctrl-b then d
```

---

## 6. Client environment (Terminal B)

```bash
cd yoruba-test-time-scaling
source .venv/bin/activate

export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"
```

Put these exports in `~/.bashrc` if you prefer them permanent for this machine.

---

## 7. Experiment configs (vLLM)

| Experiment | Config | Default `run_id` | Notes |
|------------|--------|------------------|--------|
| **E1** reasoning language | `configs/e1_reasoning_language_vllm.json` | `e1_reasoning_language_vllm` | 3 strategies, `N=1`, greedy |
| **E2** TTC (Qwen ladder) | `configs/e2_ttc_scaling_vllm.json` | `e2_ttc_scaling_vllm` | Nested `N=1…64` |
| **E2** optional families | `configs/e2_ttc_scaling_optional_vllm.json` | `e2_ttc_scaling_optional_vllm` | Gemma / Llama |
| **E4** compare (offline) | `configs/e4_comparison_vllm.json` | — | No inference; uses E2 metrics |

Shared defaults on vLLM inference configs:

- `backend`: `openai_compatible`
- `max_tokens`: **512** (AfriMGSM-friendly)
- `max_concurrent`: **8** (in-flight HTTP requests for continuous batching)
- `default_request_timeout_s`: **300**

HF Transformers configs (`e1_reasoning_language.json`, etc.) are a separate path and are **not** covered here.

---

## 8. Smoke test (do this first)

With vLLM running and env vars set:

```bash
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm_smoke \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods english_cot \
  --limit 5 \
  --max-concurrent 4 \
  --overwrite
```

Expect:

- Progress lines on stderr (`1/5`, `OK`/`WRONG`, latency).
- Artifacts under `runs/e1_afrimgsm_qwen3-4b_vllm_smoke/`:
  - `candidates.jsonl`
  - `selections.jsonl`
  - `completed_units.jsonl`
  - `manifest.json`

If this fails, fix serving/env before a full run (see [Troubleshooting](#13-troubleshooting)).

---

## 9. E1 — full AfriMGSM (reasoning language)

**RQ1:** Yoruba CoT vs English CoT vs translate-pivot.

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"

# optional: tmux new -s e1

uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm \
  --models qwen3-4b \
  --max-concurrent 8 \
  --overwrite
```

| Factor | Value |
|--------|--------|
| Methods | `yoruba_cot`, `english_cot`, `translate_pivot` |
| Examples (config path `all.jsonl`) | 258 |
| Samples per example | 1 (greedy) |
| Generations | **3 × 258 = 774** |

### Resume after crash / SSH drop

Omit `--overwrite` (resume is default):

```bash
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm \
  --models qwen3-4b \
  --max-concurrent 8
```

### Other E1 filters

```bash
# One method only
--methods yoruba_cot

# Multiple datasets
--datasets afrimgsm,afrimmlu

# Another model (serve that model in Terminal A first)
--models gemma3-4b
```

---

## 10. Evaluate E1 runs

```bash
# Summary for one run
uv run python scripts/evaluate_runs.py --run-id e1_afrimgsm_qwen3-4b_vllm

# Optional: write aggregated JSON
uv run python scripts/evaluate_runs.py \
  --run-id e1_afrimgsm_qwen3-4b_vllm \
  --output results/e1_afrimgsm_qwen3-4b_vllm_eval.json
```

Interpret **per method** (`yoruba_cot` / `english_cot` / `translate_pivot`), not only the running accuracy printed during generation.

Useful artifacts:

| Path | Contents |
|------|----------|
| `runs/<run_id>/selections.jsonl` | Selected answer, `is_correct`, gold |
| `runs/<run_id>/candidates.jsonl` | Full responses |
| `runs/<run_id>/manifest.json` | Counts, methods, concurrency |

Quick accuracy by method (optional one-liner):

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
path = Path("runs/e1_afrimgsm_qwen3-4b_vllm/selections.jsonl")
stats = defaultdict(lambda: [0, 0])  # correct, total
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    m = row["method"]
    stats[m][1] += 1
    if row.get("is_correct"):
        stats[m][0] += 1
for m, (c, t) in sorted(stats.items()):
    print(f"{m:20s}  {c}/{t}  {100*c/t:.1f}%")
PY
```

---

## 11. E2 — TTC scaling (after E1)

1. Pick the best E1 `prompt_style` (`english_cot`, `yoruba_cot`, or `translate_pivot`).
2. Set `methods[0].prompt_style` in `configs/e2_ttc_scaling_vllm.json` to that winner (default is `english_cot`).
3. Serve the model you will filter with `--models`.
4. Run:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"

uv run python scripts/run_inference.py \
  --config configs/e2_ttc_scaling_vllm.json \
  --run-id e2_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm \
  --models qwen3-4b \
  --max-concurrent 8 \
  --overwrite
```

Smoke (one expanded method name, few examples):

```bash
uv run python scripts/run_inference.py \
  --config configs/e2_ttc_scaling_vllm.json \
  --run-id e2_smoke_vllm \
  --datasets afrimgsm \
  --models qwen3-4b \
  --methods english_cot_ttc_n4 \
  --limit 5 \
  --overwrite
```

Aggregate curves:

```bash
uv run python scripts/aggregate_ttc_metrics.py \
  --runs-dir runs \
  --run-id e2_afrimgsm_qwen3-4b_vllm
```

Outputs under `results/ttc_scaling/` (metrics JSON/CSV, plots).

---

## 12. E3 and E4 (offline — no vLLM needed)

These read finished run artifacts only. Stop or keep the server; GPU is not required.

### E3 — generation vs selection

```bash
uv run python scripts/reselect_candidates.py \
  --run-id e2_afrimgsm_qwen3-4b_vllm \
  --strategies first,majority_vote
```

Writes `results/e3_reselection/<run_id>/`.

### E4 — small + TTC vs large greedy

Requires E2 traces that include both small and large models (or multiple runs merged into metrics).

```bash
uv run python scripts/compare_e4.py \
  --runs-dir runs \
  --run-id e2_ttc_scaling_vllm
```

Or from aggregated metrics:

```bash
uv run python scripts/compare_e4.py \
  --metrics-json results/ttc_scaling/metrics.json
```

---

## 13. Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `unrecognized arguments: --disable-log-requests` | Pull latest `scripts/serve_vllm.sh`, or serve with the manual command in §5 (no that flag). |
| `Connection refused` / request failed | vLLM not up; check Terminal A and `curl localhost:8000/v1/models`. |
| Model / 404 from API | Client `model` id must match served model (`Qwen/Qwen3-4B`). |
| OOM on GPU | Lower `--gpu-memory-utilization` (e.g. `0.80`), lower `--max-concurrent`, or use a smaller model. |
| Client timeouts | Config already uses 300s; raise `default_request_timeout_s` in the JSON if needed. |
| Slow or low GPU util | Increase `--max-concurrent` (8→16) carefully; ensure many requests in flight. |
| Want less concurrency | `--max-concurrent 1` or `4`. |
| Accidental HF path | Use `*_vllm.json` configs, not `e1_reasoning_language.json`. |
| Mixed engines in one folder | Always use a **new `--run-id`** for vLLM vs Transformers. |
| `flash-attn` OOM during install | **Do not install it** for this guide. |
| `yoruba_cot` still reasons in English (Qwen3 `<think>` blocks) | **Not a wrong prompt** — Qwen3 default “thinking” mode is English-heavy. Configs set `chat_template_kwargs.enable_thinking=false` for Qwen. Restart vLLM + re-run with a **new run-id**. Also a valid **paper finding** (instruction non-compliance / language switch). |

---


## 14. Quick reference

```bash
# --- once ---
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
uv pip install vllm
uv run python scripts/download_hf_datasets.py --dataset afrimgsm

# --- Terminal A ---
./scripts/serve_vllm.sh Qwen/Qwen3-4B

# --- Terminal B ---
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"

# smoke
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm_smoke \
  --datasets afrimgsm --models qwen3-4b \
  --methods english_cot --limit 5 --max-concurrent 4 --overwrite

# full E1
uv run python scripts/run_inference.py \
  --config configs/e1_reasoning_language_vllm.json \
  --run-id e1_afrimgsm_qwen3-4b_vllm \
  --datasets afrimgsm --models qwen3-4b \
  --max-concurrent 8 --overwrite

# evaluate
uv run python scripts/evaluate_runs.py --run-id e1_afrimgsm_qwen3-4b_vllm
```

---

## 15. Config / model name map

| CLI `--models` | HF / vLLM serve id |
|----------------|--------------------|
| `qwen3-4b` | `Qwen/Qwen3-4B` |
| `qwen3-14b` | `Qwen/Qwen3-14B` |
| `qwen3-32b` | `Qwen/Qwen3-32B` |
| `gemma3-4b` | `google/gemma-3-4b-it` |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` |

Serve the right id in Terminal A before filtering that name in Terminal B.
