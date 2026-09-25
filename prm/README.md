# Discriminative PRM (Yoruba TTC verifier)

A **process reward model** (PRM) that scores each step of a reasoning trace,
adapted from [Multilingual-PRM](https://github.com/weixuan-wang123/Multilingual-PRM)
(Wang et al., *Demystifying Multilingual Reasoning in Process Reward Modeling*,
Findings of EMNLP 2025). The original code is kept verbatim under
[`reference/`](reference/) for traceability; everything else here is our
adaptation.

The verifier is `Qwen/Qwen2.5-Math-PRM-7B` (an official
`Qwen2ForProcessRewardModel`), fine-tuned with **LoRA** on **English PRM800K +
a Yoruba translation** (true per-step labels) and used for Best-of-N selection
over the E2 candidate pools. It reads a per-step reward from its `<extra_0>`
reward head. Only these step labels are used for training — no outcome-derived
supervision anywhere.

## What changed vs. the reference

| Aspect | Reference | Here |
|--------|-----------|------|
| Base model | `Qwen/Qwen2.5-Math-7B-Instruct` | `Qwen/Qwen2.5-Math-PRM-7B` |
| Fine-tuning | full weights, DeepSpeed, W&B, Hub push | **LoRA**, config-driven, no side effects |
| Training data | translated PRM800K / Math-Shepherd | **English PRM800K + supplied Yoruba translation**, step-labelled |
| Candidate generation | `infer-mgsm.py` (N=64 per seed) | reuse E2 pools (`runs/*/candidates.jsonl`) |
| Grading | sympy/LaTeX math only | `ttcs_yoruba.extraction` (**number + choice**) |
| Selection | script-local | core `select_candidate(..., "prm")`; default stays `majority_vote` |

## Layout

```text
prm/
├── reference/                 Original cloned code (unmodified)
├── configs/                   build_data / train_prm / score_candidates
├── scripts/                   CLI entrypoints
├── src/prm_yoruba/            Library (steps, grading, data, model, train, select)
├── tests/                     CPU-only unit tests
├── data/
│   ├── raw/                   Downloaded PRM800K + supplied Yoruba file (gitignored)
│   └── processed/             Built train/eval JSONL (gitignored)
├── checkpoints/               LoRA adapters (gitignored)
└── results/prm_selection/     Scores + selections (gitignored)
```

## Install (cloud GPU)

```bash
uv pip install -r prm/requirements-prm.txt
```

`huggingface_hub` + `datasets` are used to fetch the released PRM800K; `torch`,
`transformers`, `peft`, `accelerate`, and `scikit-learn` are needed for
train/score. Data building is CPU-only, but by default it downloads ~456 MB of
English PRM800K (see below), so it is normally run on the cloud box too.

## Pipeline

Run everything from the repository root.

### 1. Build training data (step-labelled PRM800K)

Default recipe: **English PRM800K + your Yoruba translation**. The `prm800k`
source downloads the released English files from
`vicky23456/multilingual-PRM800K` and reads the Yoruba file you supply.

```bash
uv run python prm/scripts/build_data.py --config prm/configs/build_data.json
```

Outputs `prm/data/processed/{train,eval,all}.jsonl` + `manifest.json`
(`manifest.json` reports counts by source, language, and task).

Knobs in `sources[0]` (`build_data.json`):

| Key | Meaning |
|-----|---------|
| `languages` | `["en", "yor"]` default. Released: `de, en, es, fr, ru, sw, zh` (no Yoruba). |
| `phase1` | `true` → take `phase1_train_<lang>` in full. |
| `phase2_max_examples` | Phase2 rows sampled per language; `0`/`null` disables phase2. |
| `max_examples_per_language` | Hard cap after phase1+phase2. |
| `paths` | Local files per language; required for any language not released. |

> **Download note.** The released phase2 file is a single pretty-printed JSON
> array, so it cannot be streamed. Even with `phase2_max_examples: 20000` the
> full `phase2_train_en.new.json` (~450 MB) is downloaded before subsampling.
> Set `phase2_max_examples: 0` for a ~5.8 MB phase1-only build.

#### Supplying the Yoruba PRM800K file

Yoruba is **not** in the released dataset, so point `sources[].paths.yor` at one
or more JSON files using the released schema:

```json
[
  {"question": "...", "process": "step 1 \n\n\n\n\n step 2 \n\n\n\n\n", "label": ["+", "-"]}
]
```

- `process`: steps joined by the Qwen step tag ` \n\n\n\n\n` (a missing trailing
  tag is added automatically).
- `label`: one `+`/`-` per step; its length must equal the number of step tags.
- Invalid rows (missing fields or tag/label mismatch) are skipped.

Useful flags: `--output-dir`, `--eval-fraction`, `--seed`.

### 2. LoRA fine-tune the PRM

```bash
uv run python prm/scripts/train_prm.py --config prm/configs/train_prm.json
```

Saves a LoRA adapter to `prm/checkpoints/<run_name>`. Add `--load-in-4bit` for
QLoRA or `--full-finetune` to update all weights. Only the `+`/`-` tokens at step
boundaries contribute to the loss. Overrides: `--run-name`, `--model`,
`--train-data`, `--eval-data`, `--output-dir`, `--learning-rate`, `--epochs`.

### 3. Score E2 candidate pools and select Best-of-N

```bash
uv run python prm/scripts/score_candidates.py --config prm/configs/score_candidates.json
```

Set `adapter_path` in the config (or pass `--adapter-path`) to use the fine-tuned
adapter; leave it `null` for the base PRM. `aggregation` is one of
`last|max|mean|min`. `max_input_tokens` (default `null`) optionally truncates long
traces; the result carries a `truncated` flag.

While scoring, it prints a running summary as each N finishes, comparing the
PRM's pick against the ground truth:

```text
[translate_pivot_ttc_n1] N=1  examples=250  pass@N=42.0%  prm@N=38.0%  gap=-4.0%
```

Add `--per-example` for one line per selection (`N`, example, PRM score, pick,
gold, correct); `--no-progress` silences the per-N lines. Writes:

- `prm/results/prm_selection/scores.jsonl` — every candidate with `prm_score`
- `prm/results/prm_selection/selections_prm.jsonl` — one pick per group
- `prm/results/prm_selection/summary.json` — pass@N vs select@N

Other flags: `--run-id` (repeatable), `--runs-dir`, `--output-dir`,
`--aggregation`, `--max-steps`, `--limit-groups`.

## Selection integration

The core selector gained a GPU-free `prm` strategy
(`src/ttcs_yoruba/selection.py`). It reads a precomputed `prm_score` from each
candidate, so scoring (GPU) and selection (CPU) are decoupled:

```bash
# defaults are unchanged: first, majority_vote
uv run python scripts/reselect_candidates.py \
  --run-id <run_id> --strategies first,majority_vote,prm
```

`--strategies prm` requires candidates that already carry `prm_score` (run step 3
against the same `run_id` first); otherwise it fails with a clear error.

## Preprocessing contract

Two interfaces, selected by `prm_interface`:

**`qwen2.5-math-prm` (default)** — official `Qwen2ForProcessRewardModel`
checkpoints. Loaded with `AutoModel` + `trust_remote_code=True`. Each candidate
is segmented on our `\n` delimiter (the `\n\nFinal answer:` line is dropped,
echoed question/translation removed, list/`Step k:` prefixes stripped), then the
steps are rendered as the assistant turn `s1<extra_0>s2<extra_0>…` inside the
model's chat template (system + user + assistant). The 2-class reward head is
read at the `<extra_0>` positions: `softmax(logits, -1)[..., 1]` gives that
step's reward in `[0, 1]`. Fine-tuning trains the same head with cross-entropy
(labels 0/1 at `<extra_0>` positions, `-100` elsewhere) under LoRA adapters.

**`token_classifier`** — the reference repo's `+`/`-` LM-token method for
`Qwen2.5-Math-7B-Instruct`-style checkpoints: `question + " " + process`, step
tag `\n\n\n\n\n` (Qwen) or `ки` (Mistral), good/bad tokens ` +` / ` -`.
`prm_yoruba.model.resolve_prm_tokens` derives the ids and requires exactly two
candidate tokens.

Long traces drop **leading** steps to fit the context (the final step is kept
for `last` aggregation); `max_steps` defaults to 256 and `max_input_tokens`
defaults to the model context.

## Reproducing checks locally (no GPU)

```bash
.venv/bin/python -m pytest prm/tests -q
```

Covers step segmentation, PRM800K validation/loading, token resolution,
score aggregation, and the `prm` selection strategy.

## Limitations

- The Yoruba step labels are only as good as the translation; spot-check a
  sample before a long training run.
- The base PRM is math-trained, so expect weaker step scoring on AfriMMLU; that
  is a limitation of this data mix, not something we paper over with outcome
  labels.

## Citation

```
@article{wang2025demystifying,
  title={Demystifying Multilingual Chain-of-Thought in Process Reward Modeling},
  author={Wang, Weixuan and Wu, Minghao and Haddow, Barry and Birch, Alexandra},
  journal={arXiv preprint arXiv:2502.12663},
  year={2025}
}
```
