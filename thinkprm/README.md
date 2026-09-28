# ThinkPRM (generative Yoruba TTC verifier)

A **generative process reward model** that verifies a reasoning trace by
*thinking*: it generates a verification chain-of-thought that critiques every
step, then answers `Is the solution correct? Yes/No`. We use the `Yes`
probability as the candidate score for Best-of-N selection over the E2 pools.

Adapted from [ThinkPRM: Process Reward Models That Think](https://github.com/mukhal/ThinkPRM)
(Khalifa et al., TMLR 2025). The original code is kept verbatim under
[`reference/`](reference/) for traceability; everything else here is our
adaptation. This is the generative counterpart to the discriminative
[`prm/`](../prm) folder.

## Why a second verifier

The discriminative PRM ([`prm/`](../prm)) fine-tunes `Qwen2.5-Math-PRM-7B` to
read a per-step reward from a classification head. ThinkPRM instead fine-tunes a
*reasoning* model to verbalise a critique. The paper's headline result is data
efficiency: ~1K synthetic verification chains (~8K process labels) beat
discriminative PRMs trained on two orders of magnitude more labels, and the
generative verifier can **scale its own test-time compute** (more/shorter
verification chains) the same way the generator can.

| Aspect | Discriminative PRM (`prm/`) | ThinkPRM (`thinkprm/`) |
|--------|-----------------------------|------------------------|
| Base model | `Qwen/Qwen2.5-Math-PRM-7B` | `launch/ThinkPRM-{1.5B,7B,14B}` (R1-Distill / QwQ) |
| Output | reward-head logit per `<extra_0>` | generated verification CoT + `Yes`/`No` |
| Signal used for selection | step reward (aggregated) | `P(Yes)` prefix score (aggregated over parallel chains) |
| Supervision | per-step labels directly | synthetic chains filtered by the same per-step labels |
| Trainable on CPU | no | no (cloud GPU) |
| Verifier compute scaling | none | **parallel** (`n_verifications`) and **sequential** (budget forcing) |

## Layout

```text
thinkprm/
├── reference/                 Original cloned ThinkPRM repo (unmodified)
├── configs/                   collect_data / train_thinkprm / score_candidates
├── scripts/                   CLI entrypoints
├── src/thinkprm_yoruba/       Library (prompts, parse, model, data, train, select)
├── tests/                     CPU-only unit tests
├── data/
│   ├── raw/                   (optional) teacher/source files (gitignored)
│   └── processed/             Built train/eval JSONL (gitignored)
├── checkpoints/               LoRA adapters (gitignored)
└── results/thinkprm_selection/ Scores + selections (gitignored)
```

## Install (cloud GPU)

```bash
uv pip install -r thinkprm/requirements-thinkprm.txt
```

`torch`, `transformers`, `peft`, `accelerate` and `datasets` are needed for
train/score. Data collection only needs the repo's base requirements plus a
reachable OpenAI-compatible teacher endpoint (it calls the same
`ttcs_yoruba.backends` stack as the inference runner).

## Pipeline

Run everything from the repository root. Steps 1–3 mirror [`prm/`](../prm).

### 0. (optional) Build step-labelled PRM data

`collect_data.py` consumes step-labelled rows (`question` + `steps` +
`label`). The easiest source is the discriminative-PRM build:

```bash
uv run python prm/scripts/build_data.py --config prm/configs/build_data.json
# -> prm/data/processed/all.jsonl
```

Point `prm_data` in `thinkprm/configs/collect_data.json` at that file.

### 1. Synthesise verification chains with a teacher

```bash
uv run python thinkprm/scripts/collect_data.py --config thinkprm/configs/collect_data.json
```

A strong reasoning teacher (default `QwQ-32B-Preview`) critiques each step. A
sampled chain is kept only when:

1. it contains one `\boxed{correct}` / `\boxed{incorrect}` judgement per step,
2. those judgements match the gold PRM800K labels up to the first error, and
3. it stays within the optional length budget (`max_cot_tokens`).

Set the teacher endpoint via `teacher.base_url_env` / `teacher.api_key_env`
(loaded from `.env`). Outputs
`thinkprm/data/processed/{train,eval,all}.jsonl` + `manifest.json`.

### 2. LoRA fine-tune the ThinkPRM verifier

```bash
uv run python thinkprm/scripts/train_thinkprm.py --config thinkprm/configs/train_thinkprm.json
```

Saves a LoRA adapter to `thinkprm/checkpoints/<run_name>`. Loss is applied only
to the assistant turn (the verification CoT + final decision). Add
`--load-in-4bit` for QLoRA or `--full-finetune` to update all weights.

### 3. Score E2 candidate pools and select Best-of-N

```bash
uv run python thinkprm/scripts/score_candidates.py --config thinkprm/configs/score_candidates.json
```

Set `adapter_path` to use the fine-tuned adapter; leave it `null` to use the
**released ThinkPRM off-the-shelf** (a strong English verifier baseline).
`n_verifications` samples that many parallel verification chains and averages
their `P(Yes)` (the paper's parallel scaling). `aggregation` is `prefix`
(default, the decision probability) or `last|max|mean|min` over the per-step
`1/0` labels.

**Scoring order and cost.** E2 pools are *nested*: `n1` is a prefix of `n2`,
which is a prefix of `n4`, and so on, and a candidate's ThinkPRM score does not
depend on N. The scorer therefore walks **example-outer / N-inner** and scores
each unique trace exactly once, reusing it across N (64 traces per example for a
full sweep, instead of `1+2+4+…+64 = 127`). Only the N conditions in the config's
`n_values` (default `[2,4,8,16,32,64]`) are reported — `n=1`/`n=3` are dropped,
since the `*_greedy` baseline covers N=1. Override with `--n-values` (or
`--n-values all` to report every N present):

```bash
uv run python thinkprm/scripts/score_candidates.py \
  --config thinkprm/configs/score_candidates.json
```

Progress is printed live per `(example, N)`, including whether the verifier's
pick was correct and its score, plus a candidate-level ETA:

```text
[ex 1/115] afrimgsm_yor_test_000001 n=64 (+32 new, 64/7360 scored)
    | sel=OK pick='18' gold='18' prm=0.32 | N acc=100.0% | ... cand/s | eta ...m
```

Tune with `--progress-every` / `--no-progress`, or add `--per-example` for one
line per pick.

While scoring it prints a running per-N summary, e.g.:

```text
[translate_pivot_ttc_n1] N=1  examples=250  pass@N=42.0%  thinkprm@N=38.0%  gap=-4.0%
```

Writes:

- `thinkprm/results/thinkprm_selection/scores.jsonl` — candidates + scores
- `thinkprm/results/thinkprm_selection/selections_thinkprm.jsonl` — picks
- `thinkprm/results/thinkprm_selection/summary.json` — pass@N vs select@N

## Selection integration

ThinkPRM writes the canonical `prm_score` field, so the existing GPU-free
selector works unchanged. A `thinkprm` alias was added to
`src/ttcs_yoruba/selection.py`:

```bash
# after step 3, candidates carry prm_score
uv run python scripts/reselect_candidates.py \
  --run-id <run_id> --strategies first,majority_vote,prm,thinkprm
```

`thinkprm` requires candidates that already carry `prm_score` (run step 3
against the same `run_id` first).

## Preprocessing contract

- **Steps** are re-segmented from the candidate `response` with the same
  `\n`-delimiter logic as the discriminative PRM (`prm_yoruba.steps`), then
  rendered as `Step 1: ...\nStep 2: ...`.
- **Prompt**: the exact ThinkPRM user template (`[Math Problem]` / `[Solution]`
  + instruction), applied through the model chat template with a generation
  prompt. Off-the-shelf scoring keeps the reference instruction; fine-tuned
  scoring should reuse whatever instruction was used during collection.
- **Score**: after generating the critique, the pre-decision string
  `Is the solution correct?` is appended and the next-token distribution is
  restricted to `" Yes"` / `" No"` and softmaxed at `decision_temperature`
  (matches the reference's calibrated decision). The per-step `1/0` labels are
  parsed from `\boxed{correct}` / `\boxed{incorrect}` and padded with `0` when
  the verifier stops after the first error.

## Reproducing checks locally (no GPU)

```bash
.venv/bin/python -m pytest thinkprm/tests -q
```

Covers prompt formatting, boxed/decision parsing, step rendering, training-data
filtering against gold labels, score aggregation, and the `thinkprm` selection
alias.

## Limitations

- The synthetic-chain filter depends on the quality of the teacher and, for
  Yoruba, of the translated step labels. Spot-check a sample before a long run.
- Only **parallel** verification scaling is implemented (`n_verifications`).
  The paper's **sequential** budget-forcing path is left for future work.
- Off-the-shelf ThinkPRM is English/math-trained; expect weaker step scoring on
  Yoruba and AfriMMLU until the LoRA adapter is trained.

## Citation

```bibtex
@article{khalifa2025,
  title={Process Reward Models That Think},
  author={Muhammad Khalifa and Rishabh Agarwal and Lajanugen Logeswaran and Jaekyeom Kim and Hao Peng and Moontae Lee and Honglak Lee and Lu Wang},
  journal={arXiv preprint arXiv:2504.16828},
  year={2025}
}
```
