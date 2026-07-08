# Yoruba Test-Time Compute Scaling Project Plan

## Summary

Build a reproducible research pipeline and benchmark artifact for an EACL 2027-targeted paper tentatively titled **"When More Thinking Is Not Enough: Test-Time Compute Scaling for Yoruba Reasoning."**

The project focuses strictly on **Yoruba evaluation**. English is used only as a reasoning/pivot language inside Yoruba tasks, not as a standalone benchmark or comparison language.

Core claim to test:

**Test-time compute can improve Yoruba reasoning, but its gains are constrained by candidate-generation quality, reasoning-language choice, verifier selection, tokenization cost, and task type.**

Primary research questions:

- **RQ1:** Does increasing test-time compute improve Yoruba task performance under matched token/FLOP/cost budgets?
- **RQ2:** Is the Yoruba bottleneck candidate generation or candidate selection?
- **RQ3:** Does Yoruba performance improve when models reason in Yoruba, English, or via translate-to-English pivoting?
- **RQ4:** Can smaller open models with TTC match larger greedy models on Yoruba tasks?
- **RQ5:** Why does TTC help or fail across Yoruba math, QA, reading comprehension, and instruction following?

## Key Implementation

- **Benchmark layer**
  - Use Yoruba examples from IrokoBench/AfriMGSM, AfriMMLU, AfriQA, YORC/NaijaRC, and CL-IFEval-style instruction following.
  - Normalize all datasets to JSONL with: `id`, `task`, `language`, `question`, `choices`, `gold_answer`, `answer_type`, `source_dataset`, `requires_yoruba_output`, and `metadata`.
  - Add a targeted manually validated Yoruba subset for quality checks, cultural/contextual issues, and judge/verifier calibration.

- **Inference layer**
  - Use open models for reproducibility: Qwen, Llama, Gemma, and optionally Yoruba/African-adapted models if compatible.
  - Implement direct greedy, Yoruba CoT, English CoT with Yoruba final answer, translate-to-English pivoting, self-consistency, Best-of-N verifier reranking, and budget forcing.
  - Sweep `N = 1, 4, 8, 16, 32, 64`, with matched token budgets for direct comparisons.

- **Evaluation layer**
  - Report accuracy, exact match/F1 where appropriate, Yoruba answer-language compliance, generated tokens, latency, and estimated cost.
  - Add `pass@N`, `select@N`, oracle selection, tokenizer-cost analysis, language switching, wrong-language answers, and tokens-per-correct-answer.

## Experiment Plan

- **E1: Yoruba matched-budget TTC scaling**
  - Run all inference methods only on Yoruba evaluation tasks.
  - Compare greedy, CoT, self-consistency, BoN, verifier reranking, and budget forcing under matched budgets.
  - No standalone English benchmark runs are included.

- **E2: Generation versus selection bottleneck**
  - For each `N`, compute `pass@N`, selected accuracy, oracle accuracy, and verifier-selection error.
  - Use this to distinguish candidate-generation failure from verifier/selection failure.

- **E3: Reasoning-language intervention**
  - On the same Yoruba tasks, compare Yoruba reasoning, English reasoning, translate-to-English pivoting, and unconstrained/mixed reasoning.
  - Require final answers in Yoruba where the task is Yoruba-facing.

- **E4: Model size versus inference compute**
  - Compare small model + TTC against medium/large greedy models.
  - Report accuracy-per-cost and tokens-per-correct-answer.

- **E5: Error and cost analysis**
  - Manually inspect a stratified failure sample.
  - Label failures as reasoning failure, Yoruba understanding failure, translation artifact, cultural knowledge failure, wrong-language answer, verifier mistake, or formatting/evaluation issue.

## Assumptions

- Yoruba is the only evaluated language in Paper 1.
- English appears only inside intervention methods: English CoT, translation pivoting, and verifier prompts where needed.
- No English benchmark tables, English scaling curves, or English-vs-Yoruba headline comparisons will be included.
- Open models are the main experimental target; API models may be used only for auxiliary translation/judging if explicitly documented.
- Human validation is targeted rather than a full new benchmark creation effort.
