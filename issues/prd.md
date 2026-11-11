# TRACE-RAG v2 — Research Paper + Granite Pipeline PRD

Supersedes the WICE-skeleton PRD (that milestone is complete: verifier →
aggregator → controller runs end-to-end on WICE). This PRD covers the next
milestone: the deliverable is a **research paper** backed by a working
pipeline, with experiments on FEVER + SciFact + WICE.

## Problem Statement

TRACE-RAG (`context.md`, `docs/paper.md`) proposes claim-level hallucination
detection for RAG: extract atomic claims, verify each against multiple
passages, aggregate evidence without conflating agreement/support/coverage,
and decide via calibrated thresholds that separate contradiction from
insufficient evidence.

The WICE skeleton proved the verifier→aggregator→controller path runs, but:

- **Novelty is threatened.** The 2026 literature (see `docs/competitions.md`)
  covers every pipeline component: FreeGuard already implements SROM claim
  extraction, the same six typed dependency edges, claim-graph risk
  propagation, three-way evidence verdicts, and thresholded governance;
  RT4CHART does claim decomposition → 3-way labels on RAGTruth; ToE does
  evidence-insufficient re-decomposition; BCEA does budgeted conformal
  evidence acquisition. The proposal's surviving differentiators are the
  **facet-level evidence algebra** (weakest-link + ancestor gate),
  **risk-controlled α-guarantees on accepted claims**, the
  **insufficient≠false mass-channel semantics**, and a **cheap trained
  verifier** over frozen hidden states.
- **No claim extractor exists for real answers.** WICE provides claims; a
  paper needs extraction from generated answers (APS).
- **WICE alone cannot supervise the insufficient≠false distinction** — its
  `not_supported` conflates refute+insufficient. SciFact's
  SUPPORTS/REFUTES/NOINFO is the only dataset among our targets with a true
  three-way label.
- **Compute is constrained**: no local GPU; Kaggle 2×T4 (~30h/week quota);
  hidden-state extraction is the expensive step (must run on Kaggle, vLLM
  preferred, HF fallback).

## Solution

One base model (`granite-4.1-8b`, swappable via `--model`) plays both roles:

1. **APS extractor** — a LoRA adapter (QLoRA, r≈32, all linear layers)
   fine-tuned to emit abstractive propositions in the Gemma-APS grouped
   format (`<s> prop₁ … propₙ </s>`). Distilled from an open teacher
   (Qwen3-30B-A3B Q4 on 2×T4 via vLLM, or Gemma-APS for bulk generic
   segmentation). Output is *only* propositions — slot-fill, normalization,
   and edge inference reuse the existing `claim_extractor` adapter pipeline.
   Facets stay claim-level (k=1); dependency edges stay rule-based.
2. **Verifier** — frozen base-model last-layer hidden states + trained
   low-rank bilinear head (existing `VerdictModel`), with a **rank sweep**
   r ∈ {16…256} and a zero-parameter nearest-centroid baseline, so head
   capacity is matched to dataset size empirically.
3. **Aggregator + controller** — unchanged (§2.3 math, k=1 on these
   datasets); decision threshold τ̂_s selected on validation for
   accepted-error ≤ α.

Evaluation on **FEVER + SciFact + WICE** (all provide claim+evidence+labels
natively). RAGTruth end-to-end and question-context conditioning are deferred
to a later milestone.

Paper positioning: "audit-grade claim verification with statistical
guarantees" — lead with the evidence algebra and α-controlled acceptance;
cite FreeGuard/RT4CHART/ToE/BCEA/LoReFact/DagFC/MARCH/HalluGuard explicitly
(`docs/competitions/`). Rename recommended: "TRACE-RAG" collides with two
2026 papers.

## User Stories

1. As a researcher, I want to run the existing WICE pipeline unchanged, so
   that the v1 results remain reproducible.
2. As a researcher, I want SciFact claims fed through the same verifier head
   with SUPPORTS/REFUTES/NOINFO mapped to s/r/n channels, so that the
   insufficient≠false distinction is actually supervised.
3. As a researcher, I want FEVER claims verified against their gold evidence
   sentences (verdict accuracy, not full-pipeline FEVER Score), so that a
   third benchmark corroborates the head without a retriever dependency.
4. As a researcher, I want the verifier head's rank r swept on validation
   with macro-F1 and parameter count reported, so that capacity is matched
   to data size rather than assumed.
5. As a researcher, I want a zero-parameter nearest-centroid baseline in
   hidden space, so that the bilinear head's value is demonstrated, not
   assumed.
6. As a researcher, I want ECE, risk–coverage curves, and the full α-sweep
   reported per dataset, with α=0.10 as the headline operating point chosen
   on validation, so that the guarantee claim is auditable.
7. As a researcher, I want the aggregator's challenge slice (constructed
   agree-yet-incomplete / empty-coverage / ancestor-gate cases) run
   GPU-free, so that the facet algebra's claimed distinctions are
   demonstrated empirically even at k=1.
8. As a researcher, I want an APS baseline check first — run Gemma-APS or a
   prompted open model on the target corpus and measure atomicity /
   faithfulness / omission — so that the Granite fine-tune is justified by
   measured failure, not assumed.
9. As a researcher, I want APS training data generated by an open teacher
   (Qwen3-30B-A3B Q4 on 2×T4, or Gemma-APS for bulk segmentation), formatted
   exactly as `<s> propositions </s>`, so that distillation is a pure
   seq2seq imitation task.
10. As a researcher, I want distillation data selected by student-need
    (LLKD/SRD-style: teacher confidence × student failure), so that a few
    hundred to few thousand examples suffice on a T4 budget.
11. As a researcher, I want the APS LoRA trained with QLoRA (4-bit base,
    r=32, all linear layers, Unsloth or HF fallback), so that training fits
    on a single T4.
12. As a researcher, I want the encoder pass for hidden-state extraction to
    run with the LoRA adapter disabled (base weights), so that APS
    retraining never invalidates the verifier's cached states.
13. As a researcher, I want `GraniteAPSExtractor.extract(text)` implemented
    against the trained checkpoint, emitting `ExtractionResult` via the
    existing adapter parse path, so that downstream code is untouched.
14. As a researcher, I want extraction quality measured (claim P/R/F1 vs.
    sentence-split baseline, atomicity, decontextualization, faithfulness,
    omission rate) on whatever labeled data is feasible, so that the APS
    component is evaluated as a component, not just end-to-end.
15. As a researcher, I want all configs via argparse/`python -m` entry
    points and all checkpoints storing model id, adapter path, rank,
    layers, thresholds, and dataset metadata, so that every number in the
    paper is reproducible from a command.
16. As a researcher, I want CPU-only pytest coverage for every pure module
    touched (label mapping for SciFact/FEVER, rank sweep, centroid
    baseline, aggregator slice), so that nothing GPU-dependent blocks CI.
17. As a researcher, I want the WICE `not_supported` conflation,
    FEVER's gold-evidence caveat, and SciFact's NOINFO mapping documented
    in code and in the paper's limitations, so that label semantics are
    never silently conflated.
18. As a researcher, I want the paper's related-work section built from
    `docs/competitions/` + `docs/supporting/`, so that novelty claims are
    positioned against the actual 2026 landscape.

## Implementation Decisions

### Modules to build/modify

**`src/entailment_network/`** (mostly exists)

- `verifier_common.py`: add `NearestCentroidBaseline` (class-mean hidden
  states, cosine similarity — zero trained params) alongside `VerdictModel`;
  add `--r` rank sweep support in train/eval; add label mappers for
  SciFact (`SUPPORTS→s`, `REFUTES→r`, `NOINFO→n`) and FEVER
  (`SUPPORTED→s`, `REFUTED→r`, `NEI→n`) producing the same `[B,N,3]` chunk
  tensors as WICE.
- `data.py`: dataset adapters for `allenai/scifact` and FEVER (claims +
  evidence sentences → same row schema as WICE: claim, evidence list,
  supporting positions, split). Lazy `datasets` import preserved.
- `eval_pipeline.py`: extend to `--dataset {wice,scifact,fever}` with
  per-dataset label mapping; report macro-F1/accuracy/AUROC/ECE/
  risk–coverage/accepted-error-vs-α/abstention rate per dataset; α-sweep
  flag (`--alphas`) emitting the full curve; headline α=0.10.
- `train.py`/`infer.py`: unchanged interfaces; `--r` already exists — wire
  the sweep.
- `aggregator.py`, `calibration.py`, `controller.py`: unchanged; add the
  constructed challenge-slice generator (synthetic chunk-probability
  tensors → agree-yet-incomplete, empty-coverage, ancestor-blocked cases)
  as a pure function + eval entry, GPU-free.

**`src/claim_extractor/`**

- `granite_aps.py`: implement `extract()` — load base + LoRA adapter,
  prompt in grouped-APS format, generate `<s>…</s>` propositions, parse via
  `gemma_aps_adapter.parse_propositions`, then existing slot-fill/edge
  inference → `ExtractionResult`. Adapter disabled during any hidden-state
  extraction pass.
- `aps_distill.py` (new): teacher-label generation — stream corpus
  (RAGTruth answers if accessible; otherwise synthetic long-form answers
  from the open teacher on scientific/technical topics + WICE/FEVER/SciFact
  claims embedded in generated context) → teacher emits grouped-APS
  propositions → JSONL dataset. Optional LLKD/SRD-style selection pass
  (teacher confidence × student failure on a small probe set).
- Training script `train_aps.py` (new, `python -m` entry): QLoRA SFT on the
  JSONL via Unsloth (preferred) or HF `trl`/`peft` fallback; saves adapter
  + config; flags `--model`, `--r`, `--epochs`, `--lr`, `--limit`.
- `atomic_claim_extractor.py`, `gemma_aps_adapter.py`, `edge_finder.py`:
  unchanged (reuse as-is).

**Cross-cutting**

- Backbone id stays a flag everywhere; default `ibm-granite/granite-4.1-8b`
  **pending HF verification** (fallback `granite-3.3-8b`); model-agnostic
  interfaces preserved since the backbone may change.
- All new datasets lazy-imported; CPU tests never touch `datasets`/
  `transformers`/`vllm`/`unsloth`/`peft`.
- Naming: paper/system rename TBD (TRACE-RAG collides with two 2026
  papers) — decide before writing, not before coding.

### Interfaces / contracts

- `GraniteAPSExtractor.extract(text) -> ExtractionResult` — same contract
  as `OllamaExtractor`/`GemmaAPSAdapter`; `from_checkpoint(path)` loads
  base+adapter.
- APS training data format: JSONL `{text, propositions}` where
  `propositions` is the grouped `<s>…</s>` string — pure imitation target.
- Dataset row contract unchanged: `{claim, evidence: [sentences],
  pos_labels, split, meta}`; SciFact/FEVER adapters fill `pos_labels` from
  rationale/evidence sentences.
- Label mapping (single source of truth in `verifier_common`):
  class0=supported, class1=partially/refute-mass channel per dataset,
  class2=not_supported/NOINFO/NEI — document per-dataset semantics in code;
  SciFact NOINFO is the only true insufficient-evidence supervision.
- Checkpoint stores: backbone id, adapter path+rank, layers, τ̂_s/τ_r, α,
  dataset, extraction metadata.

### Technical clarifications

- Compute: APS LoRA fits 1×T4 (QLoRA r=32 ≈ 10–14GB); teacher inference
  Qwen3-30B-A3B Q4 needs 2×T4 TP=2 (~19GB weights + KV); hidden-state
  extraction for three datasets runs on Kaggle via vLLM (`--tp 2`,
  `--dtype float16`), HF fallback automatic.
- APS baseline-first: measure stock APS (Gemma-APS or prompted open model)
  on target corpus before committing T4-hours to the fine-tune.
- α headline = 0.10 on validation; full sweep reported.
- Head capacity: rank sweep + centroid baseline; report param-count vs.
  macro-F1.
- Deadline: exists but date TBD — scope v1 = verifier on 3 datasets +
  aggregator slice + APS fine-tune attempt; RAGTruth end-to-end = v2.

## Testing Decisions

**What makes a good test**: external behavior only — synthetic tensors,
label arrays, claim lists, JSONL rows; assert on outputs/thresholds/actions.
No mocking internals, no GPU, no dataset downloads (tiny fixtures inline).
Prior art: existing `tests/` CPU pytest suite for aggregator/calibration/
controller/edge_finder/adapter.

**Modules tested**:

- `verifier_common`: SciFact/FEVER label mappers on fixture rows;
  `macro_f1`; centroid baseline equals-or-near bilinear head on separable
  synthetic features; `parse_label` coverage.
- `data` adapters: fixture JSONL → row schema; split handling; pos-label
  alignment for evidence sentences.
- `train`/`eval` arg parsing: `--dataset`, `--r`, `--alphas` smoke via
  `--help` (no GPU).
- `aggregator` challenge slice: constructed cases produce expected
  quantities/actions (agree-yet-incomplete → not SUPPORTED; empty C_f →
  eff_f=0; ancestor-blocked facet → not established).
- `calibration`: α-sweep returns monotone risk–coverage points; selected
  τ̂_s achieves accepted-error ≤ α on tuning split.
- `granite_aps`/`aps_distill`: grouped-APS parse round-trip on fixture;
  malformed generation falls back gracefully; JSONL writer/reader.
- `train_aps`: arg-parse smoke only (training itself untested on CPU).

## Out of Scope

- RAGTruth end-to-end extraction eval (v2); question-context conditioning
  (deferred, `meta` preserved).
- Dependency-edge *training* (rule-based `edge_finder` only); facet
  granularity below claim (k=1 everywhere).
- Retrieve-more and correction actions (stubbed → ABSTAIN); retrieval
  budgets.
- Full fine-tuning of any model; frontier-API teachers (open weights only).
- FEVER Score / evidence-set metrics requiring a retriever (verdict
  accuracy on given evidence only).
- Claim–claim edge benchmarks (no dataset provides supervision).
- Baselines beyond: sentence-split+pairwise-NLI framing, centroid
  baseline, rank sweep, voting-aggregation ablation (aggregator slice).

## Further Notes

- **Repositioning is mandatory**: `docs/competitions/` shows FreeGuard owns
  typed claim graphs + governance; the paper must lead with evidence
  algebra + α-guarantees and cite the field honestly. Working title needs
  changing (TRACE-RAG name collision).
- APS baseline check gates the fine-tune: if stock APS is good enough on
  the corpus, skip QLoRA and say so — that *is* a result.
- Distillation regime is few-shot by necessity (T4 budget + no annotation
  team): lean on `docs/supporting/` methods (LLKD/SRD selection,
  skill-aware targeting, CoD counterfactuals for the verifier head if
  needed).
- WICE remains the largest extraction cost; SciFact/FEVER are small —
  budget Kaggle quota accordingly; `--limit` smoke tests first.
- `context.md` stays the research source of truth; this PRD is the current
  milestone only.
