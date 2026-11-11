## TRACE-RAG: Claim-Level Hallucination Detec- tion in Retrieval-Augmented Generation Systems

## Abstract

Retrieval-Augmented Generation (RAG) systems ground answers in retrieved passages, yet retrieval does not guarantee that every generated statement is supported. A single answer routinely mixes supported facts, unsupported details and outright contradictions, so response-level judgments hide where errors occur. TRACE-RAG is a claim-level verification module placed between answer generation and final delivery. It (1) decomposes the answer into atomic claims with entities, quantities, dates and inter-claim dependencies; (2) compares each claim with several retrieved passages and predicts supported / not supported / unclear for every claim–passage pair; (3) aggregates passage-level evidence into facet-level and claim-level quantities that separate support, contradiction, source agreement and coverage; and (4) uses calibrated thresholds to decide whether to accept, reject, qualify, retrieve more evidence, or abstain. A single backbone LLM (Granite 8B) serves both extraction and verification, which keeps the system compact and lets claim and evidence representations live in the same space. The central design principle is that insufficient evidence is not the same as falsehood, and that agreeing passages do not automatically establish every detail of a claim. We outline the architecture, the aggregation mathematics, the decision controller, and an evaluation plan on RAGTruth, FEVER-derived data, SciFact and WICE.

## Introduction

Why claim-level verification is needed. Long answers contain many facts of unequal reliability. Work on FActScore, SAFE and VeriScore showed that decomposing a generation into atomic facts and checking each one turns fac- tuality into a fine-grained, measurable target rather than a single binary label. HalluMeasure adds that claims should be extracted from the full response, not sentence by sentence, because one claim can depend on information spread across sentences, and that per-claim classification outperforms batch classification.

Why extraction quality matters. Poor decomposition propagates directly into poor hallucination detection. Good claims are atomic, decontextualized (no dangling pronouns), faithful (nothing added), and complete (nothing important dropped). Production-style verifiers further normalize numbers and dates, be- cause surface-form mismatches break retrieval, and keep character spans so each claim can be traced to its location in the answer.

Why one claim needs many passages. A single claim often has several facets, and different passages cover different facets. Evidence graph models such as GEAR and KGAT showed that letting evidence pieces interact improves FEVER scores, which motivates comparing each claim against several passages jointly rather than one at a time.


Why aggregation must be more than voting. Passages can agree with each other and still fail to establish a specific detail. Unanimous support for two facets says nothing about a third. A facet may look supported but depend on an ancestor claim that failed. A good aggregator keeps these cases apart mathematically.

Why decisions must be calibrated and conservative. Correcting every statement introduces new errors: in RAG settings many statements are already correct, so the better pattern is verify first, then correct only what is judged false. The decision layer must therefore distinguish contradiction (reject or correct), conflict (qualify with both sides), and missing evidence (retrieve more or abstain).

Together these points motivate a pipeline where each stage has a clearly defined responsibility and a measurable output.

## 1. Research Objective

RAG systems retrieve external passages before generating an answer. However, retrieved information does not guarantee that every generated statement is factually supported. A response may contain unsupported details, contradictions, or sentences that mix correct and incorrect facts.

The project develops a claim-level verification module placed between answer generation and final response delivery. It will decompose generated answers into atomic claims, align each claim with retrieved evidence, assess entailment and contradiction, estimate evidence sufficiency and uncertainty, and recommend whether to keep, qualify, correct, retrieve more evidence, or abstain.

## 2. Architecture Overview

The system has four parts:

- 1. Atomic Claim and Dependency Extractor

- 2. Multi-Evidence Entailment Network

- 3. Conflict and Coverage Aggregator

- 4. Uncertainty-Aware Decision Controller

## 2.1 Atomic Claim and Dependency Extractor

Breaks each response into atomic claims and records claim text, entities, relations, quantities, dates, source span, and dependencies between claims. Example: ”The bridge was built in 1985 and designed by Engineer A” becomes two separately verified claims: (a) the bridge was built in 1985; (b) the bridge was designed by Engineer A.


Implementation. The Granite 8B model is fine-tuned (distilled from a stronger teacher's annotations) into an abstractive proposition segmentation (APS) model, following the teacher–student recipe in which a large model labels data and a smaller model learns the segmentation policy. Output is constrained JSON with a fixed schema:

- claim text and character span in the source answer

- subject, relation, object

- normalized entities, dates and numbers

- modality marker (asserted, hedged, hypothetical)

- typed dependency edges to other claims: entity coreference, temporal constraint, numerical constraint, causal dependency, entailment, conflict

Rules. Resolve pronouns, split conjunctions that join independent predicates, deduplicate, preserve original order, filter opinions and unverifiable noise, and split recursively until a claim has one main relation and at most one time, number or modality constraint.

## 2.2 Multi-Evidence Entailment Network

One claim usually needs several passages. The verifier uses the same Granite backbone:

- 1. Encode the claim together with the question context and take the last token's last hidden state, ℎ𝑐.

- 2. Encode each retrieved passage 𝑝𝑖 (with the claim as conditioning where useful) and take its last-token last-layer hidden state, ℎ𝑖.

- 3. Score each pair with a bilinear interaction: 𝑧𝑖 = ℎ⊤ 𝑐 𝑊ℎ𝑖 (one 𝑊 per class, or a low-rank factorized 𝑊 for efficiency).

- 4. Apply a softmax to obtain three probabilities per claim–passage pair: supported, not supported (contradicts), and unclear / insufficient evidence, written (𝑠𝑖, 𝑟𝑖, 𝑛𝑖) on the probability simplex.

Passage identifiers are retained so every verdict traces to its evidence. Because hidden states can be precomputed (for example with vLLM), many claim–passage pairs can be scored cheaply, and only the small bilinear head needs training.

## 2.3 Conflict and Coverage Aggregator

Given. A claim 𝐶 decomposed into facets 𝐹 = {𝑓1,…, 𝑓𝑘} (from the extractor, with dependency edges). Passages 𝑝1,…, 𝑝𝑛 with per-passage verdict masses: support 𝑠𝑖, refute 𝑟𝑖, not-support/neutral 𝑛𝑖 (soft probabilities from the verifier). Optional source-reliability weights 𝑤𝑖.


Stage 1 — facet pooling. For each facet 𝑓, over the covering passages 𝐶𝑓 = { 𝑖 ∶ 𝑓 is covered by 𝑝𝑖 }:

- eff𝑓 = 𝑆𝑓 + 𝛾𝑀𝑓, where 𝑀𝑓 is partial-support mass and 𝛾 ≈ 0.5 is a discount

- If 𝐶𝑓 is empty then eff𝑓 = 0. No evidence is not weak support; it is zero support.

## Stage 2 — facet verdicts.

- contradicted(𝑓) ⟺ 𝑅𝑓 ≥ 𝜏𝑟 and 𝑅𝑓 > 𝑆𝑓

- established(𝑓) ⟺ eff𝑓 ≥ 𝜏𝑠, and not contradicted(𝑓), and |𝐶𝑓| ≥ 1, and every ancestor of 𝑓 is established. The last conjunct is the coherent- factuality gate.

## Stage 3 — claim-level quantities.

- support(𝐶) = exp( ∑𝑓 ln max(eff𝑓, 𝜀)). A geometric mean acts as a weakest-link score: one unestablished facet sinks the claim. 1 𝑘

- contradiction(𝐶) = max𝑓 𝑅𝑓

- coverage(𝐶) = ∑𝑓 1[established(𝑓)] 1 𝑘

- conflict intensity = is two-sided mass. 2 𝑆tot𝑅tot 𝑆tot + 𝑅tot . A harmonic mean, high only when there

- source agreement = 1 − passages. 1 means unanimous, 0 means uniform over the three labels. 𝐻(𝑝) ln 3 , where 𝑝 is the label distribution over

## Stage 4 — decision (summary).

Condition some facet contradicted and

𝑆tot > 0

coverage = 1 and support(𝐶) ≥ calibrated ̂𝜏𝑠 otherwise

Decision QUALIFY_CONFLICT: cite both sides and type the conflict SUPPORTED (error rate ≤ 𝛼, calibrated)

ABSTAIN, reporting the missing facets


## Three distinctions the aggregator keeps separate:

- 1. Agreement ≠ support. 𝑛 passages can unanimously agree that evidence is insufficient. Check |𝐶𝑓| and the mass distribution, not just consensus.

- 2. Agreement ≠ coverage. Unanimous support on facets {𝑓1, 𝑓2} says nothing about 𝑓3. The geometric mean over all facets enforces this.

- 3. Support ≠ establishment. A facet with effective support mass whose ancestor facet failed is blocked, not established

## 2.4 Uncertainty-Aware Decision Controller

The controller converts aggregator outputs into one of five actions: accept, reject, qualify, retrieve more evidence, abstain. Its guiding rule is that an ”insufficient evidence” verdict must never be confused with a claim being false.

Inputs. support(𝐶), contradiction(𝐶), coverage(𝐶), conflict intensity, source agreement, the list of missing or blocked facets, dependency status of the claim's ancestors, and the remaining retrieval budget.

## Calibration.

- Calibrate the verifier's probabilities on a held-out validation set (for exam- ple temperature scaling) and report expected calibration error.

- Choose ̂𝜏𝑠 on the validation set so that among claims marked SUPPORTED the empirical error rate is at most a target 𝛼. A split-conformal or risk- controlling procedure gives a finite-sample guarantee under exchangeability of validation and test data.

- Choose 𝜏𝑟 similarly to control false rejections

Response-level handling. Decisions are applied in dependency order. If an ancestor claim is rejected or unresolved, its dependents are blocked rather than accepted. Correction is selective: only rejected claims are rewritten, so already-supported content is never disturbed. The final response is rebuilt from accepted, qualified and corrected claims, each carrying passage identifiers as citations.

The thresholds, the five-action policy and the retrieval-budget rule are proposed design choices and should be tuned and ablated on validation data.

## 3. Design Rationale

Components of claim-level verification are often studied separately: decompo- sition, evidence comparison, aggregation and abstention. TRACE-RAG treats them as one pipeline driven by a single backbone. The same 8B model is adapted for segmentation and for evidence comparison, so claim and passage represen- tations share one space, the verifier is a lightweight bilinear head over cached


hidden states, and the aggregator and controller are transparent, auditable func- tions rather than additional black boxes. Facet-level pooling with an ancestor gate and a weakest-link score gives an explicit mechanism for the cases where consensus is misleading.

## 4. Evaluation Plan

| Module                                        | Metrics                             | Data                    |
| --------------------------------------------- | ----------------------------------- | ----------------------- |
| Claim                                         | Claim precision / recall / F1;      | Annotated RAGTruth      |
| extraction                                    | atomicity, decontextualization,     | answers;                |
|                                               | faithfulness; omission rate         | FEVER-derived atomic    |
|                                               |                                     | claims                  |
| Entities, dates,                              | Entity/relation P/R/F1; exact and   | RAGTruth,               |
| quantities                                    | normalized match for dates and      | FEVER-derived,          |
|                                               | numbers                             | SciFact                 |
| Multi-                                        | Accuracy, macro-F1, FEVER Score,    | FEVER, SciFact          |
| evidence                                      | Evidence F1 vs. a pairwise-only     |                         |
| verifier                                      | baseline                            |                         |
| Hallucination                                 | Span- and claim-level precision,    | RAGTruth                |
| detection                                     | recall, F1, AUROC, AUPRC            |                         |
| Aggregator Facet-coverage accuracy;           |                                     | Constructed challenge   |
|                                               | agree-yet-incomplete case detection | slice                   |
| Dependencies Edge precision / recall / F1 per |                                     | Small custom            |
|                                               | type; cycle rate; contradiction     | annotation              |
|                                               | flagging                            |                         |
| Decision                                      | Error rate among accepted claims    | Held-out validation and |
| controller                                    | vs. 𝛼; abstention rate;            | test splits             |
|                                               | false-rejection rate; ECE;          |                         |
|                                               | risk–coverage curve                 |                         |
| Information                                   | Pairwise precedence accuracy /      | Small custom            |
| order                                         | Kendall's tau                       | annotation              |

Baselines. Sentence-splitting plus pairwise NLI; LLM prompting for atomic claims without a schema; single-score voting aggregation; correct-everything versus verify-then-correct.

Release gates. Claim F1 clearly above the sentence-splitting baseline; near- perfect normalization of common date and number formats; cycle rate near zero on acyclic relation types; accepted-claim error rate at or below 𝛼; and no precision gains achieved by silently dropping hard claims (omission rate is the primary extraction blocker).

Cautions from published work. A reproducible end-to-end FEVER verifier built from hybrid retrieval and reranking reached a FEVER Score of only about 0.17 on the development set, a reminder that full-stack performance depends


heavily on retrieval quality. Published numbers for extraction, grounding and dependency modeling come from different corpora and annotation schemes and should not be merged into one leaderboard. No standard benchmark exists for claim–claim edges, so dependency evaluation needs custom annotations.

## 5. Limitations and Risks

- Extraction errors (missed claims, merged facts, unresolved references) propagate to every later stage.

- Facet-to-passage coverage mapping must itself be predicted and can be noisy.

- Last-token hidden states may lose detail on long passages; chunking and span-level evidence selection may be needed.

- Calibration guarantees assume validation and deployment data are similar.

- Dependency edge prediction has little prior benchmark support and needs its own annotation effort.

- Verification checks support relative to retrieved evidence, not absolute truth.

## Conclusion

TRACE-RAG proposes a compact, auditable path from generated answer to verified response: extract atomic, dependency-aware claims; compare each against multiple passages with a shared-backbone bilinear verifier; aggregate at the facet level so that agreement, support, coverage and establishment are never conflated; and decide through calibrated thresholds that treat missing evidence differently from falsehood. If the evaluation confirms the design, it would offer claim-level hallucination localization with traceable evidence, controllable error rates, and a principled option to retrieve more or abstain. Its value must be established through a current literature review and the experiments outlined above.

## Key References

- Min et al., 2023 — FActScore

- Wei et al., 2024 — SAFE / LongFact

- Akbar et al., 2024 — HalluMeasure

- Ullrich et al., 2025 — Claim extraction and FEVERFact

- Hosseini et al., 2024 — Abstractive proposition segmentation (teacher– student)

- Dou et al., 2026 — Atomic claim extraction rules (coreference, dedup, order)


- Luo, 2024 — Claim decomposition with normalization and two-stage retrieval

- Zhou et al., 2019 (GEAR); Liu et al., 2019 (KGAT) — Multi-evidence graph verification

- Li et al., 2024 — RAC: verify, then correct only false statements

- Wang et al., 2026 (FreeGuard) — Typed claim dependencies

- Wu et al., 2026 (DagFC) — Dependency-aware verification

- Xie et al., 2026 (LoReFact) — Content vs. logic factuality

- Cen et al., 2026 (AtomEval) — Truth-critical modifiers

- Malashin et al., 2026 — Schema-constrained, span-grounded extraction
