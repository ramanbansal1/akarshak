TRACE-RAG: Claim-Level Hallucination Detection in Retrieval-Augmented Generation Systems
This is a research proposal, not an established model. Novelty must be validated through a current literature review and experiments.
Abstract
Retrieval-Augmented Generation (RAG) systems ground answers in retrieved passages, yet retrieval does not guarantee that every generated statement is supported. A single answer routinely mixes supported facts, unsupported details and outright contradictions, so response-level judgments hide where errors occur. TRACE-RAG is a claim-level verification module placed between answer generation and final delivery. It (1) decomposes the answer into atomic claims with entities, quantities, dates and inter-claim dependencies; (2) compares each claim with several retrieved passages and predicts supported / not supported / unclear for every claim–passage pair; (3) aggregates passage-level evidence into facet-level and claim-level quantities that separate support, contradiction, source agreement and coverage; and (4) uses calibrated thresholds to decide whether to accept, reject, qualify, retrieve more evidence, or abstain. A single backbone LLM (Granite 8B) serves both extraction and verification, which keeps the system compact and lets claim and evidence representations live in the same space. The central design principle is that insufficient evidence is not the same as falsehood, and that agreeing passages do not automatically establish every detail of a claim. We outline the architecture, the aggregation mathematics, the decision controller, and an evaluation plan on RAGTruth, FEVER-derived data, SciFact and WICE.
Introduction
Why claim-level verification is needed. Long answers contain many facts of unequal reliability. Work on FActScore, SAFE and VeriScore showed that decomposing a generation into atomic facts and checking each one turns factuality into a fine-grained, measurable target rather than a single binary label. HalluMeasure adds that claims should be extracted from the full response, not sentence by sentence, because one claim can depend on information spread across sentences, and that per-claim classification outperforms batch classification.
Why extraction quality matters. Poor decomposition propagates directly into poor hallucination detection. Good claims are atomic, decontextualized (no dangling pronouns), faithful (nothing added), and complete (nothing important dropped). Production-style verifiers further normalize numbers and dates, because surface-form mismatches break retrieval, and keep character spans so each claim can be traced to its location in the answer.
Why one claim needs many passages. A single claim often has several facets, and different passages cover different facets. Evidence graph models such as GEAR and KGAT showed that letting evidence pieces interact improves FEVER scores, which motivates comparing each claim against several passages jointly rather than one at a time.
Why aggregation must be more than voting. Passages can agree with each other and still fail to establish a specific detail. Unanimous support for two facets says nothing about a third. A facet may look supported but depend on an ancestor claim that failed. A good aggregator keeps these cases apart mathematically.
Why decisions must be calibrated and conservative. Correcting every statement introduces new errors: in RAG settings many statements are already correct, so the better pattern is verify first, then correct only what is judged false. The decision layer must therefore distinguish contradiction (reject or correct), conflict (qualify with both sides), and missing evidence (retrieve more or abstain).
Together these points motivate a pipeline where each stage has a clearly defined responsibility and a measurable output.
1. Research Objective
RAG systems retrieve external passages before generating an answer. However, retrieved information does not guarantee that every generated statement is factually supported. A response may contain unsupported details, contradictions, or sentences that mix correct and incorrect facts.
The project develops a claim-level verification module placed between answer generation and final response delivery. It will decompose generated answers into atomic claims, align each claim with retrieved evidence, assess entailment and contradiction, estimate evidence sufficiency and uncertainty, and recommend whether to keep, qualify, correct, retrieve more evidence, or abstain.
2. Architecture Overview
The system has four parts:
    1. Atomic Claim and Dependency Extractor
    2. Multi-Evidence Entailment Network
    3. Conflict and Coverage Aggregator
    4. Uncertainty-Aware Decision Controller
2.1 Atomic Claim and Dependency Extractor
Breaks each response into atomic claims and records claim text, entities, relations, quantities, dates, source span, and dependencies between claims. Example: "The bridge was built in 1985 and designed by Engineer A" becomes two separately verified claims: (a) the bridge was built in 1985; (b) the bridge was designed by Engineer A.
Implementation. The Granite 8B model is fine-tuned (distilled from a stronger teacher's annotations) into an abstractive proposition segmentation (APS) model, following the teacher–student recipe in which a large model labels data and a smaller model learns the segmentation policy. Output is constrained JSON with a fixed schema:
    • claim text and character span in the source answer
    • subject, relation, object
    • normalized entities, dates and numbers
    • modality marker (asserted, hedged, hypothetical)
    • typed dependency edges to other claims: entity coreference, temporal constraint, numerical constraint, causal dependency, entailment, conflict
Rules. Resolve pronouns, split conjunctions that join independent predicates, deduplicate, preserve original order, filter opinions and unverifiable noise, and split recursively until a claim has one main relation and at most one time, number or modality constraint.
2.2 Multi-Evidence Entailment Network
One claim usually needs several passages. The verifier uses the same Granite backbone:
    1. Encode the claim together with the question context and take the last token's last hidden state, h_c.
    2. Encode each retrieved passage p_i (with the claim as conditioning where useful) and take its last-token last-layer hidden state, h_i.
    3. Score each pair with a bilinear interaction: z_i = h_c^T W h_i (one W per class, or a low-rank factorized W for efficiency).
    4. Apply a softmax to obtain three probabilities per claim–passage pair: supported, not supported (contradicts), and unclear / insufficient evidence, written (s_i, r_i, n_i) on the probability simplex.
Passage identifiers are retained so every verdict traces to its evidence. Because hidden states can be precomputed (for example with vLLM), many claim–passage pairs can be scored cheaply, and only the small bilinear head needs training.
2.3 Conflict and Coverage Aggregator
Given. A claim C decomposed into facets F = {f1, …, fk} (from the extractor, with dependency edges). Passages p1, …, pn with per-passage verdict masses: support s_i, refute r_i, not-support/neutral n_i (soft probabilities from the verifier). Optional source-reliability weights w_i.
Stage 1 — facet pooling. For each facet f, over the covering passages C_f = {i : f is covered by p_i}:
    • S_f = Σ w_i·s_i / Σ w_i
    • R_f = Σ w_i·r_i / Σ w_i
    • eff_f = S_f + γ·M_f, where M_f is partial-support mass and γ ≈ 0.5 is a discount
    • If C_f is empty then eff_f = 0. No evidence is not weak support; it is zero support.
Stage 2 — facet verdicts.
    • contradicted(f) ⟺ R_f ≥ τ_r and R_f > S_f
    • established(f) ⟺ eff_f ≥ τ_s, and not contradicted(f), and |C_f| ≥ 1, and every ancestor of f is established. The last conjunct is the coherent-factuality gate.
Stage 3 — claim-level quantities.
    • support(C) = exp( (1/k) Σ_f ln max(eff_f, ε) ). A geometric mean acts as a weakest-link score: one unestablished facet sinks the claim.
    • contradiction(C) = max_f R_f
    • coverage(C) = (1/k) Σ_f 1[established(f)]
    • conflict intensity = 2·S_tot·R_tot / (S_tot + R_tot). A harmonic mean, high only when there is two-sided mass.
    • source agreement = 1 − H(p) / ln 3, where p is the label distribution over passages. 1 means unanimous, 0 means uniform over the three labels.
Stage 4 — decision (summary).
Condition	Decision
some facet contradicted and S_tot > 0	QUALIFY_CONFLICT: cite both sides and type the conflict
coverage = 1 and support(C) ≥ calibrated τ̂_s	SUPPORTED (error rate ≤ α, calibrated)
otherwise	ABSTAIN, reporting the missing facets

Three distinctions the aggregator keeps separate:
    1. Agreement ≠ support. n passages can unanimously agree that evidence is insufficient. Check |C_f| and the mass distribution, not just consensus.
    2. Agreement ≠ coverage. Unanimous support on facets {f1, f2} says nothing about f3. The geometric mean over all facets enforces this.
    3. Support ≠ establishment. A facet with effective support mass whose ancestor facet failed is blocked, not established
2.4 Uncertainty-Aware Decision Controller
The controller converts aggregator outputs into one of five actions: accept, reject, qualify, retrieve more evidence, abstain. Its guiding rule is that an "insufficient evidence" verdict must never be confused with a claim being false.
Inputs. support(C), contradiction(C), coverage(C), conflict intensity, source agreement, the list of missing or blocked facets, dependency status of the claim's ancestors, and the remaining retrieval budget.
Calibration.
    • Calibrate the verifier's probabilities on a held-out validation set (for example temperature scaling) and report expected calibration error.
    • Choose τ̂_s on the validation set so that among claims marked SUPPORTED the empirical error rate is at most a target α. A split-conformal or risk-controlling procedure gives a finite-sample guarantee under exchangeability of validation and test data.
    • Choose τ_r similarly to control false rejections
Response-level handling. Decisions are applied in dependency order. If an ancestor claim is rejected or unresolved, its dependents are blocked rather than accepted. Correction is selective: only rejected claims are rewritten, so already-supported content is never disturbed. The final response is rebuilt from accepted, qualified and corrected claims, each carrying passage identifiers as citations.
The thresholds, the five-action policy and the retrieval-budget rule are proposed design choices and should be tuned and ablated on validation data.
3. Design Rationale
Components of claim-level verification are often studied separately: decomposition, evidence comparison, aggregation and abstention. TRACE-RAG treats them as one pipeline driven by a single backbone. The same 8B model is adapted for segmentation and for evidence comparison, so claim and passage representations share one space, the verifier is a lightweight bilinear head over cached hidden states, and the aggregator and controller are transparent, auditable functions rather than additional black boxes. Facet-level pooling with an ancestor gate and a weakest-link score gives an explicit mechanism for the cases where consensus is misleading.
4. Evaluation Plan
Module	Metrics	Data
Claim extraction	Claim precision / recall / F1; atomicity, decontextualization, faithfulness; omission rate	Annotated RAGTruth answers; FEVER-derived atomic claims
Entities, dates, quantities	Entity/relation P/R/F1; exact and normalized match for dates and numbers	RAGTruth, FEVER-derived, SciFact
Multi-evidence verifier	Accuracy, macro-F1, FEVER Score, Evidence F1 vs. a pairwise-only baseline	FEVER, SciFact
Hallucination detection	Span- and claim-level precision, recall, F1, AUROC, AUPRC	RAGTruth
Aggregator	Facet-coverage accuracy; agree-yet-incomplete case detection	Constructed challenge slice
Dependencies	Edge precision / recall / F1 per type; cycle rate; contradiction flagging	Small custom annotation
Decision controller	Error rate among accepted claims vs. α; abstention rate; false-rejection rate; ECE; risk–coverage curve	Held-out validation and test splits
Information order	Pairwise precedence accuracy / Kendall's tau	Small custom annotation

Baselines. Sentence-splitting plus pairwise NLI; LLM prompting for atomic claims without a schema; single-score voting aggregation; correct-everything versus verify-then-correct.
Release gates. Claim F1 clearly above the sentence-splitting baseline; near-perfect normalization of common date and number formats; cycle rate near zero on acyclic relation types; accepted-claim error rate at or below α; and no precision gains achieved by silently dropping hard claims (omission rate is the primary extraction blocker).
Cautions from published work. A reproducible end-to-end FEVER verifier built from hybrid retrieval and reranking reached a FEVER Score of only about 0.17 on the development set, a reminder that full-stack performance depends heavily on retrieval quality. Published numbers for extraction, grounding and dependency modeling come from different corpora and annotation schemes and should not be merged into one leaderboard. No standard benchmark exists for claim–claim edges, so dependency evaluation needs custom annotations.
5. Limitations and Risks
    • Extraction errors (missed claims, merged facts, unresolved references) propagate to every later stage.
    • Facet-to-passage coverage mapping must itself be predicted and can be noisy.
    • Last-token hidden states may lose detail on long passages; chunking and span-level evidence selection may be needed.
    • Calibration guarantees assume validation and deployment data are similar.
    • Dependency edge prediction has little prior benchmark support and needs its own annotation effort.
    • Verification checks support relative to retrieved evidence, not absolute truth.
Conclusion
TRACE-RAG proposes a compact, auditable path from generated answer to verified response: extract atomic, dependency-aware claims; compare each against multiple passages with a shared-backbone bilinear verifier; aggregate at the facet level so that agreement, support, coverage and establishment are never conflated; and decide through calibrated thresholds that treat missing evidence differently from falsehood. If the evaluation confirms the design, it would offer claim-level hallucination localization with traceable evidence, controllable error rates, and a principled option to retrieve more or abstain. Its value must be established through a current literature review and the experiments outlined above.
Key References
    • Min et al., 2023 — FActScore
    • Wei et al., 2024 — SAFE / LongFact
    • Akbar et al., 2024 — HalluMeasure
    • Ullrich et al., 2025 — Claim extraction and FEVERFact
    • Hosseini et al., 2024 — Abstractive proposition segmentation (teacher–student)
    • Dou et al., 2026 — Atomic claim extraction rules (coreference, dedup, order)
    • Luo, 2024 — Claim decomposition with normalization and two-stage retrieval
    • Zhou et al., 2019 (GEAR); Liu et al., 2019 (KGAT) — Multi-evidence graph verification
    • Li et al., 2024 — RAC: verify, then correct only false statements
    • Wang et al., 2026 (FreeGuard) — Typed claim dependencies
    • Wu et al., 2026 (DagFC) — Dependency-aware verification
    • Xie et al., 2026 (LoReFact) — Content vs. logic factuality
    • Cen et al., 2026 (AtomEval) — Truth-critical modifiers
    • Malashin et al., 2026 — Schema-constrained, span-grounded extraction
