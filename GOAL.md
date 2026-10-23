RESEARCH PROPOSAL | CLAIM-LEVEL HALLUCINATION DETECTION

# CLAIM-LEVEL HALLUCINATION DETECTION IN RETRIEVAL-AUGMENTED GENERATION SYSTEMS

### An Entailment-Based Verification Module

Research concept, proposed architecture, implementation plan, evaluation strategy, and reading list

Proposed model concept: TRACE-RAG — Triangulated Reliability Assessment of Claims and Evidence in RAG. This is a research proposal, not an established model. Novelty must be validated through a current literature review and experiments.

TRACE-RAG | Proposed research framework

# 1. Research objective

Retrieval-Augmented Generation (RAG) systems retrieve external passages before generating an answer. However, retrieved information does not guarantee that every generated statement is factually supported. A response may contain unsupported details, contradictions, or sentences that mix correct and incorrect facts.

The proposed project will develop a claim-level verification module placed between answer generation and final response delivery. It will decompose generated answers into atomic claims, align each claim with retrieved evidence, assess entailment and contradiction, estimate evidence sufficiency and uncertainty, and recommend whether to keep, qualify, correct, retrieve more evidence, or abstain.

# 2. Proposed system architecture

1. **1. Knowledge sources:** PDFs, research papers, reports, web pages, or domain documents.
2. **2. Standard RAG pipeline:** Retrieve relevant passages and generate a draft answer.
3. **3. Claim decomposition:** Split the answer into atomic factual claims; retain entities, relations, dates, quantities, and claim dependencies.
4. **4. Evidence alignment:** Retrieve and rerank passages for each claim, retaining source identifiers and provenance.
5. **5. Entailment analysis:** Estimate entailment, contradiction, and neutral/insufficient evidence for each claim– passage pair.
6. **6. Evidence aggregation:** Combine multi-passage support, contradiction, coverage, source agreement, and uncertainty.
7. **7. Decision controller:** Accept, reject, qualify, retrieve additional evidence, or abstain based on calibrated thresholds.
8. **8. Verified answer:** Return only adequately supported claims as verified, with evidence traceability.
# 3. Research gap and motivation

- Multi-claim sentences: one sentence may combine several factual assertions with different truth statuses.
- Partial support: evidence may establish one facet of a claim while leaving another unverified.
- Conflicting evidence: retrieved documents may disagree about dates, quantities, identities, or conclusions.
- Multi-passage reasoning: some claims require combining evidence from more than one passage.
- Confidence calibration: high model confidence does not necessarily mean that the evidence is sufficient.
- Actionability: a detector should indicate whether more retrieval, qualification, correction, or abstention is appropriate. The project should not claim that claim-level hallucination detection or NLI-based verification is itself new. The research opportunity is to test whether a specific joint architecture for evidence coverage, conflict aggregation, and uncertainty-aware decisions improves performance over established baselines.
# 4. Proposed model: TRACE-RAG

### TRACE-RAG

Triangulated Reliability Assessment of Claims and Evidence in Retrieval-Augmented Generation.

Central hypothesis: a claim should not be considered reliable simply because one retrieved passage receives a high entailment score. Reliability should reflect the extent to which the evidence covers the claim, whether other evidence conflicts with it, and whether the verifier is sufficiently certain to make a decision.

## 4.1 Components

### Atomic Claim and Dependency Extractor

Break each response into atomic claims. Record claim text, entities, relations, quantities, dates, and dependencies between claims. For example, “The bridge was built in 1985 and designed by Engineer A” becomes two separately verified claims.

### Multi-Evidence Entailment Network

Compare every claim with several retrieved passages. Predict entailment, contradiction, and neutral/insufficient evidence for each pair. Retain passage identifiers so each verdict can be traced to its evidence.

### Conflict and Coverage Aggregator

Combine passage-level scores and estimate support, contradiction, source agreement, and whether all factual facets are established. A set of passages can agree with each other yet still fail to establish a particular detail.

### Uncertainty-Aware Decision Controller

Use validation-set-calibrated thresholds to decide whether to accept, reject, qualify, retrieve more evidence, or abstain. An “insufficient evidence” verdict must not be confused with a claim being false.

## 4.2 Example of claim-level verification

Generated sentence: “The bridge was constructed in 1985 and was designed by Engineer A.”

- Claim A: The bridge was constructed in 1985.
- Claim B: Engineer A designed the bridge.
- If the evidence establishes the construction year but says nothing about the designer, Claim A may be supported while Claim B remains insufficiently supported.
- If another reliable passage names Engineer B as the designer, the verifier should flag a conflict or contradiction and preserve the source trace.
# 5. Model architecture and implementation choices

|Module|Recommended starting point|Purpose|
|---|---|---|
|Claim extraction|Instruction-tuned LLM or text-to-text model|Split answers into atomic claims|
|Dense retrieval|BGE or E5 embeddings|Find semantically relevant passages|
|Passage reranking|Cross-encoder reranker|Prioritize evidence for each claim|
|Entailment backbone|DeBERTa-v3-base/large or RoBERTa-MNLI|Classify entailment, contradiction, and neutral|
|Novel aggregator|Trainable MLP or small attention network|Combine passage signals, coverage, and conflict|
|Calibration|Temperature scaling or another validated method|Improve confidence reliability|
|Decision controller|Rule-based first; learned policy if data permits|Select action for each claim|

Recommended strategy: begin with a pretrained NLI model and train the aggregator. Fine-tune the NLI encoder only after a reliable baseline exists. This controls computational cost and makes ablation studies easier to interpret.

# 6. Mathematical formulation

Let cᵢ be the i-th atomic claim and E = {e₁, e₂, …, eₖ} be the retrieved evidence passages.

For each claim–passage pair, the NLI model predicts:

#### zᵢⱼ = [pᵢⱼᴱ, pᵢⱼᶜ, pᵢⱼᴺ]

where E = entailment, C = contradiction, and N = neutral.

##### The aggregator estimates four claim-level features:

- Sᵢ — aggregate support from relevant passages.
- Kᵢ — aggregate contradiction or conflict strength.
- Vᵢ — evidence coverage: the extent to which all relevant claim facets are established.
- Uᵢ — uncertainty in the evidence and verifier prediction.
#### Rᵢ = σ(fθ(Sᵢ, Kᵢ, Vᵢ, Uᵢ))

Here, fθ is a trainable neural network and σ is the sigmoid function. Rᵢ is a learned reliability score, not automatically the probability that a claim is objectively true. Its interpretation must be validated against labelled data and calibration metrics.

## 6.1 Multi-task training objective

##### L = λ₁L_NLI + λ₂L_claim + λ₃L_coverage + λ₄L_calibration

L_NLI trains passage-level entailment classification; L_claim trains the final claim-verdict classifier; L_coverage trains evidence-sufficiency prediction; and L_calibration encourages confidence estimates to reflect observed correctness. Coverage requires facet-level labels or carefully validated proxy labels. Calibration should use a valid differentiable objective or post-training method such as temperature scaling; ordinary cross-entropy alone does not guarantee calibration.

# 7. Step-by-step research plan

## Weeks 1–2: Literature review and problem definition

Read the core papers. Define target failure cases: partial support, multi-passage evidence, source conflicts, and insufficient evidence. Deliverables: literature matrix, research gap, and research questions.

## Weeks 3–4: Dataset preparation

Start with RAGTruth and FEVER. Normalize examples into claim–evidence pairs, preserve source IDs, map labels, and create reproducible splits. Annotate coverage/conflict where existing labels are inadequate.

## Weeks 5–6: Baseline system

Implement RAG, claim extraction, retrieval/reranking, and pretrained NLI verification. Ensure the verifier sees only evidence available at inference time.

## Weeks 7–9: Develop TRACE-RAG

Train the evidence aggregator with claim-verdict, conflict, and coverage supervision. Add calibrated thresholds and abstention.

## Weeks 10–11: Evaluation

Compare against NLI-only verification, AlignScore, and relevant claim-level baselines. Run ablations, inspect errors, and estimate uncertainty in results.

## Week 12 onward: Adaptive retrieval and writing

Trigger another retrieval attempt when evidence is insufficient or conflicting. Measure whether gains justify added latency and cost. Finalize the thesis and reproducible code.

# 8. Datasets and experimental setup

## Recommended datasets

- RAGTruth: annotated hallucinations in RAG-generated responses; useful for hallucination detection and fine-grained analysis.
- FEVER: fact-verification claims with Supported, Refuted, and NotEnoughInfo labels and evidence annotations for supported/refuted cases.
- Additional domain-specific test set: construct a small, carefully annotated set of multi-passage, partially supported, and conflicting-evidence examples to test the proposed novelty. Check the original dataset documentation and licenses before use. Dataset labels do not necessarily map one-to- one to the proposed four-way verdict system; define and document label mapping. Use separate train, validation, and test data, and avoid leakage between related claims or source documents.
# 9. Baselines and evaluation metrics

## Baselines

- Original RAG without verification.
- NLI-only verifier.
- AlignScore or another established claim–evidence alignment metric.
- FActScore-style atomic factuality evaluation where the setup is applicable.
- TRACE-RAG with coverage/conflict features removed.
- Full TRACE-RAG with all proposed components.
## Metrics

**Dimension Suggested metrics** **Detection** Macro-F1, precision, recall, per-class F1 for supported, contradicted, insufficient, and conflicting evidence. **Evidence quality** Evidence Recall@k, evidence precision, evidence sufficiency accuracy. **Calibration and abstention** Expected Calibration Error (ECE), Brier score, risk–coverage curves. **Efficiency** Latency, GPU memory, number of additional retrieval calls, inference cost.

## 9.1 Essential ablation study

9. Entailment only.
10. Entailment + contradiction.
11. Entailment + contradiction + coverage.
12. All preceding components + uncertainty-aware decisions.

13. Full model + adaptive retrieval. Keep test examples and evidence sources consistent across verifier comparisons. Report per-class performance, confidence intervals or multiple-seed results where feasible, and error analysis by claim type.
# 10. Research questions and novelty validation

- RQ1: Does explicit claim-facet coverage improve detection of partially supported claims?
- RQ2: Does conflict-aware multi-passage aggregation improve detection when retrieved sources disagree?
- RQ3: Does uncertainty-aware abstention reduce false verification decisions at a fixed coverage level?
- RQ4: Does adaptive retrieval improve verification accuracy enough to justify its additional computational cost? Novelty must be established through a current literature search and experiments, not by naming the architecture. Search closely related work on atomic factuality, multi-evidence fact verification, entailment-based evaluation, uncertainty-guided retrieval, and selective prediction. A defensible contribution would be a demonstrated improvement from the particular joint coverage–conflict learning objective and verification policy.
# 11. Recommended reading list

**1. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (2023). Min et al., EMNLP 2023.** Atomic fact decomposition and factual precision evaluation. [https://aclanthology.org/2023.emnlp-main.741/](https://aclanthology.org/2023.emnlp-main.741/) | [https://arxiv.org/abs/2305.14251](https://arxiv.org/abs/2305.14251)
##### 2. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models

##### (2024). Niu et al., ACL 2024.

Annotated hallucinations in RAG responses; useful for dataset design. [https://aclanthology.org/2024.acl-long.585/](https://aclanthology.org/2024.acl-long.585/) | [https://arxiv.org/abs/2401.00396](https://arxiv.org/abs/2401.00396)

**3. AlignScore: Evaluating Factual Consistency with a Unified Alignment Function (2023). Zha et al., ACL 2023.** An established claim–evidence alignment baseline. [https://aclanthology.org/2023.acl-long.634/](https://aclanthology.org/2023.acl-long.634/) | [https://github.com/yuh-zha/AlignScore](https://github.com/yuh-zha/AlignScore)
##### 4. TRUE: Re-evaluating Factual Consistency Evaluation (2022). Honovich et al., NAACL 2022.

Evaluation of factual consistency metrics, including NLI- and QA-based methods. [https://aclanthology.org/2022.naacl-main.287/](https://aclanthology.org/2022.naacl-main.287/)

##### 5. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

##### (2023). Manakul et al., EMNLP 2023.

A complementary hallucination detection baseline based on sampled-response inconsistency. [https://aclanthology.org/2023.emnlp-main.557/](https://aclanthology.org/2023.emnlp-main.557/) | [https://arxiv.org/abs/2303.08896](https://arxiv.org/abs/2303.08896)

##### 6. RAGAs: Automated Evaluation of Retrieval Augmented Generation (2024). Es et al., EACL 2024.

Evaluation of RAG faithfulness, retrieval quality, and answer relevance. [https://aclanthology.org/2024.eacl-demo.16/](https://aclanthology.org/2024.eacl-demo.16/)

**7. FEVER: a Large-scale Dataset for Fact Extraction and VERification (2018). Thorne et al., NAACL 2018.** Foundational fact verification dataset with Supported, Refuted, and NotEnoughInfo labels. [https://aclanthology.org/N18-1074/](https://aclanthology.org/N18-1074/) | [https://fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html)

# 12. Final recommended scope

- Dataset: RAGTruth + FEVER, supplemented by a small annotated challenge set for partial support and conflicting evidence.
- Encoder: DeBERTa-v3-base or RoBERTa-MNLI as the starting NLI backbone.
- Retrieval: BM25 plus dense retrieval, followed by reranking.
- New model: TRACE-RAG with a trainable multi-evidence aggregator, coverage supervision, conflict detection, and calibrated abstention.
- Primary contribution: improved detection of partially supported claims and claims with conflicting evidence.
- Secondary contribution: adaptive retrieval when available evidence is insufficient.
- Evaluation: Macro-F1, evidence sufficiency, calibration, abstention performance, latency, and ablation studies. **Core principle: a reliable verifier should identify which parts of a claim are supported, which are contradicted,** **and which remain unverified—rather than treating a whole answer as uniformly factual or hallucinated.**