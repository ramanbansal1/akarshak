# HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in RAG

- **Authors:** Loris Bergeron (Banque de Luxembourg; SnT, Univ. of
  Luxembourg), Ioana Buhnila (CDH Chosun Univ.; ATILF), Jérôme François
  (SnT), Radu State (SnT)
- **Links:** https://arxiv.org/abs/2510.00880 ·
  https://arxiv.org/html/2510.00880
- **Status:** arXiv v1, 01 Oct 2025 · to be released Apache 2.0 on acceptance
- **Task:** pairwise document–claim groundedness classification with
  evidence-grounded justification — a 4B Small Reasoning Model for RAG
  hallucination mitigation.

## Summary

Fine-tunes a 4B SRM to classify (document, claim) pairs as grounded or
hallucinated while generating evidence-grounded justifications. Built on
**HalluClaim**: a large-scale synthetic dataset from FineWeb via Llama-3.3-70B,
with multi-stage curation, prompt-guided style reformation, and three claim
types (grounded, intrinsic-hallucinated, extrinsic-hallucinated). Preference
training uses **Odds Ratio Preference Optimization (ORPO)** on
reasoning-guided preference pairs with model-agreement verification and LLM
consensus filtering. On the RAGTruth subset of LLM-AggreFact: 84.0% balanced
accuracy — matching MiniCheck-7B (84.0%) and Granite Guardian 3.3-8B (82.2%)
at ~half the parameters; 75.7% BAcc on the full benchmark, matching GPT-4o
(75.9%).

## Section structure

1. **Introduction**
2. **Related Work** — hallucination mitigation (prompting, RAG, decoding,
   SFT, self-reflection); fact-checking (MiniCheck, Granite Guardian)
3. **Problem Formulation** — grounded vs. intrinsic/extrinsic hallucination
   definitions over document–claim pairs
4. **Method**
   - 4.1 HalluGuard Overview — think-then-classify-then-justify pipeline
   - 4.2 Structured Claim Dataset Construction — domain-agnostic corpus
     (FineWeb); multi-stage curation; prompt-guided data reformation;
     synthetic claim generation (grounded / hallucinated_intrinsic /
     hallucinated_extrinsic)
   - 4.3 Preference Training Dataset Construction — reasoning-guided
     preference pairs; model-agreement verification; LLM-based consensus
     filtering
   - 4.4 Preference-Based Fine-Tuning — pretrained backbone; PEFT with ORPO
5. **Experimental Setup** — benchmark dataset (LLM-AggreFact/RAGTruth
   subset), evaluation metric (balanced accuracy)
6. **Results** — benchmark evaluation; RAGTruth detailed eval; justification
   evaluation; human alignment evaluation
7. **Ablation Study** — consensus filtering, reasoning contribution,
   preference alignment, ablation results
8. **Conclusion**
9. **Future Work**
- **Limitations** — synthetic data, output formatting, hallucination
  coverage, language/domain generalization
- **Ethical Considerations**
- References
- **Appendices A–J:** A technical reproducibility; B style-reformation
  prompt; C claim-generation prompts (grounded / intrinsic / extrinsic); D
  synthetic-pairs prompt; E consensus-filter prompt; F benchmark datasets;
  G reward gap across epochs; H inference parameters; I fine-tuning config;
  J full ORPO preference tuple example

## Mapping to TRACE-RAG stages

| Stage | HalluGuard |
|---|---|
| Claim extraction | None at inference — claims are inputs; synthetic claims only for training data |
| Entailment | Single pairwise (document, claim) → grounded/hallucinated classifier with justification; the "verifier" role |
| Aggregation | None — independent pairwise decisions |
| Calibration/decision | None — no thresholds, abstention, or risk control |

## Novelty threat

**Low as a system, useful as a baseline.** It is the strongest *pairwise
verifier* reference point (84% BAcc on RAGTruth at 4B params) — exactly the
"pairwise-only baseline" TRACE-RAG's proposal already names in §4. Cite it
when arguing that pairwise verdicts are insufficient without aggregation and
calibration. Its synthetic-data recipe (claim generation + consensus
filtering + ORPO) is also a candidate method for training TRACE-RAG's
verifier head if WICE/FEVER labels prove too thin.
