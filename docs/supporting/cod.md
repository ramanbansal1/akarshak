# CoD: Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations

- **Authors:** Faisal Hamman, Pasan Dissanayake, Yanjun Fu, Sanghamitra Dutta
  (University of Maryland, College Park)
- **Links:** https://arxiv.org/abs/2510.21631 ·
  https://arxiv.org/html/2510.21631 ·
  Code https://github.com/FaisalHamman/CoD
- **Venue:** NeurIPS 2025 (poster)
- **Task:** task-aware few-shot distillation — student mimics teacher with
  as few as **8–512 samples**.

## Summary

Standard task-aware distillation needs large datasets. CoD infuses
**counterfactual explanations (CFEs)** — minimally perturbed inputs that flip
the teacher's prediction — into the distillation set. CFEs act as "knowledge
probes" near the teacher's decision boundary: they carry more information per
sample than standard data. Theoretical results: (1) CFEs improve parameter
estimation via Fisher information (Theorem 1); (2) teacher–student boundary
proximity via Hausdorff distance (Theorem 2, Lemma 1). Experiments on SST-2,
CoLA, IMDB, Sentiment140, Amazon, Yelp with DeBERTa-v3 / Qwen2.5 teachers
(1.5B, 7B) — outperforms standard distillation in few-shot regimes, using
half the samples paired with CFEs.

## Section structure

1. **Introduction**
2. **Preliminaries** — Definition 1 (Closest CFE); Remark 1 (Data Manifold
   CFEs)
3. **Main Contributions**
   - Definition 2 (Fisher Information Matrix); **Theorem 1** (CFEs Improve
     Model Parameter Estimation)
   - Definition 3 (Hausdorff Distance); Lemma 1 (Existence of Boundary
     Crossing for Counterfactual Pairs); **Theorem 2** (Teacher–Student
     Boundary Proximity)
4. **Experiments**
- **Appendix A:** Fisher information background + Theorem 1 proof
  (Löwner order, trace inequality, feature spanning, datapoint diversity)
- **Appendix B:** Hausdorff distance background + Lemma 1 / Theorem 2 proofs
- **Appendix C:** C.1 datasets; C.2 CFE generation prompts; C.3 baselines;
  C.4 models/hyperparameters; C.5 additional results; C.6 ablations

## Relevance to TRACE-RAG

- **Verifier head**: directly applicable pattern — generate counterfactual
  claim variants (swap entity/date/number/negation) labeled by the teacher,
  distill the 3-way verdict head on base examples + boundary CFEs. Cheap,
  few-shot, and the perturbation types match the proposal's truth-critical
  modifiers (AtomEval SROM) and FreeGuard's leakage variants.
- **APS extractor**: weaker fit (generative, not a decision boundary), but
  the CFE idea could still produce hard segmentation cases (minimal edits
  that change claim boundaries).
