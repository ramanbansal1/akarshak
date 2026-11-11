# Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning (DED)

- **Authors:** (see arXiv 2508.09883)
- **Links:** https://arxiv.org/abs/2508.09883 · https://arxiv.org/html/2508.09883
- **Status:** arXiv preprint, Aug 2025
- **Task:** reasoning distillation with extremely few curated examples —
  SOTA on math (AIME 2024/2025, MATH-500) and code (LiveCodeBench) using
  only **0.8k carefully curated examples**, no large-scale SFT.

## Summary

Argues reasoning distillation doesn't need corpus scaling — a Pareto-optimal
small curated set beats big corpora. Three components: (1) **teacher
selection** — benchmark scores alone don't predict distillation quality;
they compare leading reasoning LLMs to pick an effective teacher; (2)
**curated corpus** — a smaller, carefully selected set beats scaled
corpora, trading in-domain gains for out-of-domain retention; (3) **diverse
reasoning trajectories** — on-policy-style diverse rollouts give the student
robust reasoning. Validated on math reasoning and code generation.

## Section structure

1. **Introduction**
2. **Related Work**
3. **Method** — teacher selection, curation, trajectory diversity
4. **Experiments and Analysis**
   - 4.1 Setup
   - 4.2 Main Results
   - 4.3 DED for Code Generation
   - 4.4 More Analyses
5. **Conclusion**
- References

## Relevance to TRACE-RAG

- **Teacher choice matters more than data volume** — directly applicable:
  before committing to a frontier-LLM teacher for APS labels, run a small
  teacher comparison on a held-out extraction metric (atomicity/faithfulness
  F1) rather than assuming the biggest model wins.
- **Curated > scaled** supports the hybrid plan: a few hundred
  hand/teacher-curated domain examples (scientific claims, RAG answers)
  may beat tens of thousands of generic web text for the APS fine-tune.
