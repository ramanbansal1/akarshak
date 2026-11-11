# Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation

- **Authors:** (see anthology 2026.acl-short.49)
- **Links:** https://aclanthology.org/2026.acl-short.49/ ·
  PDF https://aclanthology.org/2026.acl-short.49.pdf
- **Venue:** ACL 2026 Short Papers
- **Task:** transfer reasoning ability with tiny data — **1,000 examples**
  selected from a 100K teacher-generated corpus beat random SFT by +1.6%
  (Qwen3-4B) / +1.4% (Qwen3-8B) across five math benchmarks.

## Summary

Skill-centric distillation with two components: (1) **Skill-based data
selection** — map each problem onto a hierarchical skill tree (Instruct-
SkillMix, Kaur et al. 2025) via top-down LLM skill attribution, evaluate the
student's per-skill accuracy, and sample preferentially from the student's
*weaker* skills; (2) **Skill-aware fine-tuning** — prepend explicit skill
decompositions to training targets ("Skills: Mathematics → Probability →
Bayes' theorem; ...") so the student learns to decompose problems into
skills during reasoning. Analysis confirms gains concentrate on the skills
emphasized during training; full-100K-corpus SFT can even underperform the
base model — data selection matters more than volume.

## Section structure (from PDF)

1. **Introduction** — Fig. 1: Skill Tree Attribution → Skill-based Sampling
   → Skill-aware Training pipeline
2. **Related Work**
3. **Method** — three steps: skill tree attribution (top-down LLM
   attribution); skill-based sampling (per-skill student accuracy guides
   sampling); skill-aware SFT (structured reasoning targets)
4. **Experiments** — math reasoning benchmarks, Avg@8; comparison tables
   (skill-based vs. random vs. full-corpus)
5. Analysis/conclusion sections (per ACL short format)
- References

## Relevance to TRACE-RAG

- **Skill-tree framing maps onto claim structure**: our "skills" could be
  the atomicity rules (single relation, ≤1 temporal/numeric/modality
  constraint, decontextualization, span faithfulness). Attribute each
  training answer to the decomposition skills it exercises, find where
  Granite-APS is weakest, and oversample those.
- Explicit-skill targets parallel our SROM/facet schema — the student can
  be trained to emit the facet decomposition as part of its reasoning
  trace, making facets first-class output rather than post-hoc.
