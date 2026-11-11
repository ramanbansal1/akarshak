# Supporting Papers — Data-Efficient Teacher→Student Distillation

Papers on training a small student (our Granite-APS extractor / verifier head)
from a stronger teacher with **very few labeled examples**. Relevant because
TRACE-RAG's APS fine-tune must run on Kaggle 2×T4 (QLoRA-class budget) and we
want the extraction schema (claims + facets + spans) without a large
annotation effort.

| Paper | Venue | Data regime | Core trick |
|---|---|---|---|
| [CoD](cod.md) | NeurIPS 2025 | 8–512 samples | Counterfactual explanations map the teacher's decision boundary |
| [Distilling step-by-step](distilling-step-by-step.md) | ACL 2023 Best Paper | ~50–80% less data | Multi-task on teacher rationales + labels |
| [DED](ded.md) | arXiv 2508.09883 (2025) | 0.8k curated examples | Teacher selection + curated corpus + diverse trajectories |
| [SRD](srd.md) | arXiv 2508.06135 (2025) | plug-and-play | Student-reflection data selection + curriculum scheduling |
| [LLKD](llkd.md) | NAACL 2025 Findings | sample selection | Teacher-confidence × student-need adaptive sampling |
| [Skill-aware distillation](skill-aware-distillation.md) | ACL 2026 Short | 1k of 100K | Per-skill weakness targeting + skill-decomposition SFT |

## How this maps to TRACE-RAG

- **APS extraction (generative)**: the distillation target is a *structured
  output policy*, not a classifier — CoD's decision-boundary argument applies
  weakly; step-by-step/DED/SRD-style rationale or curriculum tricks transfer
  better. Teacher = frontier LLM or Gemma-APS emitting claims+facets+spans.
- **Verifier head (discriminative)**: CoD is directly relevant — few-shot
  distillation of a 3-class claim–evidence verdict with counterfactual
  claim variants (perturb entity/date/number — the same perturbations
  FreeGuard uses for leakage testing) could give boundary-fidelity with
  tiny data.
- **Compute reality**: all of these assume standard SFT; on 2×T4 the binding
  constraint is optimizer/memory, addressed by QLoRA + parameter-efficient
  tricks (see interview notes on block-diagonal LoRA / CMA-ES search).
