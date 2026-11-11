# Learning with Less: Knowledge Distillation from LLMs via Unlabeled Data (LLKD)

- **Authors:** (see anthology 2025.findings-naacl.142)
- **Links:** https://aclanthology.org/2025.findings-naacl.142/ ·
  PDF https://aclanthology.org/2025.findings-naacl.142.pdf
- **Venue:** Findings of NAACL 2025
- **Task:** adaptive sample selection for LLM→small-model distillation on
  unlabeled data — prioritize samples that are both teacher-reliable and
  student-informative.

## Summary

LLM pseudo-labeling of unlabeled data enables distillation without labels,
but noisy pseudo-labels hurt. LLKD selects each training sample using two
signals: **teacher confidence** (reliable pseudo-label) × **student
information need** (challenging for the student — high utility). Adaptive,
per-step selection improves data efficiency across datasets.

## Section structure (from PDF)

1. **Introduction**
2. **Related Work** — incl. thresholding methods for pseudo-labels
3. **Method**
   - 3.1 Notations
   - 3.2 (sample scoring / selection criteria)
   - 3.3 Student Model
   - 3.4 Data Selection (adaptive per-step)
4. **Experiments** — main results; 4.3 ablation study; 4.4 data efficiency;
   4.5 choice of teacher model; 4.6 thresholds evaluation
5. **Conclusion**
- References; Appendix A (A.2 additional results, thresholds)

## Relevance to TRACE-RAG

- Unlabeled corpora are free (web text, paper abstracts, RAG answers) —
  LLKD's teacher-confidence × student-need selection is a cheap way to pick
  the APS distillation subset without any manual annotation.
- Complements SRD: LLKD selects on teacher+student signals; SRD adds
  curriculum scheduling on top.
