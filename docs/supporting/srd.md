# Less is More: Selective Reflection for Compatible and Efficient Knowledge Distillation in Large Language Models (SRD)

- **Authors:** Lingyuan Liu, Mengxiang Zhang
- **Links:** https://arxiv.org/abs/2508.06135 · https://arxiv.org/html/2508.06135
- **Status:** arXiv preprint, Aug 2025
- **Task:** data-centric KD — improve white-box distillation by selecting
  prompt–response pairs the *student* can actually learn from.

## Summary

Existing white-box KD balances ground-truth vs. student-generated responses
but ignores **data quality and student compatibility**. SRD (Selective
Reflection Distillation) has the student model reflect on training data:
automated ranking compares ground truth with student outputs and curates
high-quality, student-compatible instances by difficulty; a **curriculum
scheduling** strategy then introduces curated subsets at fixed intervals.
Plug-and-play across KD methods and architectures: consistent gains with up
to **39% less training runtime**.

## Section structure

1. **Introduction**
2. **Background and Preliminary Study** — 2.1 white-box KD framework;
   2.2 challenges in training-data usage; 2.3 toward principled data
   utilization
3. **Methodology** — 3.1 motivation; 3.2 selective reflection on training
   data; 3.3 curriculum scheduling
4. **Experiments** — 4.1 instruction-following; 4.2 summarization/MT;
   4.3 math reasoning and code generation (each with Setup/Results)
5. **Analysis and Discussion** — difficulty ranking impact; adaptive
   parameter scheduling; training order; hyperparameter sensitivity;
   comparison with traditional curriculum learning
6. **Related Work**
7. **Conclusion**
- **Appendix A** white-box KD formulation; **B** experimental setup (base
  models, datasets, training/eval details, baselines); **C** results
  explanation; **D** additional related work; **E** limitations

## Relevance to TRACE-RAG

- **Student-aware data selection**: before fine-tuning Granite-APS, run the
  current stub/rule-based extractor (or base Granite prompted) over the
  teacher corpus and *select the examples where the student fails* — cheap
  way to build a small high-value SFT set on a T4 budget.
- Curriculum ordering helps when mixing easy generic APS examples with hard
  domain ones (scientific claims, multi-facet sentences).
