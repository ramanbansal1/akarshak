# Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

- **Authors:** Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost,
  Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, Tomas Pfister
  (Google Brain / UC Berkeley et al.)
- **Links:** https://arxiv.org/abs/2305.02301 · https://arxiv.org/html/2305.02301
- **Venue:** ACL 2023 (Best Paper)
- **Task:** distill LLM rationales + labels into small task-specific models
  using far less data than finetuning or standard distillation.

## Summary

Instead of distilling only final labels, extract **LLM-generated rationales**
and train the student multi-task: predict the label AND generate the
rationale. Rationales act as additional supervision, so the student learns
the task structure, not just input→output mapping. Results across 4 NLP
benchmarks: >50% less training data than finetuning/distillation (up to 85%
reduction); a 770M T5 beats 540B PaLM using only 80% of data on one task.
Ablations: multi-task rationale training beats single-task rationale+label
joint prediction; works across LLM sizes; unlabeled data augmentation helps.

## Section structure

1. **Introduction**
2. **Related work** — knowledge distillation from large models; learning
   with human rationales; learning with LLM-generated rationales
3. **Distilling step-by-step**
   - 3.1 Extracting rationales from LLMs
   - 3.2 Training smaller models with rationales — standard finetuning/task
     distillation vs. multi-task learning with rationales
4. **Experiments**
   - Setup; Datasets
   - 4.1 Reducing training data — vs. finetuning (less labeled), vs.
     distillation (less unlabeled)
   - 4.2 Reducing model size — across model sizes; vs. LLMs; unlabeled
     augmentation
   - 4.3 Outperforming LLMs with minimum model size and least data
   - 4.4 Further ablations — different LLM sizes; multi-task vs.
     single-task rationale+label
5. **Discussion**
- Limitations; Ethics statement; References
- **Appendix A:** A.1 Implementation; A.2 Datasets

## Relevance to TRACE-RAG

- **APS extractor**: the rationale-as-supervision idea maps well — have the
  teacher emit *why* a proposition boundary exists (the atomicity rule
  applied) alongside the segmentation; student learns the decomposition
  policy, not just surface patterns. Also supports "verifiability-aware"
  prompting: rationale can state which evidence facet each claim needs.
- Cheap supervision multiplier when labeled examples are scarce.
