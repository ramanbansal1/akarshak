# MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination

- **Authors:** Zhuo Li, Yupeng Zhang, Pengyu Cheng, Jiajun Song, Mengyu Zhou,
  Hao Li, Shujie Hu, Yu Qin, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang
  (Qwen Large Model Application Team, Alibaba; CUHK-Shenzhen)
- **Links:** https://arxiv.org/abs/2603.24579 ·
  https://arxiv.org/html/2603.24579 ·
  Code https://github.com/Qwen-Applications/MARCH ·
  ACL Anthology https://aclanthology.org/2026.acl-long.1828/
- **Bibkey:** li2026march
- **Status:** ACL 2026 Long Papers
- **Task:** reduce hallucinations in RAG via multi-agent RL — a trained
  Solver/Proposer/Checker pipeline that co-evolves for factual adherence.

## Summary

Targets **confirmation bias** in LLM-as-a-judge verification: a verifier that
sees the generator's answer tends to reproduce its errors. MARCH enforces
*deliberate information asymmetry*: the Solver generates the RAG response;
the Proposer ("Response Atomizer") decomposes it into verifiable atomic
propositions expressed as **question–answer pairs**; the Checker ("Blinded
Auditor") re-answers those questions using **only the retrieved documents**,
never seeing the Solver's output. A **Zero-Tolerance Reward (ZTR)** compares
the Solver's claims against the Checker's independent answers, and the whole
pipeline is optimized with **multi-agent RL (MARL)**. An 8B model with MARCH
is competitive with closed-source models on hallucination benchmarks
(RAGTruth, FaithBench, Facts Grounding, ContextualJudgeBench) and improves
multi-hop QA.

## Section structure

1. **Introduction** — outcome-level RL rewards too coarse; verifier
   confirmation bias; need claim-oriented supervision
2. **Methodology**
   - 2.1 Problem Formulation and Overview
   - 2.2 MARCH Modeling — overall objective; conditional dependencies of
     Solver → Proposer → Checker under shared policy π_θ
   - 2.3 MARCH Implementation — Zero-Tolerance Reward (ZTR); policy
     optimization (MARL)
3. **Experiment**
   - 3.1 Training Setup — datasets, base model, training details
   - 3.2 Experiments on Hallucination — RAGTruth, FaithBench, Facts
     Grounding, ContextualJudgeBench; metrics
   - 3.3 Experiments on Multi-Hop QA — HotpotQA, 2WikiMultiHopQA, MuSiQue
   - 3.4 Ablation Study and Further Investigation — joint optimization,
     compatibility with other boosters, reward formulations, reward scalars,
     other model families
   - 3.5 Additional Analysis
4. **Related Work** — hallucinations/fact-check in RAG; multi-agent systems
5. **Conclusion**
- **Appendix A Methodology:** A.1 Solver system prompt, A.2 Proposer system
  prompt, A.3 Checker system prompt
- **Appendix B Experiment:** B.1 training details; B.2 evaluation details
  (B.2.1 hallucination/grounding benchmarks, B.2.2 QA benchmarks); B.3
  baseline methods (+CoT, +Few-Shot, +RLHF); B.4 effect of proposed
  question count; B.5 training dynamics / time cost
- **Appendix C Case Study** — full Solver/Proposer/Checker trace with ZTR
  calculation

## Mapping to TRACE-RAG stages

| Stage | MARCH |
|---|---|
| Claim extraction | Proposer atomizes response into QA pairs — no SROM schema, no spans, no dependency edges |
| Entailment | Checker re-answers each QA pair from documents alone (generation-as-verification, not NLI/entailment head) |
| Aggregation | None — ZTR is a scalar RL reward over the claim set; no evidence algebra |
| Calibration/decision | None — reward shaping for training; no inference-time thresholds/abstention |

## Novelty threat

**Low on architecture, moderate on ideas.** Different paradigm (train-time
multi-agent RL vs. inference-time verification module). Two ideas worth
citing/differentiating: (1) information asymmetry — checker never sees the
generator's answer — an anti-confirmation-bias design TRACE-RAG's verifier
implicitly shares (it scores claim–evidence pairs, not answer-vs-answer);
(2) QA-pair reformulation of claims as an alternative atomic representation.
Not a competitor for the aggregator/calibration contribution, but a strong
related system and a possible baseline framing ("verify-then-train" vs.
"verify-then-decide").
