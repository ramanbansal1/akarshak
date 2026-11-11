# LoReFact: Bridging the Logic Gap in Fact-Checking

- **Authors:** Qiming Xie, Wenjie Zheng (NJUST); Xiangqing Shen, Rui Xia (NJU)
- **Links:** https://aclanthology.org/2026.findings-acl.346/ ·
  PDF https://aclanthology.org/2026.findings-acl.346.pdf ·
  Code/data https://github.com/NUSTM/LoReFact
- **Bibkey:** xie-etal-2026-lorefact · DOI 10.18653/v1/2026.findings-acl.346
- **Status:** Findings of ACL 2026, pp. 6974–6996, San Diego, July 2026
- **Task:** long-form fact-checking with a **content–logic coupled** paradigm —
  factuality has two dimensions: whether atomic facts are true, and whether
  the *logical relations* connecting them are valid.

## Summary

Argues the decompose-then-verify paradigm verifies facts but ignores logical
dependencies, so logically corrupted text can still score as factual. Releases
**LoReFact**: the first long-form fact-checking dataset with systematic
logical relations (36 topics, 4 domains, 1,080 samples, 5,706 annotated
logical statements; relations: causal, conditional, temporal, concessive).
**LoRe-Factcheck** runs two parallel pipelines — content factuality
evaluation (standard decompose-and-verify, factual precision) and logic
factuality evaluation (detect logical relations, judge their validity) —
then **LoReFactScore** integrates both into a unified per-statement and
response-level metric that penalizes logical fallacies.

## Section structure (from PDF)

1. **Introduction**
2. **LoReFact** (the dataset)
   - 2.1 Data Collection — LLM generation pipeline: topic → question → 20
     atomic claims (10 true / 10 false) → logical statements with planted
     errors → 500+ word response
   - 2.2 (annotation/verification stage — details in Appendix A.1)
   - 2.3 Data Analysis — dataset statistics, relation-type distribution
3. **LoRe-Factcheck** (the framework)
   - 3.1 Task Definition and Framework Overview — two parallel evaluation
     pipelines + integration (Figure 5)
   - 3.2 Statement Decomposition and Decontextualization — sentence split,
     decontextualization ("it rotates" → "Mars rotates"), checkworthiness
     filtering; all via GPT-4o
   - 3.3 Logic Factuality Evaluation — detect logical relations between
     statements, judge validity of each
   - 3.4 Content Factuality Evaluation — decontextualized atomic claims
     verified against evidence; ContentScore = factual precision
   - 3.5 LoReFactScore — unified metric; degenerates to ContentScore when a
     statement has no logical relation
4. **Experiments**
   - 4.1–4.2 Setup / baselines (three representative fact-checking baselines)
   - 4.3 Main Results — LoReFactScore stricter yet more faithful; strong
     per-relation-type results (Table 4)
   - 4.4 (generalization across scenarios/domains)
   - 4.5 Ablation Studies — logic dimension's contribution
   - 4.6 In-depth Analysis of Preliminary Study — 50 real-world political
     statements
   - 4.7 Case Study
5. Related Work / Conclusion / Limitations / Ethics (order per ACL
   Findings template)
- **Appendix A:** A.1 LoReFact details (A.1.1–A.1.3 generation prompts),
  A.2 LoRe-Factcheck details (A.2.1, A.2.2 instruction prompts), A.3 worked
  example of the pipeline
- **Appendix B:** B.1, B.2 additional results/analysis

## Mapping to TRACE-RAG stages

| Stage | LoReFact |
|---|---|
| Claim extraction | GPT-4o sentence split + decontextualization + checkworthiness filter; logical statements extracted as relations between atomic claims — closest existing analog to TRACE-RAG's typed dependency edges (causal/conditional/temporal/concessive) |
| Entailment | Content pipeline = standard decompose-and-verify factual precision per statement |
| Aggregation | LoReFactScore = precision-style aggregation penalizing logical fallacies — a *metric*, not an evidence aggregator |
| Calibration/decision | None — scoring framework, no thresholds or abstention |

## Novelty threat

**Moderate-high on dependencies, low elsewhere.** It owns the
"content vs. logic factuality" framing and provides the only benchmark with
annotated logical relations — directly relevant to TRACE-RAG's dependency
edges and a candidate evaluation dataset. It does *not* do evidence
aggregation, calibration, or decision control. Position TRACE-RAG's
ancestor/coherent-factuality gate as the operational counterpart of
LoReFact's logic-factuality dimension: they *measure* logical validity
post-hoc; TRACE-RAG *enforces* it during verification (a claim cannot be
established if its ancestors fail).
