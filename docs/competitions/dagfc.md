# DagFC: Dependency-Aware Fact-Checking via Claim-Constructed Knowledge Graphs and Large Language Models

- **Authors:** Zhouhui Wu, Zhuohua Yang, Jiaojiao Jiang, Shuiqiao Yang, Nan Sun
  (UNSW Sydney and affiliates)
- **Links:** https://dl.acm.org/doi/10.1145/3773966.3777966 ·
  DOI 10.1145/3773966.3777966
- **Status:** WSDM 2026, pp. 745–755 (closed access — full text paywalled;
  structure below is from abstract + secondary descriptions, marked where
  inferred)
- **Task:** automated fact-checking (veracity prediction) of individual
  claims, with emphasis on multi-hop claims whose sub-claims are dependent.

## Summary (from abstract)

Existing LLM fact-checking follows a divide-and-conquer paradigm: complex
claims are split into simpler sub-claims verified in isolation, which fails
for multi-hop claims where inter-sub-claim dependencies matter. DagFC builds
a **Knowledge Graph from each claim** to guide decomposition of the
fact-checking problem into **dependent verification subtasks** — clarification
tasks (resolve ambiguous entities) and retrieval tasks (collect evidence) —
scheduled by difficulty, then has an LLM produce the final verdict over the
collected evidence. LLMs are used throughout: KG construction, task
generation/scheduling, and final verification. Outperforms SOTA in accuracy
and macro-F1 on **FEVEROUS** and **HoVer**; includes a user-facing prototype.

## Section structure

Full text is paywalled; the following is the typical WSDM structure inferred
from the abstract and available summaries — **verify against the PDF before
citing section numbers**:

1. Introduction
2. Related Work — fact-checking / claim verification pipelines
3. Method
   - 3.x Knowledge graph construction from claim
   - 3.x Dependency-aware task generation (clarification + retrieval tasks)
   - 3.x Task scheduling by difficulty / dependency order
   - 3.x Task processing and evidence integration
   - 3.x Final veracity prediction by LLM
4. Experiments — FEVEROUS, HoVer; accuracy and macro-F1 vs. SOTA
5. Analysis / ablations (inferred)
6. Conclusion
- User-facing prototype description (per abstract)

## Mapping to TRACE-RAG stages

| Stage | DagFC |
|---|---|
| Claim extraction | Inverse direction: decomposes a *single claim* into dependent verification subtasks via a claim-constructed KG (entities/relations/properties), not a response into claims |
| Entailment | LLM answers generated retrieval/clarification subtasks, then verifies the original claim over collected evidence |
| Aggregation | Intermediate task results integrated by the LLM in dependency order — no formal evidence algebra |
| Calibration/decision | Single verdict per claim; no abstention, calibration, or risk control |

## Novelty threat

**Moderate.** Closest on "dependencies between sub-claims matter," but the
dependencies live *inside one claim's KG* (clarification/retrieval scheduling)
rather than *between claims of a generated response*. Different task
(fact-checking vs. hallucination detection), different mechanism (task
scheduling vs. evidence aggregation), no calibration. Citable as evidence
that dependency-aware verification improves over isolated sub-claim checks —
useful motivation for TRACE-RAG's ancestor gate, but it does not implement
response-level claim graphs or calibrated decisions.
