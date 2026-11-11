# TRACE-RAG Competition Index

Survey of the 2025–2026 papers closest to TRACE-RAG's claim-level verification
pipeline. Each file lists the paper's full section structure plus a mapping onto
TRACE-RAG's four stages (claim extraction → entailment → aggregation →
calibration/decision) and an assessment of the novelty threat.

**Headline finding:** every individual pipeline component now has a close
neighbor. FreeGuard is the most dangerous — it already implements SROM claim
extraction, the *same six typed dependency edges*, claim-graph risk propagation,
three-way evidence verdicts, thresholded governance, and abstention. The
surviving TRACE-RAG differentiators are the facet-level evidence algebra
(weakest-link + ancestor gate), risk-controlled α-guarantees on accepted
claims, the insufficient≠false mass-channel semantics, and the
verifiability-aware APS extraction.

## Papers

| Paper | Venue | Task | Closest overlap with TRACE-RAG |
|---|---|---|---|
| [RT4CHART](competitions/rt4chart.md) | arXiv 2603.27752 (2026) | Context-faithfulness detection on RAGTruth | Claim decomposition → 3-way verdicts → span mapping; AND/OR-join aggregation |
| [FreeGuard](competitions/freeguard.md) | Discover Computing 29:713 (Oct 2026), OA | Training-free hallucination mitigation | **Highest threat**: SROM extraction, same 6 edge types, claim graph, noisy-or evidence aggregation, threshold governance, abstention |
| [DagFC](competitions/dagfc.md) | WSDM 2026 (paywalled) | Dependency-aware fact-checking | Claim→KG→dependent verification subtasks on FEVEROUS/HoVer |
| [LoReFact](competitions/lorefact.md) | ACL 2026 Findings, pp. 6974–6996 | Content+logic coupled factuality | Statement decomposition, logic-relation detection, LoReFactScore metric |
| [MARCH](competitions/march.md) | ACL 2026 Long | Multi-agent hallucination self-check | Proposer atomizes → Checker re-answers QA pairs in isolation; MARL |
| [HalluGuard](competitions/halluguard.md) | arXiv 2510.00880 (Oct 2025) | Pairwise doc–claim groundedness (4B SRM) | Pairwise verifier baseline; no aggregation/calibration |

## Also relevant (not yet profiled)

- **ToE** — arXiv 2606.27736: hierarchical claim verification tree; re-decomposes
  evidence-insufficient nodes and prunes confident subtrees. Crowds the
  "retrieve more / re-decompose" loop.
- **BCEA** — arXiv 2606.16667: Budgeted Conformal Evidence Acquisition (VLM);
  three-way answer/abstain/acquire-more under a conformal guarantee. Direct
  analog of TRACE-RAG's retrieve-more action in the vision domain.
- **Conformal factuality line** — Quach et al. 2024 (arXiv 2402.10978),
  level-adaptive CP (arXiv 2406.09714), UniCR (arXiv 2509.01455), CAP (PMLR
  v304), IntroConformal (arXiv 2609.01375): claim-level calibrated abstention
  is an active area — calibration alone is not novel.
- **APS lineage** — Hosseini et al. 2024 (arXiv 2406.19803), Google Gemma-APS;
  motivates the Granite-APS fine-tune.
- **Name collisions** — "TRACE-RAG" is already used by a trace-consistent
  retrieval framework (agentspex.ai, 2026) and "TraceRAG" (ICIC 2026,
  trajectory-aware concept tracing). Rename recommended.
