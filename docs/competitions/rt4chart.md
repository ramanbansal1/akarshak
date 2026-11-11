# RT4CHART: Retromorphic Testing for Hallucination Detection in RAG Pipelines

- **Authors:** Boxi Yu, Yuzhong Zhang, Liting Lin, Lionel Briand, Emir Muñoz
  (Lero/University of Limerick; CUHK-Shenzhen; University of Ottawa; Genesys)
- **Links:** https://arxiv.org/abs/2603.27752 · https://arxiv.org/html/2603.27752
- **Status:** arXiv v2, 20 Jul 2026 · License CC BY 4.0
- **Task:** context-faithfulness (hallucination) detection in RAG — strictly
  grounded in the retrieved context, not parametric knowledge.

## Summary

Treats detection as a *traceability* problem via retromorphic testing: decompose
the answer into self-contained claims, verify each claim locally over
overlapping context chunks and then globally against the full context, map
claim labels back to answer spans and context evidence spans, and aggregate
claim labels into an answer-level verdict. Evaluated on RAGTruth++ (408
samples) and a re-annotated RAGTruth-Enhance (2,675 samples). Answer-level F1
0.776 on RAGTruth++ (0.845 precision / 0.718 recall), 47.5% span-level F1 on
RAGTruth-Enhance; claims 83% relative F1 improvement over strongest baseline.
Re-annotation uncovered 1.68× more hallucination cases than original labels.

## Section structure

1. **Introduction** — faithfulness defined w.r.t. retrieved context; auditing
   requires span + evidence, not just a score.
2. **Preliminaries** — formal relation RR((C,Q),T): every claim's evidence
   spans must be grounded in context and justify the claim label.
3. **Approach**
   - 3.1 Theoretical Foundation: Retromorphic Testing
   - 3.2 Pipeline Overview — four stages: decomposition → local verification →
     global verification → answer-level aggregation
   - 3.3 Claim Decomposition — sentence-split then LLM decomposes each
     sentence independently; preserves negation/quantities/temporal/modality;
     decomposer sees only the answer, not the context
   - 3.4 Local Verification — overlapping sentence chunks (window W, overlap O);
     LLM judge per (claim, chunk) → v ∈ {Ent, Nic, Con}; OR-join:
     Con dominates > any Ent > else Nic
   - 3.5 Global Verification — re-verify each claim against full context;
     adaptive prompting: Nic claims searched from scratch, Ent/Con claims get
     the local chunk as a "location hint"
   - 3.6 Answer-Level Aggregation — AND-join over final claim labels:
     any Con → Con; else any Nic → Nic; else Ent
4. **Experiments**
   - 4.1 Research Questions (RQ1–RQ4)
   - 4.2 Experimental Setup — datasets, RAGTruth-Enhance construction,
     compared detectors, unified evaluation contract, baseline adaptations,
     implementation details, metrics
   - 4.3 Overall Effectiveness — main results, span-level localization,
     context-evidence diagnostic
   - 4.4 Complementary Roles of Local and Global Verification — ablations
   - 4.5 Design Sensitivity — decomposition granularity, chunking hyperparams
   - 4.6 Robustness, Reproducibility, and Cost — judge-model choice,
     run-to-run reproducibility, deployment cost
   - 4.7 Case Study: Uncovering Overlooked Hallucinations
5. **Limitations and Threats to Validity** — context-only scope, failure
   modes, residual gap, model dependence, generality
6. **Related Work** — hallucination detection; claim decomposition and fact
   verification; test-oracle techniques in SE
7. **Conclusion**
- Data Availability, Acknowledgements, References

## Mapping to TRACE-RAG stages

| Stage | RT4CHART |
|---|---|
| Claim extraction | Sentence-split + LLM decomposition; no schema, no spans-to-source beyond answer text, no dependency edges |
| Entailment | LLM-as-judge, hard 3-way labels {Entailed, Not-Enough-Info, Contradicted} per claim×chunk, then full-context re-judge |
| Aggregation | Deterministic joins only: OR over chunks (Con>Ent>Nic), AND over claims. No probabilities, weights, or facets |
| Calibration/decision | None — hard labels; answer flagged if any claim unfaithful; no abstention budget or risk control |

## Novelty threat

**High on the pipeline shape** (decompose → verify → aggregate → flag on
RAGTruth), **low on the math**: its aggregation is deliberately simple
priority joins, it has no calibration, no facet coverage semantics, no
dependencies, no insufficient≠false mass channels. TRACE-RAG must beat or
contextualize its RAGTruth numbers; its Nic label is the closest existing
"insufficient evidence" concept but is collapsed into the same label space
as Ent/Con rather than a separate mass channel.
