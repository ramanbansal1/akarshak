# FreeGuard: Claim Level Graph Reasoning for Training-Free LLM Hallucination Mitigation

- **Authors:** Haonan Wang, Wen Zhao (City University of Hong Kong;
  Northwestern Polytechnical University)
- **Links:** https://link.springer.com/article/10.1007/s10791-026-10667-z
  (open access) · DOI 10.1007/s10791-026-10667-z
- **Status:** Discover Computing 29:713, published 07 Oct 2026
- **Task:** post-hoc hallucination *mitigation* (detect + revise/hedge/mark/
  abstain) over long-form LLM responses; training-free.

## Summary

Decomposes a response into atomic claims, builds a **typed claim graph**,
propagates hallucination risk non-parametrically over it, routes high-risk
claims to stronger verification, verifies via a **bidirectional evidence
graph** with three-way NLI verdicts, and applies **counterfactual
evidence-chain testing** (does the evidence also "support" minimally perturbed
false variants?). Final claim states {Supported, Contradicted, Fragile,
Ambiguous, Unverified} map to governance actions {Preserve, ReviseRemove,
Hedge, MarkUnsupported, DowngradeConfidence}; abstains on
{Contradicted, Unverified, Fragile} for high-stakes claims. Five benchmarks:
AUROC 83.1→85.5, hallucination rate 32.0%→16.2%, confident-hallucination rate
25.9%→9.8%, correctness 63.2%→70.2%.

## Section structure

1. **Introduction**
2. **Related work**
   - 2.1 Hallucination detection and uncertainty estimation
   - 2.2 Claim-level factuality evaluation
   - 2.3 Retrieval-augmented hallucination mitigation
   - 2.4 Evidence graph reasoning and counterfactual verification
3. **Method**
   - 3.1 Overall framework
   - 3.2 Atomic claim graph construction — SROM tuple c=(s,r,o,t,v,m);
     recursive splitting (one relation, ≤1 temporal/numerical/modality);
     six relation types {coreference, temporal, numerical, causal,
     entailment, conflict}; edge when Conf_r > τ_r
   - 3.3 Training-free risk propagation and routing — node features
     (embedding, atomicity, graph-structural, semantic/verbal uncertainty
     mismatch Δ); fixed-weight risk init; relation-aware message passing
     (λ=0.55 self-retention, relation priors); 4-tier verification routing
     by risk thresholds γ1–γ3
   - 3.4 Bidirectional evidence graph verification — heterogeneous evidence
     graph (passage/source/attribute nodes); three-way {sup, con, neu} per
     claim–evidence pair; evidence quality Q = Rel·Auth·Fresh·Spec;
     **noisy-or aggregation** S_i, C_i; contradiction threshold θ_c
   - 3.5 Counterfactual evidence-chain governance — 4 perturbations
     (entity/time/number/relation); leakage L_i = max S⁻/S; 5-state decision
     table on (S_i, C_i, L_i) with thresholds θ_s, θ_c, θ_l; governance
     actions; abstention rule
   - 3.6 Implementation and hyperparameter settings — Llama-3.1-8B-Instruct
     generator; frozen LLM/NLI/reranker; cross-model eval on Mistral-7B,
     Qwen2.5-7B
4. **Experiments**
   - 4.1 Datasets — dev split for threshold selection only
   - 4.2 Evaluation metrics — AUROC primary, ACC reference
   - 4.3 Comparison with state-of-the-art methods
   - 4.4 Efficiency analysis
   - 4.5 Analysis of FreeGuard governance
   - 4.6 Evidence and counterfactual analysis — vs GEAR/KGAT-style
     evidence-graph baselines
   - 4.7 Ablation studies
   - 4.8 Cross-model generalization — fixed hyperparameters across models
   - 4.9 Hyperparameter sensitivity — T, γ2, θ_l, K
5. **Limitations** — errors in decomposition/retrieval/NLI can still
   false-positive; leakage ≠ falsehood
6. **Conclusion**
- Data/Materials/Code availability, Acknowledgements, Funding, Author
  contributions, Ethics declarations, Publisher's Note
- **Appendix A: Prompt Templates and Implementation Details** — A.1 atomic-claim
  decomposition, A.2 recursive-splitting, A.3 atomicity validation rules,
  A.4 claim-relation classification, A.6 verbal-confidence mapping,
  A.7 evidence-quality quantification, A.8 counterfactual perturbation rules,
  A.9 counterfactual-validation prompt, A.10 failure handling

## Mapping to TRACE-RAG stages

| Stage | FreeGuard |
|---|---|
| Claim extraction | Frozen LLM → SROM tuples, recursive splitting — **same slot schema and atomicity rule as TRACE-RAG §2.1** |
| Dependencies | Typed claim graph, same six edge types as proposal; soft risk propagation, not hard gating |
| Entailment | Frozen NLI → {sup, con, neu} per claim–evidence; quality-weighted |
| Aggregation | **Noisy-or** over evidence units at *claim* level (no facets); 5-state table with fixed thresholds |
| Calibration/decision | Thresholds from dev split only; **no conformal/α guarantee**; governance = preserve/revise/hedge/mark/downgrade; abstains on 3 of 5 states |

## Novelty threat

**Critical.** FreeGuard occupies: SROM extraction schema, typed claim
dependency graph, multi-source evidence verification, thresholded multi-state
governance with abstention. What it does *not* have: facet-level coverage
semantics (its unit is the whole claim), weakest-link/geometric-mean
aggregation, ancestor/coherent-factuality gating (risk propagates softly),
conformal α-controlled acceptance, insufficient-vs-refute mass separation
(n_eu is one of three symmetric channels), a *trained* cheap verifier head
(it uses frozen NLI + LLM components), and benchmark verification on
WICE/FEVER/SciFact (it evaluates mitigation outcomes on 5 benchmarks).

TRACE-RAG must explicitly differentiate on: formal evidence algebra,
statistical guarantees, and the trained shared-backbone verifier. Any claim
of "typed claim dependencies" or "claim-level governance" alone is now
unoriginal.
