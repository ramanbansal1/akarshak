"""Conflict and Coverage Aggregator (proposal §2.3) — pure numpy, no model deps.

Inputs per claim: chunk-level verdict probabilities p_i = (s_i, m_i, r_i) where
  s_i = P(supported), m_i = P(partially_supported) -> partial mass M,
  r_i = P(not_supported) -> refute/insufficient channel.
NOTE: WICE's not_supported conflates refutation and insufficient evidence; the
refute/unclear split is deferred to FEVER-style data. r_i is treated as the
"not established" mass here.

Passages = evidence chunks; facet = claim sub-component. On WICE each claim is
its own facet (k=1) and covers all its chunks, so the ancestor gate is vacuous —
but the code is facet-general and accepts dependency edges.

Class indices follow verifier_common.LABELS = [supported, partially_supported,
not_supported].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED, PARTIAL, NOT_SUP = 0, 1, 2
GAMMA = 0.5        # partial-support discount (§2.3)
EPS = 1e-6         # geometric-mean floor
LN3 = np.log(3.0)

SUPPORTED_ACTION = "SUPPORTED"
QUALIFY_ACTION = "QUALIFY_CONFLICT"
ABSTAIN_ACTION = "ABSTAIN"


@dataclass
class FacetStats:
    facet_id: int
    n_chunks: int = 0
    S: float = 0.0      # mean support mass
    R: float = 0.0      # mean not-supported mass
    M: float = 0.0      # mean partial mass
    eff: float = 0.0    # S + gamma * M  (0 when no covering chunks)
    contradicted: bool = False
    established: bool = False
    blocked_by: list = field(default_factory=list)   # ancestor facet ids that failed


@dataclass
class ClaimQuantities:
    support: float = 0.0          # geometric mean of eff_f (weakest link)
    contradiction: float = 0.0    # max_f R_f
    coverage: float = 0.0         # fraction of established facets
    conflict: float = 0.0         # 2*S_tot*R_tot/(S_tot+R_tot)
    agreement: float = 0.0        # 1 - H(p)/ln3 over chunk argmax labels
    S_tot: float = 0.0
    R_tot: float = 0.0
    facets: list = field(default_factory=list)       # list[FacetStats]
    missing_facets: list = field(default_factory=list)


def pool_facets(chunk_probs: np.ndarray, facet_of_chunk: np.ndarray,
                facet_ids: list[int], weights: np.ndarray | None = None,
                gamma: float = GAMMA) -> dict[int, FacetStats]:
    """chunk_probs [N,3]; facet_of_chunk [N] int; facet_ids: all facet ids present."""
    if weights is None:
        weights = np.ones(len(chunk_probs))
    stats = {f: FacetStats(facet_id=f) for f in facet_ids}
    for f in facet_ids:
        mask = facet_of_chunk == f
        idx = np.where(mask)[0]
        st = stats[f]
        st.n_chunks = int(mask.sum())
        if st.n_chunks == 0:
            st.eff = 0.0                      # no evidence is not weak support
            continue
        w = weights[idx]
        wsum = w.sum()
        p = chunk_probs[idx]
        st.S = float((w * p[:, SUPPORTED]).sum() / wsum)
        st.M = float((w * p[:, PARTIAL]).sum() / wsum)
        st.R = float((w * p[:, NOT_SUP]).sum() / wsum)
        st.eff = st.S + gamma * st.M
    return stats


def _ancestors(facet_id: int, edges: list[tuple[int, int]]) -> set[int]:
    """edges: (dependent, dependency) pairs — dependent's verdict requires
    dependency to be established. Returns all transitive dependencies."""
    adj = {}
    for dep, anc in edges:
        adj.setdefault(dep, set()).add(anc)
    seen, stack = set(), [facet_id]
    while stack:
        cur = stack.pop()
        for anc in adj.get(cur, ()):
            if anc not in seen:
                seen.add(anc)
                stack.append(anc)
    return seen


def facet_verdicts(stats: dict[int, FacetStats], tau_s: float, tau_r: float,
                   edges: list[tuple[int, int]] | None = None) -> None:
    """Apply contradicted/established rules in dependency order (§2.3 stage 2)."""
    edges = edges or []
    for f, st in stats.items():
        st.contradicted = st.R >= tau_r and st.R > st.S
    # establishment propagates: a facet is established iff eff >= tau_s,
    # not contradicted, has >=1 chunk, and every ancestor is established.
    memo, visiting = {}, set()

    def est(f):
        if f in memo:
            return memo[f]
        if f in visiting:                     # cycle -> treat as not established
            return False
        visiting.add(f)
        st = stats[f]
        ok = (st.eff >= tau_s and not st.contradicted and st.n_chunks >= 1
              and all(est(a) for a in _ancestors(f, edges) if a in stats))
        visiting.discard(f)
        memo[f] = ok
        return ok

    for f in stats:
        stats[f].established = est(f)
        failed = [a for a in _ancestors(f, edges)
                  if a in stats and not stats[a].established]
        stats[f].blocked_by = failed


def claim_quantities(stats: dict[int, FacetStats], chunk_probs: np.ndarray,
                     pool: str = "geomean", eps: float = EPS) -> ClaimQuantities:
    """pool='geomean' -> weakest-link support (§2.3); 'mean' -> arithmetic
    mean of facet eff (ablation). eps floors the geomean."""
    q = ClaimQuantities(facets=list(stats.values()))
    effs = np.array([st.eff for st in stats.values()])
    if pool == "geomean":
        q.support = float(np.exp(np.mean(np.log(np.maximum(effs, eps))))) \
            if len(effs) else 0.0
    elif pool == "mean":
        q.support = float(effs.mean()) if len(effs) else 0.0
    else:
        raise ValueError(f"unknown pool {pool!r}")
    q.contradiction = max((st.R for st in stats.values()), default=0.0)
    q.coverage = float(np.mean([st.established for st in stats.values()])) if stats else 0.0
    S_tot = float(np.mean([st.S for st in stats.values()])) if stats else 0.0
    R_tot = float(np.mean([st.R for st in stats.values()])) if stats else 0.0
    q.S_tot, q.R_tot = S_tot, R_tot
    q.conflict = 2 * S_tot * R_tot / (S_tot + R_tot) if (S_tot + R_tot) > 0 else 0.0
    if len(chunk_probs):
        labels = chunk_probs.argmax(-1)
        counts = np.bincount(labels, minlength=3) / len(labels)
        h = -(counts[counts > 0] * np.log(counts[counts > 0])).sum()
        q.agreement = float(1 - h / LN3)
    q.missing_facets = [st.facet_id for st in stats.values() if not st.established]
    return q


def aggregate(chunk_probs, facet_of_chunk, facet_ids, edges=None,
              weights=None, tau_s=0.5, tau_r=0.5, gamma=GAMMA,
              pool="geomean", eps=EPS):
    """Full §2.3 pipeline for one claim. chunk_probs: [N,3] array-like."""
    p = np.asarray(chunk_probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    p = p / p.sum(-1, keepdims=True)          # guard against unnormalized inputs
    foc = np.asarray(facet_of_chunk, dtype=int)
    stats = pool_facets(p, foc, list(facet_ids), weights, gamma)
    facet_verdicts(stats, tau_s, tau_r, edges)
    return claim_quantities(stats, p, pool=pool, eps=eps)


def decide_action(q: ClaimQuantities, tau_s: float) -> str:
    """§2.3 stage-4 decision table (retrieval/correction handled by caller)."""
    if any(st.contradicted and q.S_tot > 0 for st in q.facets):
        return QUALIFY_ACTION
    if q.coverage == 1.0 and q.support >= tau_s:
        return SUPPORTED_ACTION
    return ABSTAIN_ACTION
