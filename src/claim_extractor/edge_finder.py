"""Dependency edge finder: candidate generation + rule-based typing.

Candidate generation uses reciprocal rank fusion (RRF, k=60) over three
rankers:
  1. BM25 (pure-python Okapi BM25, no external deps)
  2. entity-overlap (Jaccard on extracted entity sets)
  3. dense similarity — optional callable dense_score(i, j) -> float in [0,1]

Typing is rule-based (extends RuleBasedExtractor._infer_edges heuristics):
  shared entities            -> entity_coreference
  differing temporal exprs   -> temporal_constraint
  differing numerical values -> numerical_constraint
  causal markers             -> causal_dependency
semantic_entailment / semantic_conflict are emitted only via an optional
`llm_typer` callback — no confident rule exists, so they are not guessed.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .atomic_claim_extractor import AtomicClaim, DependencyEdge

_TOKEN = re.compile(r"[a-z0-9]+")
_CAUSAL = re.compile(r"\b(because|therefore|thus|hence|leads? to|due to|as a result|consequently|so that)\b", re.I)
_RRF_K = 60


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ------------------------------------------------------------------ BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / max(self.n, 1)
        self.df = defaultdict(int)
        self.tf = []
        for d in docs:
            tf = defaultdict(int)
            for t in d:
                tf[t] += 1
            self.tf.append(tf)
            for t in tf:
                self.df[t] += 1
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5))
                    for t, f in self.df.items()}

    def scores(self, query: list[str], doc_idx: int) -> float:
        tf, dl = self.tf[doc_idx], self.dl[doc_idx]
        s = 0.0
        for t in set(query):
            if t not in tf:
                continue
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf[t] * f * (self.k1 + 1) / denom
        return s

    def ranking(self, query: list[str], exclude: int) -> list[tuple[int, float]]:
        scored = [(i, self.scores(query, i)) for i in range(self.n) if i != exclude]
        scored.sort(key=lambda x: -x[1])
        return scored


# ------------------------------------------------------------------ rankers -> RRF
def _rank_list(pairs: list[tuple[int, float]]) -> dict[int, int]:
    return {idx: r for r, (idx, _) in enumerate(pairs)}


def fuse_candidates(rankings: list[list[tuple[int, float]]], top_k: int = 10,
                    rrf_k: int = _RRF_K) -> list[tuple[int, float]]:
    """rankings: one ranked [(doc_idx, score)] list per ranker. Returns fused
    [(doc_idx, rrf_score)] sorted desc, truncated to top_k."""
    fused = defaultdict(float)
    for ranking in rankings:
        for idx, rank in _rank_list(ranking).items():
            fused[idx] += 1.0 / (rrf_k + rank + 1)
    out = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    return out


# ------------------------------------------------------------------ typing
def _edge_type(a: AtomicClaim, b: AtomicClaim) -> tuple[str, str] | None:
    shared = set(a.entities) & set(b.entities)
    if shared:
        return "entity_coreference", f"shared: {', '.join(sorted(shared))}"
    if a.temporal_expr and b.temporal_expr and a.temporal_expr != b.temporal_expr:
        return "temporal_constraint", f"{a.temporal_expr} vs {b.temporal_expr}"
    if a.numerical_value and b.numerical_value and a.numerical_value != b.numerical_value:
        return "numerical_constraint", f"{a.numerical_value} vs {b.numerical_value}"
    if _CAUSAL.search(a.text) or _CAUSAL.search(b.text):
        return "causal_dependency", "causal marker present"
    return None


def find_edges(claims: list[AtomicClaim], dense_score=None, top_k: int = 5,
               min_score: float = 0.0,
               llm_typer=None) -> list[DependencyEdge]:
    """claims: extracted AtomicClaims (same response). dense_score(i, j) optional.
    llm_typer(a, b) optional -> (dep_type, note) override for the 2 LLM-only types.
    Returns DependencyEdges; deterministic ordering by (source, target)."""
    n = len(claims)
    if n < 2:
        return []
    toks = [_tokens(c.text) for c in claims]
    bm25 = BM25(toks)
    ent_sets = [set(c.entities) for c in claims]

    edges = []
    for i, a in enumerate(claims):
        rankers = [bm25.ranking(toks[i], i)]
        # entity-overlap ranker
        ov = sorted(((j, len(ent_sets[i] & ent_sets[j]) /
                      max(len(ent_sets[i] | ent_sets[j]), 1))
                     for j in range(n) if j != i and ent_sets[i] & ent_sets[j]),
                    key=lambda x: -x[1])
        if ov:
            rankers.append(ov)
        if dense_score is not None:
            ds = sorted(((j, dense_score(i, j)) for j in range(n) if j != i),
                        key=lambda x: -x[1])
            rankers.append(ds)
        for j, score in fuse_candidates(rankers, top_k=top_k):
            if score <= min_score:
                continue
            b = claims[j]
            typed = _edge_type(a, b)
            if typed is None and llm_typer is not None:
                typed = llm_typer(a, b)
            if typed is None:
                continue
            dep_type, note = typed
            edges.append(DependencyEdge(a.claim_id, b.claim_id, dep_type,
                                        note=f"{note} | rrf={score:.4f}"))
    return sorted(edges, key=lambda e: (e.source_id, e.target_id, e.dep_type))
