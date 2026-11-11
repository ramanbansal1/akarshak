import math

import pytest

from src.claim_extractor.atomic_claim_extractor import AtomicClaim
from src.claim_extractor.edge_finder import BM25, _edge_type, find_edges, fuse_candidates


def claim(text, entities=None, temporal="", numerical=""):
    return AtomicClaim(text=text, entities=entities or [],
                       temporal_expr=temporal, numerical_value=numerical)


class TestBM25:
    def test_repeated_term_outranks_absent(self):
        bm = BM25([["the", "cat", "sat"], ["dog", "ran", "fast"],
                   ["cat", "cat", "cat"]])
        r = bm.ranking(["cat"], exclude=0)
        assert r[0][0] == 2                       # doc 2 has cat x3

    def test_excludes_self(self):
        bm = BM25([["a", "b"], ["a", "b"]])
        r = bm.ranking(["a"], exclude=0)
        assert all(i != 0 for i, _ in r)


class TestRRF:
    def test_fusion_prefers_consistent_high_rank(self):
        # doc 1 ranks first in both rankers; doc 2 first in one, absent other
        fused = fuse_candidates([[(1, 1.0), (2, 0.5)], [(1, 1.0)]], top_k=5)
        assert fused[0][0] == 1
        scores = dict(fused)
        assert scores[1] > scores[2]


class TestEdgeType:
    def test_shared_entities_coreference(self):
        a = claim("X happened", entities=["Paris"])
        b = claim("It occurred", entities=["Paris"])
        assert _edge_type(a, b)[0] == "entity_coreference"

    def test_differing_temporal(self):
        a = claim("Built in 1985", temporal="1985")
        b = claim("Opened in 1990", temporal="1990")
        assert _edge_type(a, b)[0] == "temporal_constraint"

    def test_differing_numerical(self):
        a = claim("Costs 5 million", numerical="5 million")
        b = claim("Costs 7 million", numerical="7 million")
        assert _edge_type(a, b)[0] == "numerical_constraint"

    def test_causal_marker(self):
        a = claim("The dam failed because of the flood")
        b = claim("The flood was severe")
        assert _edge_type(a, b)[0] == "causal_dependency"

    def test_unrelated_returns_none(self):
        a = claim("The cat sat")
        b = claim("Quantum chromodynamics")
        assert _edge_type(a, b) is None


class TestFindEdges:
    def test_no_edges_for_single_claim(self):
        assert find_edges([claim("Only one claim here")]) == []

    def test_coreference_edge_emitted(self):
        claims = [claim("Paris is the capital of France", entities=["Paris"]),
                  claim("Paris hosted the olympics", entities=["Paris"]),
                  claim("Bananas are yellow fruits", entities=["Bananas"])]
        edges = find_edges(claims, top_k=3)
        pairs = {(e.source_id, e.target_id) for e in edges}
        # the two Paris claims share an entity -> at least one direction typed
        paris_ids = {claims[0].claim_id, claims[1].claim_id}
        assert any({e.source_id, e.target_id} == paris_ids and
                   e.dep_type == "entity_coreference" for e in edges)

    def test_deterministic_and_no_self_edges(self):
        claims = [claim(f"Claim number {i} about topic{i}", entities=[f"E{i}"])
                  for i in range(4)]
        e1 = find_edges(claims)
        e2 = find_edges(claims)
        assert [(e.source_id, e.target_id, e.dep_type) for e in e1] == \
               [(e.source_id, e.target_id, e.dep_type) for e in e2]
        assert all(e.source_id != e.target_id for e in e1)
