"""CPU tests for eval_pipeline's certified selector block (ticket 03)."""

import numpy as np
import pytest

from src.entailment_network.claimscore import ClaimScore
from src.entailment_network.eval_pipeline import (_score_at_tau,
                                                  certified_block,
                                                  claim_records)


def _recs(n_parents=20, per=2, seed=0):
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n_parents):
        for j in range(per):
            good = rng.rand() < 0.6
            recs.append(ClaimScore(u=float(rng.rand()) if good else 0.05,
                                   y=int(good), instance_id=f"p{i}"))
    return recs


def test_certified_block_structure():
    calib, test = _recs(30, seed=1), _recs(15, seed=2)
    out = certified_block(calib, test, alpha=0.20, delta=0.10, m_max=6)
    assert out["n_cal_instances"] == 30 and out["n_test_instances"] == 15
    assert set(out["baselines"]) == {"empirical", "rcps"}
    assert set(out["certified"]) == {"bernstein_micro", "bernstein_macro",
                                     "hoeffding_micro", "hoeffding_macro"}
    for name, blk in out["certified"].items():
        assert "calib" in blk and "test" in blk
        assert {"tau", "accepted", "coverage"} <= set(blk["test"])
        if "U" in blk["calib"]:
            assert blk["calib"]["U"] <= 0 or blk["calib"]["tau"] == float("inf")


def test_certified_block_respects_parent_grouping():
    # all claims in ONE parent -> n_instances=1; bound must be vacuous
    recs = [ClaimScore(u=0.9, y=1, instance_id="only") for _ in range(10)]
    out = certified_block(recs, recs, alpha=0.5, delta=0.10, m_max=6)
    assert out["n_cal_instances"] == 1
    for blk in out["certified"].values():
        assert blk["calib"]["tau"] == float("inf") or \
            blk["calib"]["n_instances"] == 1


def test_score_at_tau():
    u = np.array([0.9, 0.8, 0.1])
    c = np.array([True, False, True])
    r = _score_at_tau(u, c, 0.5)
    assert r["accepted"] == 2 and r["coverage"] == pytest.approx(2 / 3)
    assert r["error_rate"] == pytest.approx(0.5)
    r = _score_at_tau(u, c, float("inf"))
    assert r["accepted"] == 0 and r["error_rate"] is None


def test_claim_records_k1_path():
    # fabricated row_stats: 3 chunks, one supporting -> label supported
    row = {"id": "wice:p1-0", "label": "supported",
           "supporting_sentences": [[0, 1]]}
    probs = np.array([[0.9, 0.05, 0.05]] * 3)
    stats = [dict(probs=probs, spans=[(0, 1), (1, 2), (2, 3)],
                  label=0, row=row)]
    recs = claim_records(stats, tau_s=0.5, tau_r=0.5)
    assert len(recs) == 1
    r = recs[0]
    assert r.instance_id == "p1" and r.y == 1
    # all chunks supported-ish -> eff high, established -> u == support
    assert r.u == pytest.approx(float(np.exp(np.mean(np.log(
        np.maximum(0.9 + 0.5 * 0.05, 1e-12))))))
