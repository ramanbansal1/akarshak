"""CPU tests for experiments/resplit_runner.py (ticket 04)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import resplit_runner as rr  # noqa: E402


def _synthetic_arrays(n_inst=12, per=3, seed=0):
    rng = np.random.RandomState(seed)
    scores, correct, inst = [], [], []
    for i in range(n_inst):
        for _ in range(per):
            good = rng.rand() < 0.6
            scores.append(float(rng.rand()) if good else 0.05)
            correct.append(bool(good))
            inst.append(i)
    return dict(scores=np.array(scores),
                correct=np.array(correct, dtype=bool),
                instances=np.array(inst, dtype=int),
                instance_ids=[f"p{i}" for i in range(n_inst)])


def test_run_resplits_cells_and_rates():
    arr = _synthetic_arrays()
    recs = rr.run_resplits(arr, n_resplits=4, n_cals=[4, 8], alphas=[0.2, 0.35],
                           grids=[10], selectors=["empirical", "bernstein_macro"],
                           seed=1)
    # 2 n_cal x 2 alpha x (empirical@grid10 + bernstein_macro@grid10)
    assert len(recs) == 2 * 2 * 2
    for r in recs:
        assert 0.0 <= r["cert_rate"] <= 1.0
        assert 0.0 <= r["violation_rate"] <= 1.0
        assert r["n_resplits"] == 4
        assert r["n_cal_instances"] if "n_cal_instances" in r else True


def test_oracle_arrays_real_data_shape():
    arr = rr.oracle_arrays()
    assert len(arr["instance_ids"]) == 707
    assert len(arr["scores"]) == 1907
    sizes = np.bincount(arr["instances"])
    assert sizes.max() <= 6 and sizes.min() >= 1


def test_noisy_oracle_corrupts_scores_not_labels():
    arr = rr.noisy_oracle_arrays(0.5, seed=0)
    base = rr.oracle_arrays()
    assert not np.allclose(arr["scores"], base["scores"])
    assert (arr["correct"] == base["correct"]).all()
    assert set(arr["instance_ids"]) == set(base["instance_ids"])


def test_real_arrays_merges_npz(tmp_path):
    from src.entailment_network.claimscore import (ClaimScore, write_npz)
    recs_a = [ClaimScore(u=0.9, y=1, instance_id="pa0"),
              ClaimScore(u=0.1, y=0, instance_id="pa1")]
    recs_b = [ClaimScore(u=0.8, y=1, instance_id="pb0"),
              ClaimScore(u=0.7, y=0, instance_id="pa0")]
    fa, fb = tmp_path / "a.npz", tmp_path / "b.npz"
    write_npz(recs_a, fa)
    write_npz(recs_b, fb)
    arr = rr.real_arrays([str(fa), str(fb)])
    assert arr["instance_ids"] == ["pa0", "pa1", "pb0"]
    assert len(arr["scores"]) == 4
    # pa0's two claims share one instance slot
    pa0_slot = arr["instance_ids"].index("pa0")
    assert (arr["instances"][arr["scores"] == 0.9] == pa0_slot).all()
    assert (arr["instances"][arr["scores"] == 0.7] == pa0_slot).all()


def test_csv_roundtrip_and_gate(tmp_path):
    recs = [{"mode": "oracle", "n_cal": 100.0, "alpha": 0.2, "grid": 50.0,
             "bound": "bernstein", "sel_mode": "macro",
             "selector": "bernstein_macro", "cert_rate": 1.0,
             "mean_cal_coverage": 0.5, "mean_test_coverage": 0.5,
             "test_coverage_std": 0.01, "violation_rate": 0.0,
             "mean_test_error": 0.0, "n_resplits": 10},
            {"mode": "real", "n_cal": 100.0, "alpha": 0.2, "grid": 50.0,
             "bound": "bernstein", "sel_mode": "macro",
             "selector": "bernstein_macro", "cert_rate": 0.4,
             "mean_cal_coverage": 0.3, "mean_test_coverage": 0.3,
             "test_coverage_std": 0.02, "violation_rate": 0.0,
             "mean_test_error": 0.05, "n_resplits": 10}]
    fo, fr = tmp_path / "o.csv", tmp_path / "r.csv"
    rr.write_csv([recs[0]], fo)
    rr.write_csv([recs[1]], fr)
    text = rr.gate_summary(fo, fr)
    assert "verifier-limited" in text
    # oracle-fail cell -> n-limited
    recs[0]["cert_rate"] = 0.1
    rr.write_csv([recs[0]], fo)
    assert "n-limited" in rr.gate_summary(fo, fr)
