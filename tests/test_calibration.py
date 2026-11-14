import numpy as np
import pytest

from src.entailment_network.calibration import (ece, risk_coverage,
                                                select_tau_hoeffding,
                                                select_tau_rcps,
                                                select_tau_refute,
                                                select_tau_support)


class TestSelectTauSupport:
    def test_guarantee_holds_on_tuning_data(self):
        rng = np.random.default_rng(0)
        scores = rng.random(500)
        correct = scores > 0.6                      # high score -> likely correct
        res = select_tau_support(scores, correct, alpha=0.10)
        assert res["accepted"] > 0
        assert res["error_rate"] <= 0.10

    def test_smallest_threshold_maximizes_coverage(self):
        # perfect separation: score>0.5 correct, below incorrect
        scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
        correct = np.array([True, True, True, False, False, False])
        res = select_tau_support(scores, correct, alpha=0.10)
        assert res["tau"] == pytest.approx(0.7)
        assert res["accepted"] == 3

    def test_no_acceptable_threshold_abstains_all(self):
        scores = np.array([0.5, 0.6])
        correct = np.array([False, False])
        res = select_tau_support(scores, correct, alpha=0.10)
        assert res["accepted"] == 0 and res["coverage"] == 0.0


class TestSelectTauRcps:
    def test_perfect_separation_accepts_correct_region(self):
        # 100 high-score correct + 100 low-score wrong: the accepted set should
        # cover (most of) the correct region — small n cannot certify alpha.
        scores = np.concatenate([np.linspace(0.9, 0.6, 100),
                                 np.linspace(0.4, 0.1, 100)])
        correct = np.concatenate([np.ones(100, bool), np.zeros(100, bool)])
        res = select_tau_rcps(scores, correct, alpha=0.10, delta=0.10)
        assert res["accepted"] >= 90
        assert res["error_rate"] <= 0.15

    def test_abstains_when_no_safe_threshold(self):
        scores = np.array([0.5, 0.6])
        correct = np.array([False, False])
        res = select_tau_rcps(scores, correct, alpha=0.10, delta=0.10)
        assert res["accepted"] == 0 and res["tau"] == float("inf")

    def test_more_conservative_than_empirical(self):
        rng = np.random.default_rng(0)
        scores = rng.random(400)
        correct = scores > 0.55
        emp = select_tau_support(scores, correct, alpha=0.10)
        rcps = select_tau_rcps(scores, correct, alpha=0.10, delta=0.10)
        assert rcps["tau"] >= emp["tau"] - 1e-9
        assert rcps["accepted"] <= emp["accepted"]

    def test_empty_input(self):
        res = select_tau_rcps(np.array([]), np.array([], dtype=bool))
        assert res["accepted"] == 0 and res["tau"] == float("inf")


class TestSelectTauHoeffding:
    def _instances(self, n_inst=60, m=4, seed=0):
        rng = np.random.default_rng(seed)
        scores, correct, inst = [], [], []
        for a in range(n_inst):
            s = rng.random(m)
            c = s > 0.55
            scores += s.tolist(); correct += c.tolist()
            inst += [a] * m
        return (np.array(scores), np.array(correct, dtype=bool),
                np.array(inst))

    def test_perfect_scores_accept_everything(self):
        # instance-level Hoeffding is conservative: needs many instances.
        scores, correct, inst = self._instances(n_inst=2000, m=3)
        scores = np.where(correct, 0.9, 0.1)         # perfect separation
        res = select_tau_hoeffding(scores, correct, inst,
                                   alpha=0.10, delta=0.10)
        assert res["accepted"] == int(correct.sum())
        assert res["error_rate"] == 0.0
        assert res["n_instances"] == 2000

    def test_conservative_vs_empirical(self):
        scores, correct, inst = self._instances()
        emp = select_tau_support(scores, correct, alpha=0.10)
        hf = select_tau_hoeffding(scores, correct, inst,
                                  alpha=0.10, delta=0.10)
        assert hf["tau"] >= emp["tau"] - 1e-9
        assert hf["accepted"] <= emp["accepted"]

    def test_abstains_when_no_certified_threshold(self):
        # every claim wrong -> L_tau(a) = N_tau(a) > 0 for any nonempty accept
        scores = np.full(40, 0.5)
        correct = np.zeros(40, dtype=bool)
        inst = np.repeat(np.arange(10), 4)
        res = select_tau_hoeffding(scores, correct, inst,
                                   alpha=0.10, delta=0.10)
        assert res["accepted"] == 0 and res["tau"] == float("inf")

    def test_instance_unit_not_claim(self):
        # same claims, but all in ONE instance -> n=1, bound is huge -> abstain
        scores = np.linspace(0.9, 0.1, 40)
        correct = scores > 0.5
        inst = np.zeros(40, dtype=int)
        res = select_tau_hoeffding(scores, correct, inst,
                                   alpha=0.10, delta=0.10)
        assert res["n_instances"] == 1
        assert res["accepted"] == 0                     # cannot certify at n=1

    def test_empty_input(self):
        res = select_tau_hoeffding(np.array([]), np.array([], dtype=bool),
                                   np.array([]))
        assert res["accepted"] == 0 and res["tau"] == float("inf")


class TestSelectTauRefute:
    def test_controls_false_rejections(self):
        scores = np.array([0.9, 0.8, 0.4, 0.3])
        refuted = np.array([True, True, False, False])
        res = select_tau_refute(scores, refuted, beta=0.10)
        assert res["false_rejection_rate"] <= 0.10
        assert res["tau"] == pytest.approx(0.8)


class TestECE:
    def test_perfect_calibration_zero(self):
        # one bin, confidence == empirical accuracy -> ece 0
        conf = np.array([0.5] * 20)
        corr = np.array([True] * 10 + [False] * 10)
        assert ece(conf, corr) == pytest.approx(0.0, abs=1e-9)

    def test_overconfident_positive(self):
        conf = np.array([0.99] * 10)
        corr = np.array([False] * 10)
        assert ece(conf, corr) > 0.9


class TestRiskCoverage:
    def test_curve_endpoints(self):
        scores = np.linspace(0.1, 1.0, 50)
        correct = scores > 0.5
        rc = risk_coverage(scores, correct, n_points=10)
        pts = rc["points"]
        assert pts[-1]["coverage"] == pytest.approx(1.0)
        # best prefix (top confidences) should have lower risk than full set
        assert pts[0]["risk"] <= pts[-1]["risk"] + 1e-9
