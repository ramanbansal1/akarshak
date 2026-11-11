import numpy as np
import pytest

from src.entailment_network.calibration import (ece, risk_coverage,
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
