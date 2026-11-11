"""Calibration utilities (proposal §2.4) — pure numpy.

- select_tau_support: smallest threshold on a support score such that the
  empirical error rate among accepted claims <= alpha (risk-controlling).
- select_tau_refute: analogous for the contradiction channel (false rejections).
- ece: expected calibration error over confidence bins.
- risk_coverage: coverage vs. selective risk curve.
"""

from __future__ import annotations

import numpy as np


def _candidate_thresholds(scores: np.ndarray, grid: int = 200) -> np.ndarray:
    s = np.unique(np.asarray(scores, dtype=np.float64))
    if len(s) <= grid:
        return s
    return np.quantile(s, np.linspace(0, 1, grid))


def select_tau_support(scores, correct, alpha: float = 0.10) -> dict:
    """Pick smallest tau s.t. P(error | score >= tau) <= alpha.

    scores: per-claim support confidence (e.g. eff_f). correct: bool, claim truly
    supported. Returns dict(tau, accepted, error_rate, coverage).
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    best = None
    for tau in _candidate_thresholds(scores):
        acc = scores >= tau
        n = int(acc.sum())
        if n == 0:
            continue
        err = 1.0 - float(correct[acc].mean())
        if err <= alpha:
            best = dict(tau=float(tau), accepted=n,
                        error_rate=err, coverage=n / len(scores))
            break                       # thresholds ascend -> first hit = smallest tau
    if best is None:                    # nothing satisfies alpha; safest = reject all
        best = dict(tau=float("inf"), accepted=0, error_rate=0.0, coverage=0.0)
    return best


def select_tau_refute(scores, is_refuted_true, beta: float = 0.10) -> dict:
    """Pick tau_r on contradiction mass s.t. among {score >= tau}, the fraction
    that is NOT truly refuted/contradicted <= beta (controls false rejections).

    scores: per-claim refute mass (e.g. max_f R_f). is_refuted_true: bool.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_refuted_true = np.asarray(is_refuted_true, dtype=bool)
    best = None
    for tau in _candidate_thresholds(scores):        # ascend: first hit = most sensitive
        acc = scores >= tau
        n = int(acc.sum())
        if n == 0:
            continue
        false_rate = 1.0 - float(is_refuted_true[acc].mean())
        if false_rate <= beta:
            best = dict(tau=float(tau), flagged=n,
                        false_rejection_rate=false_rate,
                        coverage=n / len(scores))
            break
    if best is None:
        best = dict(tau=float("inf"), flagged=0, false_rejection_rate=0.0,
                    coverage=0.0)
    return best


def ece(confidence, correct, n_bins: int = 10) -> float:
    """Expected calibration error. confidence in [0,1], correct bool."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(float(correct[m].mean()) - float(confidence[m].mean()))
    return float(total)


def risk_coverage(scores, correct, n_points: int = 20) -> dict:
    """Selective prediction curve: sort by confidence desc, report
    (coverage, risk) at deciles. scores: confidence; correct: bool."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(-scores)
    c_sorted, s_sorted = correct[order], scores[order]
    points = []
    for frac in np.linspace(1.0 / len(scores), 1.0, min(n_points, len(scores))):
        k = max(1, int(round(frac * len(scores))))
        points.append(dict(coverage=k / len(scores),
                           risk=1.0 - float(c_sorted[:k].mean())))
    return dict(points=points)
