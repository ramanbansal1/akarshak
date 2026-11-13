"""Calibration utilities (proposal §2.4) — pure numpy.

- select_tau_support: smallest threshold on a support score such that the
  empirical error rate among accepted claims <= alpha (risk-controlling).
- select_tau_refute: analogous for the contradiction channel (false rejections).
- select_tau_rcps: claim-level Clopper-Pearson grid (comparison baseline).
- select_tau_hoeffding: instance-level Hoeffding bound on
  L_tau(a) = F_tau(a) - alpha*N_tau(a) (the certified method, method.tex
  Eq. ucb); the sampling unit is the instance, not the claim.
- ece: expected calibration error over confidence bins.
- risk_coverage: coverage vs. selective risk curve.
"""

from __future__ import annotations

import math

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


def _cp_upper(e, n, level: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial proportion.

    e errors observed among n accepted; returns U s.t. P(p <= U) >= 1-level.
    Falls back to a Wilson bound if scipy is unavailable."""
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import beta as beta_dist
        return float(beta_dist.ppf(1.0 - level, e + 1, n - e))
    except Exception:
        phat = e / n
        z = 1.645
        den = 1 + z * z / n
        centre = (phat + z * z / (2 * n)) / den
        half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
        return float(min(1.0, centre + half))


def select_tau_rcps(scores, correct, alpha: float = 0.10,
                    delta: float = 0.10, grid: int = 50) -> dict:
    """Risk-controlled acceptance threshold (docs/risk_control.md §4).

    Conditional accepted-error target: P(wrong | score >= tau) <= alpha with
    probability >= 1-delta over the calibration draw. A G-point grid of
    candidate thresholds is tested with Bonferroni-corrected
    Clopper-Pearson upper bounds (per-test level delta/G); the loosest safe
    threshold is selected. Returns dict(tau, accepted, error_rate, coverage,
    delta, grid, n_safe).
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    n = len(scores)
    if n == 0:
        return dict(tau=float("inf"), accepted=0, error_rate=0.0,
                    coverage=0.0, delta=delta, grid=grid, n_safe=0)
    order = np.argsort(-scores, kind="stable")
    ss = scores[order]
    n_acc = np.arange(1, n + 1)
    n_err = np.cumsum((~correct[order]).astype(np.int64))
    G = min(grid, n)
    idx = np.unique(np.linspace(0, n - 1, G).astype(int))
    lvl = delta / len(idx)
    try:
        from scipy.special import betaincinv
        upper = betaincinv(n_err[idx] + 1, n_acc[idx] - n_err[idx],
                           1.0 - lvl)
    except Exception:
        upper = np.array([_cp_upper(int(e), int(n), lvl)
                          for e, n in zip(n_err[idx], n_acc[idx])])
    safe = idx[upper <= alpha]
    if len(safe) == 0:
        return dict(tau=float("inf"), accepted=0, error_rate=0.0,
                    coverage=0.0, delta=delta, grid=grid, n_safe=0)
    k = safe[-1]                                     # largest accepted set
    tau = float(ss[k])
    acc = scores >= tau
    return dict(tau=tau, accepted=int(acc.sum()),
                error_rate=float((~correct[acc]).mean()),
                coverage=float(acc.mean()), delta=delta, grid=grid,
                n_safe=len(safe))


def _fixed_grid(grid=50) -> np.ndarray:
    """Data-INDEPENDENT threshold grid (required for the certificate: the
    grid must be fixed before calibration, not derived from calibration
    scores). Accepts an int (uniform grid on (0,1)) or an explicit array."""
    if isinstance(grid, (int, np.integer)):
        return np.linspace(0.02, 0.98, int(grid))
    return np.unique(np.asarray(grid, dtype=np.float64))


def _instance_stats(scores, correct, inst_idx, n, taus, alpha, mode):
    """Per-instance losses L_a(tau) for every grid point.

    mode="micro": L_a = F_a - alpha*N_a in [-alpha*m_max, m_max].
    mode="macro": L_a = F_a/max(N_a,1) - alpha in [-alpha, 1-alpha] (per-answer
    false-discovery proportion; ablation only — controls a different,
    macro-averaged quantity).
    Returns L [n_tau, n_inst].
    """
    L = np.empty((len(taus), n))
    for t, tau in enumerate(taus):
        acc = scores >= tau
        if not acc.any():
            L[t] = 0.0
            continue
        N = np.bincount(inst_idx[acc], minlength=n).astype(np.float64)
        F = np.bincount(inst_idx[acc & ~correct], minlength=n).astype(np.float64)
        L[t] = (F / np.maximum(N, 1.0) - alpha) if mode == "macro" \
            else (F - alpha * N)
    return L


def select_tau_hoeffding(scores, correct, instances, alpha: float = 0.10,
                         delta: float = 0.10, grid=50, m_max: int | None = None,
                         bound: str = "bernstein",
                         mode: str = "micro") -> dict:
    """Instance-level certified threshold (method.tex Sec. decision / prop:cert).

    For each candidate tau and each instance a:
        N_tau(a) = #{claims of a with score >= tau}
        F_tau(a) = #{accepted claims of a that are wrong}
        L_tau(a) = F_tau(a) - alpha*N_tau(a)          (mode="micro")
        L_tau(a) = F_tau(a)/max(N_tau(a),1) - alpha   (mode="macro", ablation)
    With n i.i.d. instances, a union bound over the fixed |T|-grid certifies
    E[L_tau] <= 0 simultaneously for all tau; the smallest certified tau is
    returned (max coverage), tau=inf if none. bound="bernstein" uses the
    variance-adaptive empirical Bernstein radius; "hoeffding" is the
    distribution-free worst-case fallback.

    scores/correct: per-claim arrays. instances: per-claim instance ids;
    the i.i.d. unit is the instance, so claims within one instance may be
    arbitrarily correlated. m_max should be the extractor's hard cap on
    claims per instance (NOT estimated from test data); inferred from the
    calibration split only when not provided.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    inst = np.asarray(instances)
    empty = dict(tau=float("inf"), accepted=0, error_rate=0.0, coverage=0.0,
                 delta=delta, grid=None, n_instances=0, m_max=0,
                 bound=bound, mode=mode)
    if len(scores) == 0 or len(inst) == 0:
        return empty
    uniq, inst_idx = np.unique(inst, return_inverse=True)
    n = len(uniq)
    if m_max is None:
        m_max = int(np.bincount(inst_idx).max())
    taus = _fixed_grid(grid)
    width = (m_max * (1.0 + alpha)) if mode == "micro" else 1.0
    Lmin = (-alpha * m_max) if mode == "micro" else (-alpha)
    L = _instance_stats(scores, correct, inst_idx, n, taus, alpha, mode)
    best = None
    for t, tau in enumerate(taus):                       # ascending: max coverage first
        col = L[t]
        Lbar = float(col.mean())
        # scale column to [0,1] using the THEORETICAL range for the bound
        z = np.clip((col - Lmin) / width, 0.0, 1.0)
        if bound == "hoeffding":
            radius_z = math.sqrt(math.log(2 * len(taus) / delta) / (2.0 * n))
        else:
            if n < 2:
                radius_z = 1.0
            else:
                v = float(z.var(ddof=1))
                c = math.log(2 * len(taus) / delta)
                radius_z = (math.sqrt(2.0 * v * c / n)
                            + 7.0 * c / (3.0 * (n - 1)))
        U = Lbar + width * radius_z
        if U <= 0:
            acc = scores >= tau
            best = dict(tau=float(tau), accepted=int(acc.sum()),
                        error_rate=float((~correct[acc]).mean()),
                        coverage=float(acc.mean()), U=float(U), Lbar=Lbar)
            break
    if best is None:
        return dict(empty, n_instances=n, m_max=m_max, width=width,
                    grid=len(taus))
    return dict(best, delta=delta, grid=len(taus), n_instances=n,
                m_max=m_max, width=width, bound=bound, mode=mode)


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
