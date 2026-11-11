"""Uncertainty-Aware Decision Controller (proposal §2.4) — skeleton scope.

Maps aggregator ClaimQuantities to one of three actions:
  SUPPORTED        - coverage == 1 and support(C) >= tau_s_hat
  QUALIFY_CONFLICT - some facet contradicted while two-sided mass exists
  ABSTAIN          - everything else; report names the missing/blocked facets

"Retrieve more evidence" and "correct the claim" collapse into ABSTAIN for the
skeleton (retrieval budget = 0, no corrector exists) — the Decision.report
keeps the full audit trail so a real controller can be dropped in later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aggregator import (ABSTAIN_ACTION, ClaimQuantities, QUALIFY_ACTION,
                         SUPPORTED_ACTION, decide_action)


@dataclass
class Decision:
    action: str
    report: dict = field(default_factory=dict)


def decide(q: ClaimQuantities, tau_s: float, tau_r: float | None = None,
           retrieval_budget: int = 0) -> Decision:
    """tau_s: calibrated supported-threshold; tau_r: refute threshold (already
    applied inside aggregate() via facet_verdicts — kept for report symmetry)."""
    action = decide_action(q, tau_s)
    if action == ABSTAIN_ACTION and retrieval_budget > 0:
        # skeleton: no retriever wired; reserve the branch for the real controller
        action = ABSTAIN_ACTION
    report = dict(
        action=action,
        support=q.support,
        contradiction=q.contradiction,
        coverage=q.coverage,
        conflict=q.conflict,
        agreement=q.agreement,
        S_tot=q.S_tot,
        R_tot=q.R_tot,
        missing_facets=q.missing_facets,
        facets=[dict(facet_id=st.facet_id, n_chunks=st.n_chunks, S=st.S, R=st.R,
                     M=st.M, eff=st.eff, contradicted=st.contradicted,
                     established=st.established, blocked_by=st.blocked_by)
                for st in q.facets],
    )
    return Decision(action=action, report=report)
