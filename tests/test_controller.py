import numpy as np

from src.entailment_network.aggregator import aggregate
from src.entailment_network.controller import decide


def q_for(probs, tau_s=0.5, tau_r=0.5):
    return aggregate(np.array([probs]), np.zeros(1, int), [0],
                     tau_s=tau_s, tau_r=tau_r)


class TestController:
    def test_supported_action_and_report(self):
        d = decide(q_for([0.9, 0.1, 0.0]), tau_s=0.5)
        assert d.action == "SUPPORTED"
        for key in ("support", "contradiction", "coverage", "conflict",
                    "agreement", "missing_facets", "facets"):
            assert key in d.report

    def test_abstain_reports_missing_facets(self):
        # facet 1 has middling masses: eff < tau_s and not contradicted ->
        # unestablished but no conflict -> ABSTAIN naming the missing facet
        q = aggregate(np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.1]]),
                      np.array([0, 1]), [0, 1], tau_s=0.8, tau_r=0.5)
        d = decide(q, tau_s=0.8)
        assert d.action == "ABSTAIN"
        assert d.report["missing_facets"] == [1]

    def test_qualify_conflict_action(self):
        d = decide(q_for([0.45, 0.0, 0.55], tau_s=0.9, tau_r=0.5), tau_s=0.9)
        assert d.action == "QUALIFY_CONFLICT"

    def test_retrieval_budget_does_not_change_skeleton(self):
        # no retriever wired: budget > 0 still abstains honestly
        q = q_for([0.3, 0.3, 0.4], tau_s=0.9, tau_r=0.9)
        d = decide(q, tau_s=0.9, retrieval_budget=3)
        assert d.action == "ABSTAIN"
