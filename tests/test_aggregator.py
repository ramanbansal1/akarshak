import numpy as np
import pytest

from src.entailment_network.aggregator import (ABSTAIN_ACTION, GAMMA,
                                                QUALIFY_ACTION, SUPPORTED_ACTION,
                                                aggregate, claim_quantities,
                                                decide_action, facet_verdicts,
                                                pool_facets)


def probs(*rows):
    return np.array(rows, dtype=float)


class TestPoolFacets:
    def test_empty_coverage_is_zero_not_weak_support(self):
        stats = pool_facets(probs([0.9, 0.05, 0.05]), np.array([1]), [0, 1])
        assert stats[0].n_chunks == 0 and stats[0].eff == 0.0
        assert stats[1].S == pytest.approx(0.9)

    def test_partial_mass_discounted(self):
        # all chunks fully partial -> eff = 0 + gamma*1
        stats = pool_facets(probs([0.0, 1.0, 0.0]), np.array([0]), [0])
        assert stats[0].eff == pytest.approx(GAMMA)

    def test_weighted_mean(self):
        p = probs([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        stats = pool_facets(p, np.array([0, 0]), [0], weights=np.array([3.0, 1.0]))
        assert stats[0].S == pytest.approx(0.75)
        assert stats[0].R == pytest.approx(0.25)


class TestFacetVerdicts:
    def test_contradicted_requires_R_ge_tau_and_beats_S(self):
        stats = pool_facets(probs([0.2, 0.0, 0.8]), np.array([0]), [0])
        facet_verdicts(stats, tau_s=0.5, tau_r=0.5)
        assert stats[0].contradicted and not stats[0].established

        stats = pool_facets(probs([0.6, 0.0, 0.4]), np.array([0]), [0])
        facet_verdicts(stats, tau_s=0.5, tau_r=0.5)
        assert not stats[0].contradicted          # R < S

    def test_established_needs_chunks_and_threshold(self):
        stats = pool_facets(probs([0.9, 0.1, 0.0]), np.array([0]), [0, 1])
        facet_verdicts(stats, tau_s=0.5, tau_r=0.5)
        assert stats[0].established and not stats[1].established  # empty coverage

    def test_ancestor_gate_blocks_dependent(self):
        # facet 1 depends on facet 0 (edge 1->0); facet 0 unestablished
        p = np.vstack([probs([0.1, 0.0, 0.9]), probs([0.9, 0.1, 0.0])])
        foc = np.array([0, 1])
        stats = pool_facets(p, foc, [0, 1])
        facet_verdicts(stats, tau_s=0.5, tau_r=0.5, edges=[(1, 0)])
        assert not stats[1].established
        assert stats[1].blocked_by == [0]

        # when ancestor is established, dependent establishes
        p2 = np.vstack([probs([0.9, 0.1, 0.0]), probs([0.9, 0.1, 0.0])])
        stats2 = pool_facets(p2, foc, [0, 1])
        facet_verdicts(stats2, tau_s=0.5, tau_r=0.5, edges=[(1, 0)])
        assert stats2[0].established and stats2[1].established


class TestClaimQuantities:
    def test_geometric_mean_weakest_link(self):
        p = np.vstack([probs([0.9, 0.1, 0.0]), probs([0.0, 1.0, 0.0])])
        foc = np.array([0, 1])
        stats = pool_facets(p, foc, [0, 1])
        facet_verdicts(stats, tau_s=0.4, tau_r=0.9)
        q = claim_quantities(stats, p)
        # eff_0 = 0.95, eff_1 = 0.5 -> support = sqrt(0.95*0.5)
        assert q.support == pytest.approx(np.sqrt(0.95 * 0.5), rel=1e-3)
        assert q.coverage == pytest.approx(1.0)   # tau_r=0.9 -> nothing contradicted

    def test_conflict_high_only_two_sided(self):
        p = np.vstack([probs([0.5, 0.0, 0.5]), probs([0.5, 0.0, 0.5])])
        stats = pool_facets(p, np.zeros(2, int), [0])
        facet_verdicts(stats, tau_s=0.9, tau_r=0.9)
        q = claim_quantities(stats, p)
        # S=R=0.5 -> 2*.5*.5/(.5+.5) = 0.5; formula maxes ~0.5 since S+R<=1
        assert q.conflict == pytest.approx(0.5)

        p2 = probs([0.9, 0.05, 0.05])
        stats2 = pool_facets(p2, np.zeros(1, int), [0])
        facet_verdicts(stats2, tau_s=0.9, tau_r=0.9)
        q2 = claim_quantities(stats2, p2)
        assert q2.conflict < 0.2                    # one-sided -> low

    def test_source_agreement(self):
        unanimous = probs([0.9, 0.05, 0.05], [0.9, 0.05, 0.05])
        stats = pool_facets(unanimous, np.zeros(2, int), [0])
        facet_verdicts(stats, 0.5, 0.5)
        q = claim_quantities(stats, unanimous)
        assert q.agreement == pytest.approx(1.0, abs=1e-6)

        mixed = probs([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        stats2 = pool_facets(mixed, np.zeros(3, int), [0])
        facet_verdicts(stats2, 0.5, 0.5)
        q2 = claim_quantities(stats2, mixed)
        assert q2.agreement == pytest.approx(0.0, abs=1e-6)   # uniform over 3 labels


class TestDecideAction:
    def test_supported(self):
        p = probs([0.9, 0.1, 0.0])
        q = aggregate(p, np.zeros(1, int), [0], tau_s=0.5, tau_r=0.5)
        assert decide_action(q, 0.5) == SUPPORTED_ACTION

    def test_abstain_on_weak_support(self):
        p = probs([0.3, 0.3, 0.4])
        q = aggregate(p, np.zeros(1, int), [0], tau_s=0.8, tau_r=0.9)
        assert decide_action(q, 0.8) == ABSTAIN_ACTION

    def test_qualify_on_two_sided_conflict(self):
        # S and R both substantial; contradicted facet + S_tot > 0
        p = probs([0.45, 0.0, 0.55])
        q = aggregate(p, np.zeros(1, int), [0], tau_s=0.9, tau_r=0.5)
        assert decide_action(q, 0.9) == QUALIFY_ACTION
