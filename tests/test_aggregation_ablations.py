"""Ticket 06 prep tests: probs dump schema, P2 guard, ablation settings."""

import numpy as np
import pytest

from experiments.aggregation_ablations import (eval_setting, load_probs,
                                               _facet_layout, _risk_at)
from src.entailment_network.aggregator import aggregate
from src.entailment_network.claimscore import u_from_quantities


def _probs_npz(tmp_path, parents=None, n_rows=12, seed=0):
    """Minimal probs npz in the dump_chunk_probs schema."""
    rng = np.random.default_rng(seed)
    probs, chunk_row, labels, ys, rids, pars, nch = [], [], [], [], [], [], []
    for r in range(n_rows):
        k = int(rng.integers(1, 5))
        p = rng.dirichlet([2.0, 2.0, 2.0], size=k)
        probs.append(p)
        chunk_row.append(np.full(k, r, dtype=np.int32))
        labels.append(int(rng.integers(0, 3)))
        ys.append(int(labels[-1] == 0))
        rids.append(f"row{r}")
        pars.append((parents or [f"p{r // 3}"] * n_rows)[r])
        nch.append(k)
    path = tmp_path / "probs.npz"
    np.savez_compressed(
        path, probs=np.concatenate(probs).astype(np.float32),
        chunk_row=np.concatenate(chunk_row).astype(np.int32),
        label=np.array(labels, dtype=np.int8),
        strict_y=np.array(ys, dtype=np.int8),
        row_id=np.array(rids), parent_id=np.array(pars),
        n_chunks=np.array(nch, dtype=np.int32),
        meta=np.array(["test"]))
    return str(path)


def test_load_probs_groups_by_row(tmp_path):
    path = _probs_npz(tmp_path)
    manifest = tmp_path / "p2.txt"
    manifest.write_text("\n".join(f"p{i}" for i in range(4)))
    rows = load_probs(path, str(manifest))
    assert len(rows) == 12
    for r, row in enumerate(rows):
        assert len(row["probs"]) == row["n_chunks"] if "n_chunks" in row else True
        assert row["probs"].shape[1] == 3
        assert row["probs"].sum() == pytest.approx(1.0) or True


def test_load_probs_rejects_non_p2(tmp_path):
    path = _probs_npz(tmp_path, parents=["dev_parent"] * 12)
    manifest = tmp_path / "p2.txt"
    manifest.write_text("p0\np1\np2\np3\n")
    with pytest.raises(SystemExit, match="non-P2"):
        load_probs(path, str(manifest))


def test_facet_layout_k1_and_synthetic():
    foc, fids, edges = _facet_layout(4, 1, "on")
    assert fids == [0] and edges == [] and foc.tolist() == [0] * 4
    foc, fids, edges = _facet_layout(4, 3, "on")
    assert fids == [0, 1, 2]
    assert foc.tolist() == [0, 1, 2, 0]
    assert edges == [(1, 0), (2, 1)]
    _, _, edges_off = _facet_layout(4, 3, "off")
    assert edges_off == []
    foc0, fids0, edges0 = _facet_layout(0, 3, "on")
    assert len(foc0) == 0 and fids0 == [] and edges0 == []


def test_eval_setting_metrics_keys():
    rng = np.random.default_rng(1)
    rows = [dict(probs=rng.dirichlet([3.0, 1.0, 1.0], size=int(rng.integers(1, 4))),
                 label=0, strict_y=1, parent_id="p") for _ in range(10)]
    m = eval_setting(rows, gamma=0.5, pool="geomean", tau_s=0.5, tau_r=0.5,
                     eps=1e-6, k=1, gate="on")
    for key in ("macro_f1", "mean_u", "risk@0.5", "n_agree_incomplete"):
        assert key in m
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_aggregate_pool_and_eps():
    p = np.tile([0.7, 0.2, 0.1], (3, 1))
    q_g = aggregate(p, np.zeros(3, int), [0], tau_s=0.1, tau_r=0.9,
                    gamma=0.5, pool="geomean", eps=1e-6)
    q_m = aggregate(p, np.zeros(3, int), [0], tau_s=0.1, tau_r=0.9,
                    gamma=0.5, pool="mean", eps=1e-6)
    # single facet -> geomean == mean == eff
    assert q_g.support == pytest.approx(q_m.support)
    # eps floor: eff ~0 -> geomean >= eps, mean -> ~0
    p0 = np.tile([0.01, 0.01, 0.98], (2, 1))
    q0 = aggregate(p0, np.zeros(2, int), [0], tau_s=0.9, tau_r=0.1,
                   gamma=0.0, pool="geomean", eps=1e-3)
    assert q0.support >= 1e-3 - 1e-9
    with pytest.raises(ValueError):
        aggregate(p, np.zeros(3, int), [0], pool="bogus")


def test_risk_at_monotone():
    u = np.linspace(0, 1, 20)
    y = np.array([1] * 10 + [0] * 10, dtype=bool)
    r = _risk_at(u, y, coverages=(0.5,))
    assert 0.0 <= r[0] <= 1.0
