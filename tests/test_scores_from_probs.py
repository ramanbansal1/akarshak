"""Ticket 08 prep: probs->ClaimScore npz path feeding resplit_runner real mode."""

import numpy as np

from experiments.scores_from_probs import (records_for, stats_and_parents)
from src.entailment_network.claimscore import read_npz, write_npz
from experiments.resplit_runner import real_arrays


def _probs_npz(path, rows_spec, seed=0):
    """rows_spec: list of (parent_id, row_id, label0/1/2, strict_y)."""
    rng = np.random.default_rng(seed)
    probs, chunk_row, labels, ys, rids, pars, nch = [], [], [], [], [], [], []
    for r, (pid, rid, lab, sy) in enumerate(rows_spec):
        k = int(rng.integers(1, 4))
        # label-peaked probs so u is meaningful
        alpha = np.array([[5.0, 1.0, 1.0], [1.0, 5.0, 1.0],
                          [1.0, 1.0, 5.0]][lab])
        p = rng.dirichlet(alpha, size=k)
        probs.append(p)
        chunk_row.append(np.full(k, r, dtype=np.int32))
        labels.append(lab); ys.append(sy); rids.append(rid); pars.append(pid)
        nch.append(k)
    np.savez_compressed(
        path, probs=np.concatenate(probs).astype(np.float32),
        chunk_row=np.concatenate(chunk_row).astype(np.int32),
        label=np.array(labels, dtype=np.int8),
        strict_y=np.array(ys, dtype=np.int8),
        row_id=np.array(rids), parent_id=np.array(pars),
        n_chunks=np.array(nch, dtype=np.int32), meta=np.array(["test"]))
    return str(path)


def test_scores_from_probs_roundtrip(tmp_path):
    # two parents x two subclaims each; parent "pa" all supported, "pb" mixed
    spec = [("pa", "pa-0", 0, 1), ("pa", "pa-1", 0, 1),
            ("pb", "pb-0", 2, 0), ("pb", "pb-1", 1, 0)]
    calib = _probs_npz(tmp_path / "calib.npz", spec, seed=1)
    pool = _probs_npz(tmp_path / "pool.npz", spec, seed=2)

    stats, parents, z = stats_and_parents(calib)
    assert len(stats) == 4 and parents == ["pa", "pa", "pb", "pb"]

    # fixed taus for determinism
    recs = records_for(stats, parents, z, tau_s=0.5, tau_r=0.5)
    assert len(recs) == 4
    assert recs[0].instance_id == "pa" and recs[0].y == 1
    assert recs[2].y == 0 and recs[3].y == 0
    # supported rows should get higher u than unsupported
    assert recs[0].u > recs[2].u

    out = tmp_path / "scores.pool.npz"
    write_npz(recs, out)
    back = read_npz(out)
    assert back["instance_ids"] == ["pa", "pb"]
    arr = real_arrays([str(out)])
    assert len(arr["scores"]) == 4 and len(arr["instance_ids"]) == 2
