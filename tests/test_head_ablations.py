"""Ticket 07 prep tests: head ablations on a synthetic FeatureCache."""

import numpy as np
import pytest
import torch

from src.entailment_network.verifier_common import save_cache, FeatureCache


def make_cache(path, n_rows=40, d=32, L=4, seed=0, separable=True):
    """Synthetic feats cache: he last-layer ~ class template + noise so the
    bilinear head and centroid baseline have real signal to find."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    templates = torch.tensor(rng.normal(0, 1, (3, d)) * 3.0)
    hq, he, y_chunks, n_ev, seq_row, row_y = [], [], [], [], [], []
    chunk = 0
    for r in range(n_rows):
        k = int(rng.integers(1, 4))
        lab = int(rng.integers(0, 3))
        row_y.append(lab)
        seq_row.append(r)
        n_ev.append(k)
        hq.append(torch.randn(1, L, d) * 0.5)
        for _ in range(k):
            cl = lab if separable and rng.random() < 0.8 else int(
                rng.integers(0, 3))
            v = templates[cl] + torch.randn(d) * 0.5
            he.append(torch.randn(1, L, d) * 0.1)
            he[-1][0, -1, :] = v                     # signal in last layer
            y_chunks.append(cl)
            chunk += 1
    # save_cache wants states list [1+N, L, d] per sequence; rebuild per-row
    states, labels = [], []
    i = 0
    for r in range(n_rows):
        k = n_ev[r]
        st = torch.cat([hq[r]] + he[i:i + k], dim=0)   # [1+k, L, d]
        states.append(st)
        labels.append(y_chunks[i:i + k])
        i += k
    meta = dict(backbone="synth", layers=[L - 1], data_cfg={})
    save_cache(path, states, labels, meta,
               extra=dict(row_y=torch.tensor(row_y),
                          seq_row=torch.tensor(seq_row)))
    return str(path)


def test_cache_roundtrip(tmp_path):
    p = make_cache(tmp_path / "feats_train.pt", n_rows=10)
    c = FeatureCache(p)
    assert c.n_rows == 10 and len(c.he) == int(c.n_ev.sum())
    assert c.chunk_row.max() < 10


def test_mode_rank_and_centroid(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    d.mkdir()
    for split in ("train", "validation", "test"):
        make_cache(d / f"feats_{split}.pt", n_rows=30, seed=hash(split) % 1000)
    import experiments.head_ablations as ha
    a = ha.args.__wrapped__() if hasattr(ha.args, "__wrapped__") else None
    import argparse
    ns = argparse.Namespace(mode="all", dir=str(d), ranks="8,16",
                            epochs=3, bs=64, lr=1e-3, wd=0.01,
                            class_power=0.5, seed=0, probs_calib=None,
                            probs_pairs=[], out=str(tmp_path / "out"),
                            device="cpu")
    ha.mode_rank(ns, "cpu")
    ha.mode_centroid(ns, "cpu")
    rank_csv = (tmp_path / "out" / "head_ablations_rank.csv").read_text()
    assert "kind,size" in rank_csv
    assert "8" in rank_csv and "16" in rank_csv
    cent_csv = (tmp_path / "out" / "head_ablations_centroid.csv").read_text()
    assert "row_f1" in cent_csv
    # separable synthetic data: both should beat chance (row_f1 > 0.2)
    import csv, io
    for row in csv.DictReader(io.StringIO(cent_csv)):
        assert float(row["row_f1"]) > 0.2
