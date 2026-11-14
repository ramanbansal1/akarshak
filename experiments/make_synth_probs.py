"""Synthetic chunk-probs fixture with REAL P2 row identity — ticket 06 dev.

Builds a probs npz in exactly the dump_chunk_probs schema, but with
synthetic verifier outputs: real P2 row ids / labels / parent_ids (so the
P2-only guard passes and row counts match the real pool), plausible chunk
counts, and probs correlated with the WICE label:

  supported            -> mass concentrated on class 0
  partially_supported  -> mass split 0/1
  not_supported        -> mass concentrated on class 2

plus per-chunk noise so aggregation has something to pool. Use ONLY to
develop/test the ablation pipeline; delete or ignore the CSV it produces —
it is labelled synthetic=fixture, not a result.

Usage (repo root):
  python experiments/make_synth_probs.py --out results/probs_synth_p2.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.entailment_network.claimscore import strict_y
from src.entailment_network.data import load_universal
from src.entailment_network.splits import parent_of
from src.entailment_network.verifier_common import parse_label

# Dirichlet-ish concentration per WICE label (row label, not chunk label):
# how peaked each class's evidence probs are on average.
CONC = {0: [6.0, 1.5, 1.0], 1: [2.5, 3.5, 1.5], 2: [1.0, 1.5, 6.0]}


def synth_row_probs(label: int, n_chunks: int, rng: np.random.Generator):
    """[n_chunks,3] probs: label-peaked dirichlet + mild chunk noise."""
    conc = np.array(CONC[label], dtype=np.float64)
    # per-chunk jitter around the row-level concentration; gamma normalization
    # ~ dirichlet with per-chunk concentration
    alpha = conc[None, :] * rng.uniform(0.6, 1.4, size=(n_chunks, 1))
    g = rng.gamma(alpha, 1.0)
    p = g / g.sum(-1, keepdims=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="wice_p1p2", choices=["wice_p1p2", "wice"])
    ap.add_argument("--split", default="test",
                    help="pool split ('validation' maps to dev.jsonl)")
    a = ap.parse_args()

    rows = list(load_universal(a.name, a.split))
    rng = np.random.default_rng(a.seed)

    probs_all, chunk_row, labels, ys, row_ids, parents = [], [], [], [], [], []
    n_chunks = []
    for r, row in enumerate(rows):
        lab = parse_label(row["label"])
        k = int(min(8, 1 + rng.poisson(2.0)))       # 1..8 chunks, ~3 avg
        p = synth_row_probs(lab, k, rng)
        probs_all.append(p)
        chunk_row.append(np.full(k, r, dtype=np.int32))
        labels.append(lab)
        ys.append(strict_y(row["label"]))
        row_ids.append(str(row.get("id", "")))
        parents.append(parent_of(row))
        n_chunks.append(k)

    payload = dict(
        probs=np.concatenate(probs_all).astype(np.float32),
        chunk_row=np.concatenate(chunk_row).astype(np.int32),
        label=np.array(labels, dtype=np.int8),
        strict_y=np.array(ys, dtype=np.int8),
        row_id=np.array(row_ids),
        parent_id=np.array(parents),
        n_chunks=np.array(n_chunks, dtype=np.int32),
        meta=np.array([f"SYNTHETIC:{a.name}:{a.split}:seed{a.seed}"]),
    )
    np.savez_compressed(a.out, **payload)
    print(f"{len(rows)} rows, {len(payload['probs'])} chunks, "
          f"{len(set(parents))} parents -> {a.out}")
    print("WARNING: synthetic probs — pipeline fixture only, not a result.")


if __name__ == "__main__":
    main()
