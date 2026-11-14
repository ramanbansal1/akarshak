"""Dump per-row chunk probabilities from feature caches + trained head.

Ticket 06 enabler: aggregation ablations need the raw chunk-level verifier
probabilities [C,3], not just the scalar u_j that scores_from_cache emits.
This script runs the VerdictModel head over feats_{split}.pt caches once and
writes one npz per pool:

  probs_{name}_{split}.npz
    probs      [C,3] float32  chunk-level softmax probs (row-major order)
    chunk_row  [C]    int32    chunk -> dataset row index
    label      [R]    int8     WICE label via parse_label (0/1/2)
    strict_y   [R]    int8     1 iff supported
    row_id     [R]    str      universal row id
    parent_id  [R]    str      WICE parent claim id (i.i.d. unit)
    n_chunks   [R]    int32    chunks per row (0 allowed)

CPU-friendly: VerdictModel is ~3M params; no backbone, no datasets download
(labels/ids come from data/universal/<name>/<split>.jsonl).

Usage (repo root):
  python experiments/dump_chunk_probs.py \
      --ckpt runs/wice_p1p2/head.pt \
      --pairs uni:wice_p1p2:test=runs/wice_p1p2/feats_test.pt \
             uni:wice:validation=runs/wice_pool/feats_validation.pt \
      --out results/probs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.entailment_network.claimscore import strict_y
from src.entailment_network.data import load_universal
from src.entailment_network.splits import parent_of
from src.entailment_network.verifier_common import (FeatureCache,
                                                    VerdictModel,
                                                    parse_label,
                                                    predict_cache)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="head.pt from train.py")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="name:split=feats.pt entries to dump")
    p.add_argument("--out", required=True, help="output npz prefix")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _parse(spec: str):
    name_split, path = spec.split("=", 1)
    name, split = name_split.split(":", 1)
    return name, split, path


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["model_cfg"]
    model = VerdictModel(cfg["d"], cfg["n_layers"], K=cfg.get("K", 3),
                         r=cfg.get("r", 256))
    model.load_state_dict(ck["state_dict"])
    return model.to(dev)


def dump(cache: FeatureCache, model, dev, name: str, split: str):
    """Return the npz payload dict for one pool."""
    rows = list(load_universal(name, split))
    assert len(rows) == cache.n_rows, \
        f"{name}:{split} rows {len(rows)} != cache rows {cache.n_rows}"

    probs = predict_cache(model, cache, dev).numpy().astype(np.float32)
    chunk_row = cache.chunk_row.numpy().astype(np.int32)

    n_chunks = np.bincount(chunk_row, minlength=cache.n_rows).astype(np.int32)
    labels = np.array([parse_label(r["label"]) for r in rows], dtype=np.int8)
    return dict(
        probs=probs,
        chunk_row=chunk_row,
        label=labels,
        strict_y=np.array([strict_y(r["label"]) for r in rows],
                          dtype=np.int8),
        row_id=np.array([str(r.get("id", "")) for r in rows]),
        parent_id=np.array([parent_of(r) for r in rows]),
        n_chunks=n_chunks,
        meta=np.array([f"{name}:{split}"]),
    )


def main():
    a = args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, dev)

    out = Path(a.out)
    for spec in a.pairs:
        name, split, path = _parse(spec)
        payload = dump(FeatureCache(path), model, dev, name, split)
        suffix = f"{name.replace(':', '_')}_{split}"
        np.savez_compressed(f"{out}.{suffix}.npz", **payload)
        print(f"{spec}: {len(payload['label'])} rows, "
              f"{len(payload['probs'])} chunks, "
              f"{len(set(payload['parent_id'].tolist()))} parents "
              f"-> {out}.{suffix}.npz")


if __name__ == "__main__":
    main()
