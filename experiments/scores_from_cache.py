"""ClaimScore npz from feature caches + trained head — no backbone rerun.

Loads feats_{split}.pt caches (hidden states), runs the small VerdictModel
head via predict_cache, aggregates chunk probs per row (k=1 WICE path),
and writes ClaimScore arrays in the format experiments/resplit_runner.py
--mode real consumes.

tau_s/tau_r for the aggregator's coverage gate are calibrated on a chosen
"calib" cache+split (protocol: the P2 tuning pool) via eval_pipeline's
calibrate(); every other cache is scored with those same thresholds so u_j
is fixed across cal/test resplits.

Usage (repo root):
  python experiments/scores_from_cache.py \
      --ckpt runs/wice_p1p2/head.pt \
      --calib uni:wice_p1p2:test=runs/wice_p1p2/feats_test.pt \
      --pairs uni:wice_p1p2:validation=runs/wice_p1p2/feats_validation.pt \
             uni:wice:validation=runs/wice/feats_validation.pt \
             uni:wice:test=runs/wice/feats_test.pt \
      --out results/scores --alpha 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.entailment_network.aggregator import aggregate
from src.entailment_network.calibration import select_tau_rcps
from src.entailment_network.claimscore import (ClaimScore, instance_id_of,
                                               record_from_row, strict_y,
                                               u_from_quantities, write_npz)
from src.entailment_network.controller import decide
from src.entailment_network.data import load_universal
from src.entailment_network.eval_pipeline import calibrate
from src.entailment_network.verifier_common import (FeatureCache,
                                                    VerdictModel,
                                                    parse_label,
                                                    predict_cache)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="head.pt from train.py")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="name:split=feats.pt entries to score")
    p.add_argument("--calib", required=True,
                   help="name:split=feats.pt used to calibrate tau_s/tau_r")
    p.add_argument("--out", required=True, help="output npz prefix")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _parse(spec: str):
    # specs look like "uni:wice_p1p2:test=<feats.pt>" — strip the uni: scheme
    # prefix, then split dataset name from split on the LAST colon
    name_split, path = spec.split("=", 1)
    ns = name_split[4:] if name_split.startswith("uni:") else name_split
    name, split = ns.rsplit(":", 1)
    return name, split, path


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["model_cfg"]
    model = VerdictModel(cfg["d"], cfg["n_layers"], K=cfg.get("K", 3),
                         r=cfg.get("r", 256))
    model.load_state_dict(ck["state_dict"])
    return model.to(dev), ck


def row_probs(cache: FeatureCache, model, dev):
    """Chunk probs [M,3] -> per-row probs list (row order = dataset order)."""
    probs = predict_cache(model, cache, dev).numpy()
    chunk_row = cache.chunk_row.numpy()
    return [probs[chunk_row == r] for r in range(cache.n_rows)]


def stats_for(name, split, probs_per_row):
    ds = load_universal(name, split)
    rows = list(ds)
    assert len(rows) == len(probs_per_row), \
        f"{name}:{split} rows {len(rows)} != cache rows {len(probs_per_row)}"
    stats = []
    for r, p in zip(rows, probs_per_row):
        stats.append(dict(probs=np.asarray(p, dtype=np.float64),
                          spans=[], label=parse_label(r["label"]), row=r))
    return stats


def records_for(stats, tau_s, tau_r):
    recs = []
    for s in stats:
        p = np.asarray(s["probs"], dtype=np.float64)
        if len(p) == 0:
            q = aggregate(np.zeros((0, 3)), np.zeros(0, int), [])
        else:
            q = aggregate(p, np.zeros(len(p), int), [0],
                          tau_s=tau_s, tau_r=tau_r)
        recs.append(ClaimScore(
            u=u_from_quantities(q.support, q.coverage),
            y=strict_y(s["row"].get("label", "")),
            instance_id=instance_id_of(s["row"]),
            action=decide(q, tau_s).action,
            claim_id=str(s["row"].get("id", ""))))
    return recs


def main():
    a = args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(a.ckpt, dev)

    cn, cs, cp = _parse(a.calib)
    calib_cache = FeatureCache(cp)
    calib_stats = stats_for(cn, cs, row_probs(calib_cache, model, dev))
    cal = calibrate(calib_stats, a.alpha, a.beta, a.delta)
    tau_s, tau_r = cal["tau_s"], cal["tau_r"]
    print(f"calib {cn}:{cs} -> tau_s={tau_s:.4f} tau_r={tau_r:.4f} "
          f"n_contradicted={cal['n_contradicted']}")

    out = Path(a.out)
    for spec in a.pairs:
        name, split, path = _parse(spec)
        cache = FeatureCache(path)
        stats = stats_for(name, split, row_probs(cache, model, dev))
        recs = records_for(stats, tau_s, tau_r)
        suffix = f"{name.replace(':', '_')}_{split}"
        write_npz(recs, f"{out}.{suffix}.npz")
        ys = np.array([r.y for r in recs])
        us = np.array([r.u for r in recs])
        print(f"{spec}: {len(recs)} claims, {len({r.instance_id for r in recs})} "
              f"instances, base_err={1 - ys.mean():.3f}, "
              f"mean_u={us.mean():.3f} -> {out}.{suffix}.npz")


if __name__ == "__main__":
    main()
