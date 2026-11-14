"""ClaimScore npz from chunk-probs dumps — probs->scores path for ticket 08.

Mirror of scores_from_cache.py but starting from dump_chunk_probs npz files
instead of feats caches: calibrates tau_s/tau_r on a "calib" probs pool via
eval_pipeline.calibrate(), aggregates each row k=1, and writes ClaimScore
npz in the exact format resplit_runner --mode real consumes.

This is the local/CPU twin of the Kaggle scores step — same u_j definition
(u = support * 1{coverage == 1}), same calibrated thresholds, so swapping
probs npz -> real dump_chunk_probs output is the only change needed.

Usage (repo root):
  python experiments/scores_from_probs.py \
      --calib results/probs.wice_p1p2_test.npz \
      --pairs wice:validation=results/probs.wice_validation.npz \
             wice:test=results/probs.wice_test.npz \
      --out results/scores --alpha 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.entailment_network.aggregator import aggregate
from src.entailment_network.claimscore import (ClaimScore, strict_y,
                                               u_from_quantities, write_npz)
from src.entailment_network.controller import decide
from src.entailment_network.eval_pipeline import calibrate
from src.entailment_network.verifier_common import parse_label


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib", required=True, help="calib probs npz (P2)")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="name=probs.npz pools to score")
    p.add_argument("--out", required=True, help="output npz prefix")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.10)
    return p.parse_args()


def stats_and_parents(path):
    """Return (stats for calibrate/evaluate, parent_ids) from a probs npz."""
    z = np.load(path, allow_pickle=False)
    probs, chunk_row = z["probs"], z["chunk_row"]
    n = len(z["label"])
    stats, parents = [], []
    for r in range(n):
        p = np.asarray(probs[chunk_row == r], dtype=np.float64)
        stats.append(dict(probs=p, spans=[],
                          label=parse_label(int(z["label"][r])), row={}))
        parents.append(str(z["parent_id"][r]))
    return stats, parents, z


def records_for(stats, parents, z, tau_s, tau_r):
    recs = []
    for s, pid, rid, sy in zip(
            stats, parents, z["row_id"].astype(str), z["strict_y"]):
        p = s["probs"]
        q = aggregate(p, np.zeros(len(p), int), [0] if len(p) else [],
                      tau_s=tau_s, tau_r=tau_r)
        recs.append(ClaimScore(
            u=u_from_quantities(q.support, q.coverage),
            y=int(sy), instance_id=pid, action=decide(q, tau_s).action,
            claim_id=rid))
    return recs


def main():
    a = args()
    stats, _, _ = stats_and_parents(a.calib)
    cal = calibrate(stats, a.alpha, a.beta, a.delta)
    tau_s, tau_r = cal["tau_s"], cal["tau_r"]
    print(f"calib {a.calib} -> tau_s={tau_s:.4f} tau_r={tau_r:.4f} "
          f"n_contradicted={cal['n_contradicted']}")

    out = Path(a.out)
    for spec in a.pairs:
        name, path = spec.split("=", 1)
        stats, parents, z = stats_and_parents(path)
        recs = records_for(stats, parents, z, tau_s, tau_r)
        suffix = name.replace(":", "_")
        write_npz(recs, f"{out}.{suffix}.npz")
        ys = np.array([r.y for r in recs])
        us = np.array([r.u for r in recs])
        print(f"{spec}: {len(recs)} claims, {len({r.instance_id for r in recs})} "
              f"instances, base_err={1 - ys.mean():.3f}, mean_u={us.mean():.3f}")


if __name__ == "__main__":
    main()
