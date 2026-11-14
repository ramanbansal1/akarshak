"""Aggregation + decision ablations on P2 only — ticket 06.

Consumes a chunk-probs dump from experiments/dump_chunk_probs.py (the ONLY
input — swap the npz to go from synthetic fixture to real head outputs):

  python experiments/aggregation_ablations.py \
      --probs results/probs.wice_p1p2_test.npz --out results/ablation_aggregation.csv

Sweeps (ticket 06): gamma x {geomean,mean} pooling x tau_s/tau_r around the
P2-calibrated defaults x eps. tau_s/tau_r defaults come from
eval_pipeline.calibrate() run on the same pool (P2 is the tuning pool —
tuning on it is the point; dev/test are never touched: every parent_id must
appear in data/wice/p2_parents.txt).

WICE rows are k=1 facets (claim = own facet), so the ancestor gate and k>1
facet slices are SYNTHETIC ONLY: --synthetic_facets K assigns chunks
round-robin to K facets (optionally chained by --ancestor_gate on) and marks
every output row synthetic=1. Real k=1 runs are synthetic=0.

Metrics per setting: 3-class row macro-F1 (controller action -> label),
mean u, frac u>0, risk@coverage {0.5,0.75,0.9} on strict y, and the two
challenge-slice counts (agree-yet-incomplete, ancestor-blocked).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.entailment_network.aggregator import aggregate
from src.entailment_network.claimscore import u_from_quantities
from src.entailment_network.controller import decide
from src.entailment_network.eval_pipeline import calibrate, ACTION_TO_LABEL
from src.entailment_network.verifier_common import macro_f1

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
POOLS = ["geomean", "mean"]
TAU_FACTORS = [0.8, 1.0, 1.25]
EPSES = [1e-6, 1e-3, 1e-2]
P2_MANIFEST = (Path(__file__).resolve().parents[1] / "data" / "wice"
               / "p2_parents.txt")

CSV_FIELDS = ["gamma", "pool", "tau_s", "tau_r", "eps", "n_facets",
              "ancestor_gate", "synthetic", "macro_f1", "f1_supported",
              "mean_u", "frac_u_pos", "risk@0.5", "risk@0.75", "risk@0.9",
              "n_agree_incomplete", "n_ancestor_blocked", "n_rows"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--probs", required=True, help="probs npz from dump_chunk_probs")
    p.add_argument("--p2-manifest", default=str(P2_MANIFEST))
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--synthetic_facets", type=int, default=1,
                   help="K>1 assigns chunks round-robin to K facets (synthetic)")
    p.add_argument("--ancestor_gate", choices=["on", "off"], default="on",
                   help="chain edges among synthetic facets (synthetic only)")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_probs(path: str, p2_manifest: str, limit=None):
    """npz -> per-row chunk probs + metadata; asserts P2-only parents."""
    z = np.load(path, allow_pickle=False)
    probs, chunk_row = z["probs"], z["chunk_row"]
    n_rows = len(z["label"])
    allowed = set(Path(p2_manifest).read_text().split())
    parents = z["parent_id"].astype(str)
    bad = sorted(set(parents.tolist()) - allowed)
    if bad:
        raise SystemExit(f"probs contain {len(bad)} non-P2 parents "
                         f"(e.g. {bad[:3]}); P2-only guard tripped")
    rows = []
    for r in range(n_rows):
        p = probs[chunk_row == r]
        rows.append(dict(probs=np.asarray(p, dtype=np.float64),
                         label=int(z["label"][r]),
                         strict_y=int(z["strict_y"][r]),
                         parent_id=str(parents[r])))
    return rows[:limit] if limit else rows


def _stats_for_calibrate(rows):
    return [dict(probs=r["probs"], spans=[], label=r["label"], row={})
            for r in rows]


def _facet_layout(n_chunks: int, k: int, gate: bool):
    """(facet_of_chunk, facet_ids, edges) for one row.

    k=1: the real WICE path (claim = own facet, no edges).
    k>1: SYNTHETIC round-robin facets; gate=on chains i -> i-1.
    """
    if n_chunks == 0:
        return np.zeros(0, int), [], []
    if k <= 1:
        return np.zeros(n_chunks, int), [0], []
    foc = np.arange(n_chunks) % k
    edges = [(i, i - 1) for i in range(1, k)] if gate == "on" else []
    return foc, list(range(k)), edges


def _risk_at(u, y, coverages=(0.5, 0.75, 0.9)):
    order = np.argsort(-np.asarray(u))
    ys = np.asarray(y, dtype=bool)[order]
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(ys))))
        out.append(1.0 - float(ys[:k].mean()))
    return out


def eval_setting(rows, gamma, pool, tau_s, tau_r, eps, k, gate):
    us, preds, ys, strict = [], [], [], []
    n_incomplete = n_blocked = 0
    for r in rows:
        n = len(r["probs"])
        foc, fids, edges = _facet_layout(n, k, gate)
        q = aggregate(r["probs"], foc, fids, edges=edges,
                      tau_s=tau_s, tau_r=tau_r, gamma=gamma,
                      pool=pool, eps=eps)
        us.append(u_from_quantities(q.support, q.coverage))
        preds.append(ACTION_TO_LABEL[decide(q, tau_s).action])
        ys.append(r["label"])
        strict.append(r["strict_y"])
        if q.agreement >= 0.99 and q.coverage < 1.0:
            n_incomplete += 1
        if any(st.blocked_by for st in q.facets):
            n_blocked += 1
    us, preds = np.array(us), np.array(preds)
    f1, per = macro_f1(np.array(preds), np.array(ys))
    risks = _risk_at(us, strict)
    return dict(macro_f1=round(float(f1), 4),
                f1_supported=round(float(per.get("supported", 0.0)), 4),
                mean_u=round(float(us.mean()), 4),
                frac_u_pos=round(float((us > 0).mean()), 4),
                **{"risk@0.5": round(risks[0], 4),
                   "risk@0.75": round(risks[1], 4),
                   "risk@0.9": round(risks[2], 4)},
                n_agree_incomplete=n_incomplete,
                n_ancestor_blocked=n_blocked)


def main():
    a = args()
    rows = load_probs(a.probs, a.p2_manifest, a.limit)
    print(f"{len(rows)} P2 rows loaded; calibrating default tau_s/tau_r "
          f"(alpha={a.alpha}, beta={a.beta})")

    cal = calibrate(_stats_for_calibrate(rows), a.alpha, a.beta, a.delta)
    tau_s0, tau_r0 = cal["tau_s"], cal["tau_r"]
    print(f"defaults: tau_s={tau_s0:.4f} tau_r={tau_r0:.4f} "
          f"n_contradicted={cal['n_contradicted']}")

    synthetic = a.synthetic_facets > 1
    gate = a.ancestor_gate
    k = a.synthetic_facets
    results = []
    for gamma in GAMMAS:
        for pool in POOLS:
            for fs in TAU_FACTORS:
                for fr in TAU_FACTORS:
                    for eps in EPSES:
                        m = eval_setting(rows, gamma, pool,
                                         float(np.clip(tau_s0 * fs, 0.01, 0.99)),
                                         float(np.clip(tau_r0 * fr, 0.01, 0.99)),
                                         eps, k, gate)
                        results.append(dict(
                            gamma=gamma, pool=pool,
                            tau_s=round(float(np.clip(tau_s0 * fs, 0.01, 0.99)), 4),
                            tau_r=round(float(np.clip(tau_r0 * fr, 0.01, 0.99)), 4),
                            eps=eps, n_facets=k, ancestor_gate=gate,
                            synthetic=int(synthetic), n_rows=len(rows), **m))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k2: r.get(k2, "") for k2 in CSV_FIELDS})
    print(f"{len(results)} settings -> {out}")
    best = max(results, key=lambda r: r["macro_f1"])
    print("best macro_f1:", {k2: best[k2] for k2 in
          ("gamma", "pool", "tau_s", "tau_r", "eps", "macro_f1")})


if __name__ == "__main__":
    main()
