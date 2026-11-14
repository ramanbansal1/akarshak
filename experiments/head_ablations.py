"""Head-capacity ablations — ticket 07 (CPU once caches exist).

Modes:
  rank     VerdictModel rank sweep r in --ranks on feats_{train,validation,
           test}.pt (P1-train fit, P1-val early stop, P2 test metrics).
           Answers: does the ~3M-param head earn its parameters?
  centroid nearest-centroid baseline in he last-layer space -> same
           chunk->row->decide path -> row macro-F1, directly comparable.
  lgbm     LightGBM on the 5 ClaimQuantities -> P(strict_y) as alternative
           u_j. Trained on the P2 probs pool with GroupKFold over parent
           ids (parent-level CV, never dev/test); emits ClaimScore npz per
           extra pool in the resplit_runner --mode real format.
  all      rank + centroid + lgbm

Layer sweep is INFEASIBLE with current caches: extract caches only the last
hidden layer (verifier_common.default_layers). Would need --layers change +
GPU re-extraction — flagged, not run.

Usage (repo root, after ticket 05 artifacts land):
  python experiments/head_ablations.py --mode all --dir runs/wice_p1p2 \
      --probs_calib results/probs.wice_p1p2_test.npz \
      --probs_pairs uni:wice:validation=results/probs.wice_validation.npz \
                   uni:wice:test=results/probs.wice_test.npz
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.entailment_network.aggregator import aggregate
from src.entailment_network.claimscore import ClaimScore, write_npz
from src.entailment_network.eval_pipeline import calibrate
from src.entailment_network.verifier_common import (FeatureCache,
                                                    VerdictModel,
                                                    macro_f1, predict_cache,
                                                    row_scores, tune_thresholds,
                                                    decide as vdecide)

QUANTITIES = ["support", "contradiction", "coverage", "conflict", "agreement"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="all",
                   choices=["rank", "centroid", "lgbm", "all"])
    p.add_argument("--dir", default="runs/wice_p1p2",
                   help="dir with feats_{train,validation,test}.pt")
    p.add_argument("--ranks", default="32,64,128,256,512")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--class_power", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probs_calib", default=None,
                   help="P2 probs npz for lgbm training/calibration")
    p.add_argument("--probs_pairs", nargs="*", default=[],
                   help="name=probs.npz pools to score with lgbm-u")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default=None)
    return p.parse_args()


# ---------------------------------------------------------------- rank sweep
def _row_eval(model, cache, dev, thr=None):
    probs = predict_cache(model, cache, dev)
    chunk_f1, _ = macro_f1(probs.argmax(-1), cache.y)
    s = row_scores(probs, cache.chunk_row, cache.n_rows)
    y = cache.extra["row_y"]
    ts, tp = thr if thr is not None else tune_thresholds(s, y)[1:]
    pred = vdecide(s, ts, tp)
    f1, per = macro_f1(pred, y)
    return dict(chunk_f1=float(chunk_f1), row_f1=float(f1),
                row_acc=float((pred == y).float().mean()),
                thr=(float(ts), float(tp)))


def train_head(tr, va, r, a, dev):
    """Same recipe as train.py main(): AdamW + cosine, early stop on val
    row_f1, returns (best_state, best_val_row_f1, history)."""
    torch.manual_seed(a.seed)
    L, d = tr.hq.shape[1], tr.hq.shape[2]
    model = VerdictModel(d, L, K=3, r=r).to(dev)
    cnt = torch.bincount(tr.y, minlength=3).float()
    w = (cnt.sum() / cnt.clamp(min=1)) ** a.class_power
    w = (w / w.mean()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    M = len(tr.y)
    total = a.epochs * math.ceil(M / a.bs)
    warm = max(total // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm)
        * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))
    best_f1, best_state, hist = -1.0, None, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(M)
        for i in range(0, M, a.bs):
            hq, he, y = tr.batch(perm[i:i + a.bs])
            loss = F.cross_entropy(model(hq.to(dev), he.to(dev))[:, 0],
                                   y.to(dev), weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        m = _row_eval(model, va, dev)
        hist.append((ep, m["row_f1"]))
        if m["row_f1"] > best_f1:
            best_f1 = m["row_f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(dev)
    return model, best_f1, hist


def n_params(r, d):
    return 2 * 3 * d * r + 3          # U,V [K,d,r] + bias[K]


def mode_rank(a, dev):
    tr, va = FeatureCache(f"{a.dir}/feats_train.pt"), \
        FeatureCache(f"{a.dir}/feats_validation.pt")
    te = FeatureCache(f"{a.dir}/feats_test.pt") if \
        Path(f"{a.dir}/feats_test.pt").exists() else None
    rows = []
    for r in [int(x) for x in a.ranks.split(",")]:
        model, vf1, _ = train_head(tr, va, r, a, dev)
        vm = _row_eval(model, va, dev)
        rec = dict(r=r, d=tr.hq.shape[2], n_params=n_params(r, tr.hq.shape[2]),
                   val_row_f1=round(vf1, 4))
        if te is not None:
            tm = _row_eval(model, te, dev)
            rec.update(test_chunk_f1=round(tm["chunk_f1"], 4),
                       test_row_f1=round(tm["row_f1"], 4),
                       test_row_acc=round(tm["row_acc"], 4))
        rows.append(rec)
        print("rank", rec, flush=True)
    _write_csv(rows, Path(a.out) / "head_ablations_rank.csv")
    return rows


# ------------------------------------------------------------- centroid base
def mode_centroid(a, dev):
    """Nearest-centroid in he last-layer space -> softmax(-dist^2) probs."""
    va = FeatureCache(f"{a.dir}/feats_validation.pt")
    te_path = f"{a.dir}/feats_test.pt"
    te = FeatureCache(te_path) if Path(te_path).exists() else None
    out = []
    for name, cache in [("validation", va)] + ([("test", te)] if te else []):
        X = cache.he[:, -1, :].numpy().astype(np.float64)   # [C,d]
        y = cache.y.numpy()
        cents = np.stack([X[y == c].mean(0) for c in range(3)])
        d2 = ((X[:, None, :] - cents[None]) ** 2).sum(-1)   # [C,3]
        probs = torch.tensor(np.exp(-d2 - d2.min(1, keepdims=True))
                             / np.exp(-d2 - d2.min(1, keepdims=True)).sum(1, keepdims=True),
                             dtype=torch.float32)
        # monkey-patch predict path: reuse row_scores + decide directly
        chunk_f1, _ = macro_f1(probs.argmax(-1), cache.y)
        s = row_scores(probs, cache.chunk_row, cache.n_rows)
        ry = cache.extra["row_y"]
        _, ts, tp = tune_thresholds(s, ry)
        pred = vdecide(s, ts, tp)
        f1, _ = macro_f1(pred, ry)
        out.append(dict(split=name, chunk_f1=round(float(chunk_f1), 4),
                        row_f1=round(float(f1), 4),
                        row_acc=round(float((pred == ry).float().mean()), 4)))
        print("centroid", out[-1], flush=True)
    _write_csv(out, Path(a.out) / "head_ablations_centroid.csv")
    return out


# ------------------------------------------------------------------ lgbm-u
def _xy(rows, tau_s, tau_r):
    """Recompute quantities at given taus -> (X [n,5], y)."""
    X, y = [], []
    for r in rows:
        p = r["probs"]
        q = aggregate(p, np.zeros(len(p), int), [0] if len(p) else [],
                      tau_s=tau_s, tau_r=tau_r)
        X.append([getattr(q, k) for k in QUANTITIES])
        y.append(r["strict_y"])
    return np.array(X, dtype=np.float64), np.array(y)


def _load_probs_rows(path):
    """probs npz -> per-row {probs, label(3-class), strict_y, parent_id, row_id}."""
    z = np.load(path, allow_pickle=False)
    probs, chunk_row = z["probs"], z["chunk_row"]
    n = len(z["label"])
    return [dict(probs=np.asarray(probs[chunk_row == r], dtype=np.float64),
                 label=int(z["label"][r]),
                 strict_y=int(z["strict_y"][r]),
                 parent_id=str(z["parent_id"][r]),
                 row_id=str(z["row_id"][r])) for r in range(n)]


def mode_lgbm(a):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    cal_rows = _load_probs_rows(a.probs_calib)
    # tau_s/tau_r from the SAME calibrate() the pipeline uses, on P2
    stats = [dict(probs=r["probs"], spans=[], label=r["label"], row={})
             for r in cal_rows]
    cal = calibrate(stats, 0.10, 0.10, 0.10)
    tau_s, tau_r = cal["tau_s"], cal["tau_r"]
    print(f"lgbm calib on P2: tau_s={tau_s:.4f} tau_r={tau_r:.4f} "
          f"n_contradicted={cal['n_contradicted']}")

    X, y = _xy(cal_rows, tau_s, tau_r)
    _, groups = np.unique([r["parent_id"] for r in cal_rows],
                          return_inverse=True)          # stable parent groups
    oof = np.zeros(len(y))
    for tr_i, va_i in GroupKFold(n_splits=5).split(X, y, groups):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               num_leaves=15, verbose=-1)
        m.fit(X[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(X[va_i])[:, 1]
    auc = roc_auc_score(y, oof) if len(set(y)) > 1 else float("nan")
    print(f"lgbm P2 group-CV AUC(u->strict_y) = {auc:.4f}")

    final = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                               num_leaves=15, verbose=-1)
    final.fit(X, y)
    rows = [dict(mode="lgbm_p2_groupcv", pool="P2", n=len(y),
                 mean_u=round(float(oof.mean()), 4),
                 auc=round(float(auc), 4),
                 tau_s=round(float(tau_s), 4),
                 tau_r=round(float(tau_r), 4))]
    for spec in a.probs_pairs:
        name, path = spec.split("=", 1)
        prows = _load_probs_rows(path)
        Xp, _ = _xy(prows, tau_s, tau_r)
        u = final.predict_proba(Xp)[:, 1]
        recs = [ClaimScore(u=float(ui), y=int(ri["strict_y"]),
                           instance_id=ri["parent_id"],
                           claim_id=ri["row_id"]) for ui, ri in zip(u, prows)]
        suffix = name.replace(":", "_")
        write_npz(recs, f"{a.out}/scores_lgbm.{suffix}.npz")
        rows.append(dict(mode="lgbm_scored", pool=name, n=len(recs),
                         mean_u=round(float(u.mean()), 4),
                         auc="", tau_s="", tau_r=""))
        print("lgbm scored", rows[-1])
    _write_csv(rows, Path(a.out) / "head_ablations_lgbm.csv")
    return rows


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


def main():
    a = args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.mode in ("rank", "all"):
        mode_rank(a, dev)
    if a.mode in ("centroid", "all"):
        mode_centroid(a, dev)
    if a.mode in ("lgbm", "all"):
        if not a.probs_calib:
            raise SystemExit("--probs_calib required for lgbm mode")
        mode_lgbm(a)


if __name__ == "__main__":
    main()
