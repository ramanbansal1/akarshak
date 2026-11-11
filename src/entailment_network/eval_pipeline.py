"""End-to-end WICE evaluation: verifier -> aggregator -> controller -> metrics.

Usage (repo root):
    python -m src.entailment_network.eval_pipeline \
        --ckpt runs/wice/head.pt --dataset jon-tow/wice --config subclaim \
        --split test --alpha 0.10

Thresholds (tau_s_hat, tau_r) are always calibrated on the validation split and
applied once to --split. Reports the proposal's metric set: row/chunk macro-F1,
ECE, risk-coverage, accepted-error vs alpha, abstention rate.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .aggregator import (ABSTAIN_ACTION, GAMMA, QUALIFY_ACTION,
                         SUPPORTED_ACTION, aggregate)
from .calibration import (ece, risk_coverage, select_tau_refute,
                          select_tau_support)
from .controller import decide
from .infer import Verifier
from .verifier_common import LABELS, macro_f1, parse_label

# controller action -> WICE label index (documented approximation:
# QUALIFY_CONFLICT ~ partially_supported, ABSTAIN ~ not_supported)
ACTION_TO_LABEL = {SUPPORTED_ACTION: 0, QUALIFY_ACTION: 1, ABSTAIN_ACTION: 2}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="jon-tow/wice")
    p.add_argument("--config", default="subclaim")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--calib_split", default="validation")
    p.add_argument("--alpha", type=float, default=0.10, help="accepted-error target")
    p.add_argument("--beta", type=float, default=0.10, help="false-rejection target")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="optional jsonl of per-claim decisions")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--shm_tokens", type=int, default=150_000)
    p.add_argument("--shm_root", default="/dev/shm/hs_extract")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    return p.parse_args()


def load_rows(dataset, config, split, limit):
    from .data import load_universal
    if dataset.startswith("uni:"):
        ds = load_universal(dataset[4:], split)
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset, config, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return list(ds)


def row_stats(verifier, rows):
    """Per-row: chunk probs [N,3], chunk sentence spans, WICE label."""
    res = verifier.verify([{"claim": r["claim"], "evidence": r["evidence"]} for r in rows])
    out = []
    for r, v in zip(rows, res):
        probs = np.array([c["probs"] for c in v["chunks"]], dtype=np.float64) \
            if v["chunks"] else np.zeros((0, 3))
        spans = [tuple(c["span"]) for c in v["chunks"]]
        out.append(dict(probs=probs, spans=spans, label=parse_label(r["label"]),
                        row=r))
    return out


def chunk_labels(stat):
    """WICE-style chunk labels: row label where the chunk overlaps a supporting
    sentence, else not_supported — mirrors data.py's pos rule."""
    sup = set()
    for group in stat["row"].get("supporting_sentences", []):
        sup.update(group if isinstance(group, list) else [group])
    y = []
    for a, b in stat["spans"]:
        pos = any(a <= i < b for i in sup)
        y.append(stat["label"] if pos else 2)
    return y


def _facet_mass(probs, col):
    return float(probs[:, col].mean()) if len(probs) else 0.0


def calibrate(stats, alpha, beta):
    """Two-step calibration on validation stats (k=1: claim = own facet).

    tau_r first (on refute mass R), then tau_s on eff with contradicted claims
    excluded — they can never reach SUPPORTED because coverage < 1.
    """
    eff = np.array([_facet_mass(s["probs"], 0) + GAMMA * _facet_mass(s["probs"], 1)
                    for s in stats])
    S = np.array([_facet_mass(s["probs"], 0) for s in stats])
    R = np.array([_facet_mass(s["probs"], 2) for s in stats])
    y = np.array([s["label"] for s in stats])

    refute = select_tau_refute(R, y == 2, beta)     # label 2 ~ not-supported proxy
    tau_r = refute["tau"]
    contradicted = (R >= tau_r) & (R > S)
    sup = select_tau_support(eff[~contradicted], (y == 0)[~contradicted], alpha)
    return dict(tau_s=sup["tau"], tau_r=tau_r, refute=refute, support=sup,
                n_contradicted=int(contradicted.sum()))


def evaluate(stats, tau_s, tau_r, alpha):
    acts, labels, confs, corrects = [], [], [], []
    chunk_probs_all, chunk_y_all = [], []
    decisions = []
    for s in stats:
        q = aggregate(s["probs"], np.zeros(len(s["probs"]), int), [0],
                      tau_s=tau_s, tau_r=tau_r)
        d = decide(q, tau_s)
        acts.append(ACTION_TO_LABEL[d.action])
        labels.append(s["label"])
        decisions.append(d)
        confs.append(float(q.support))
        corrects.append(s["label"] == 0)
        if len(s["probs"]):
            chunk_probs_all.append(s["probs"])
            chunk_y_all.extend(chunk_labels(s))
    acts, labels = np.array(acts), np.array(labels)
    confs, corrects = np.array(confs), np.array(corrects, dtype=bool)

    f1, per = macro_f1(torch.tensor(acts), torch.tensor(labels))
    chunk_f1 = None
    if chunk_probs_all:
        cp = np.concatenate(chunk_probs_all)
        cy = np.array(chunk_y_all)
        chunk_f1, _ = macro_f1(torch.tensor(cp.argmax(-1)), torch.tensor(cy))

    accepted = acts == 0
    accepted_err = float((labels[accepted] != 0).mean()) if accepted.any() else None
    return dict(
        n=len(stats),
        action_macro_f1=round(f1, 4),
        per_class={k: round(v, 4) for k, v in per.items()},
        chunk_macro_f1=round(chunk_f1, 4) if chunk_f1 is not None else None,
        abstain_rate=round(float((acts == 2).mean()), 4),
        qualify_rate=round(float((acts == 1).mean()), 4),
        accepted=int(accepted.sum()),
        accepted_error_rate=round(accepted_err, 4) if accepted_err is not None else None,
        alpha=alpha,
        alpha_satisfied=bool(accepted_err is not None and accepted_err <= alpha),
        ece=round(ece(confs, corrects), 4),
        risk_coverage=[{k: round(v, 4) for k, v in pt.items()}
                       for pt in risk_coverage(confs, corrects)["points"][::4]],
        calibration=dict(tau_s=tau_s, tau_r=tau_r),
    ), decisions


def main():
    a = args()
    verifier = Verifier(a.ckpt, dtype=a.dtype, tp=a.tp, gpu_mem=a.gpu_mem,
                        shm_tokens=a.shm_tokens, shm_root=a.shm_root,
                        backend=a.backend)

    calib_rows = load_rows(a.dataset, a.config, a.calib_split, a.limit)
    calib_stats = row_stats(verifier, calib_rows)
    cal = calibrate(calib_stats, a.alpha, a.beta)
    print("CALIBRATION", json.dumps({k: v for k, v in cal.items()
                                     if k in ("tau_s", "tau_r", "n_contradicted")},
                                    indent=2, default=str))
    print("  tau_s detail:", cal["support"])
    print("  tau_r detail:", cal["refute"])

    rows = load_rows(a.dataset, a.config, a.split, a.limit)
    stats = row_stats(verifier, rows)
    metrics, decisions = evaluate(stats, cal["tau_s"], cal["tau_r"], a.alpha)
    print("METRICS", json.dumps(metrics, indent=2, default=str))

    if a.out:
        with open(a.out, "w") as f:
            for r, d in zip(rows, decisions):
                f.write(json.dumps(dict(claim=r["claim"], label=r["label"],
                                        action=d.action, report=d.report)) + "\n")


if __name__ == "__main__":
    main()
