"""Phase 2: train the low-rank bilinear verdict head on cached hidden states.

CPU/GPU-light — runs on CPU or a single GPU. Requires feature caches produced
by `python -m src.pre_computing.extract` in --out_dir (feats_{split}.pt).

  python -m src.entailment_network.train --out_dir runs/wice
"""

import argparse, csv, math, os
import torch, torch.nn.functional as F
from .runlog import finish as wb_finish, get_wandb, log as wb_log
from .verifier_common import *

try:
    from tqdm import tqdm
except Exception:                                    # noqa: BLE001
    tqdm = None


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="runs/wice",
                   help="dir containing feats_{train,validation,test}.pt; head.pt written here")
    p.add_argument("--r", type=int, default=256); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=512); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--class_power", type=float, default=0.5, help="class weight = freq^-power (0=off, 1=full)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def evaluate(model, cache, dev, thr=None):
    probs = predict_cache(model, cache, dev)
    chunk_f1, _ = macro_f1(probs.argmax(-1), cache.y)
    s = row_scores(probs, cache.chunk_row, cache.n_rows)
    y = cache.extra["row_y"]
    if thr is None:
        _, ts, tp = tune_thresholds(s, y)
    else:
        ts, tp = thr
    pred = decide(s, ts, tp)
    f1, per = macro_f1(pred, y)
    return dict(chunk_f1=chunk_f1, row_f1=f1, row_acc=(pred == y).float().mean().item(),
                per_class=per, thr=(ts, tp))


def main():
    a = args(); torch.manual_seed(a.seed); os.makedirs(a.out_dir, exist_ok=True)
    cp = {s: f"{a.out_dir}/feats_{s}.pt" for s in ("train", "validation", "test")}
    for s in ("train", "validation"):
        if not os.path.exists(cp[s]):
            raise SystemExit(f"missing {cp[s]} — run "
                             f"`python -m src.pre_computing.extract` first")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = FeatureCache(cp["train"]), FeatureCache(cp["validation"])
    L, d = tr.hq.shape[1:]
    model = VerdictModel(d, L, K=3, r=a.r).to(dev)
    cnt = torch.bincount(tr.y, minlength=3).float()
    print("chunk label counts", dict(zip(LABELS, cnt.tolist())))
    w = (cnt.sum() / cnt.clamp(min=1)) ** a.class_power; w = (w / w.mean()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    M = len(tr.y); total = a.epochs * math.ceil(M / a.bs); warm = max(total // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))
    wb = get_wandb(name=f"head-{os.path.basename(a.out_dir.rstrip('/'))}",
                   config=vars(a))
    wb_log(dict(model_cfg=dict(d=d, n_layers=L, K=3, r=a.r),
                chunks=int(len(tr.y)), rows=int(tr.n_rows)))

    hist_path = f"{a.out_dir}/head_history.csv"
    fresh = not os.path.exists(hist_path)
    hist = open(hist_path, "a", newline="")
    wr = csv.writer(hist)
    if fresh:
        wr.writerow(["epoch", "loss", "chunk_f1", "row_f1", "row_acc", "lr"])

    best = -1
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(M)
        steps = range(0, M, a.bs)
        if tqdm is not None:
            steps = tqdm(steps, desc=f"ep{ep:02d}", unit="batch",
                         total=math.ceil(M / a.bs), leave=False)
        for i in steps:
            hq, he, y = tr.batch(perm[i:i + a.bs])
            loss = F.cross_entropy(model(hq.to(dev), he.to(dev))[:, 0], y.to(dev), weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        m = evaluate(model, va, dev)
        row = [ep, f"{loss.item():.6f}", f"{m['chunk_f1']:.6f}",
               f"{m['row_f1']:.6f}", f"{m['row_acc']:.6f}",
               f"{sched.get_last_lr()[0]:.3e}"]
        wr.writerow(row); hist.flush()
        print(f"ep {ep:02d} loss {loss.item():.4f} | val chunk_f1 {m['chunk_f1']:.4f} "
              f"row_f1 {m['row_f1']:.4f} row_acc {m['row_acc']:.4f} thr {m['thr']}")
        wb_log(dict(epoch=ep, train_loss=loss.item(), chunk_f1=m["chunk_f1"],
                    row_f1=m["row_f1"], row_acc=m["row_acc"], lr=sched.get_last_lr()[0]))
        if m["row_f1"] > best:
            best = m["row_f1"]
            torch.save(dict(state_dict=model.state_dict(), model_cfg=dict(d=d, n_layers=L, K=3, r=a.r),
                            labels=LABELS, thresholds=m["thr"], **tr.meta), f"{a.out_dir}/head.pt")
    hist.close()

    ck = torch.load(f"{a.out_dir}/head.pt"); model.load_state_dict(ck["state_dict"])
    if os.path.exists(cp["test"]):
        m = evaluate(model, FeatureCache(cp["test"]), dev, thr=ck["thresholds"])
        print("TEST", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        wb_log({f"test/{k}": v for k, v in m.items()
                if isinstance(v, (int, float))})
    wb_finish()


if __name__ == "__main__":
    main()
