"""Training-curve plots from logged history CSVs -> PNGs.

Two input schemas are supported (mix freely via --glob):

  head_history.csv           (src.entailment_network.train)
      epoch, loss, chunk_f1, row_f1, row_acc, lr        — one run per file

  head_ablations_history.csv (experiments/head_ablations.py)
      r, epoch, val_row_f1, train_loss                   — one file, line per rank

Emits one figure per metric plus a 2x2 overview, saved under --out
(default results/plots/). Colorblind-safe palette, no in-figure titles —
captions carry them per the venue guidelines.

Usage:
  python experiments/plot_training.py \
      --glob 'runs/*/head_history.csv' 'results/ablation/head_ablations_history.csv' \
      --out results/plots
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

RUN_METRICS = [("loss", "train_loss"), ("row_f1", "val row macro-F1"),
               ("chunk_f1", "val chunk macro-F1"), ("row_acc", "val row accuracy")]
ABL_METRICS = [("train_loss", "train_loss"), ("val_row_f1", "val row macro-F1")]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", nargs="+", default=["runs/*/head_history.csv"],
                   help="glob(s) for history csv files")
    p.add_argument("--out", default="results/plots")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load(path: Path):
    return list(csv.DictReader(open(path)))


def fnum(rows, col):
    return [float(r[col]) for r in rows]


def plot_runs(files, out, dpi):
    """head_history.csv schema: epoch/loss/chunk_f1/row_f1/row_acc/lr."""
    runs = []
    for f in files:
        rows = load(f)
        if not rows or "loss" not in rows[0]:
            continue
        d = {k: fnum(rows, k) for k in rows[0]}
        runs.append((f.parent.name, d))
    if not runs:
        return 0

    for col, label in RUN_METRICS:
        if not any(col in d for _, d in runs):
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for i, (name, d) in enumerate(runs):
            if col not in d:
                continue
            ax.plot(d["epoch"], d[col], marker="o", ms=3, lw=1.4,
                    color=COLORS[i % len(COLORS)], label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if len(runs) > 1:
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"train_{col}.png", dpi=dpi)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
    for ax, (col, label) in zip(axes.ravel(), RUN_METRICS):
        for i, (name, d) in enumerate(runs):
            if col not in d:
                continue
            ax.plot(d["epoch"], d[col], marker="o", ms=3, lw=1.4,
                    color=COLORS[i % len(COLORS)], label=name)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel("epoch", fontsize=8)
    if len(runs) > 1:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, frameon=False,
                   loc="upper center", ncol=min(len(runs), 4),
                   bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out / "train_overview.png", dpi=dpi)
    plt.close(fig)
    return len(runs)


def plot_ablations(files, out, dpi):
    """head_ablations_history.csv schema: r/epoch/val_row_f1/train_loss."""
    made = 0
    for f in files:
        rows = load(f)
        if not rows or "val_row_f1" not in rows[0] or "r" not in rows[0]:
            continue
        # group by (kind, r) when a kind column exists (lowrank vs blockdiag)
        keys = sorted({(r.get("kind", ""), r["r"]) for r in rows},
                      key=lambda t: (t[0], float(t[1])))
        by_r = {}
        for kind, r in keys:
            by_r[(kind, r)] = [x for x in rows
                               if x.get("kind", "") == kind and x["r"] == r]
        for col, label in ABL_METRICS:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            for i, key in enumerate(keys):
                kind, r = key
                d = by_r[key]
                tag = f"{kind}={r}" if kind else f"r={r}"
                ax.plot(fnum(d, "epoch"), fnum(d, col), marker="o", ms=3,
                        lw=1.4, color=COLORS[i % len(COLORS)], label=tag)
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, frameon=False, title="rank")
            fig.tight_layout()
            stem = f"{f.stem.replace('head_ablations_history', 'ablations')}_{col}"
            fig.savefig(out / f"{stem}.png", dpi=dpi)
            plt.close(fig)
        made += 1
    return made


def main():
    a = args()
    files = sorted({Path(f) for g in a.glob for f in glob.glob(g)})
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(f"no history csv files match {a.glob!r}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    n_runs = plot_runs(files, out, a.dpi)
    n_abl = plot_ablations(files, out, a.dpi)
    n_png = len(list(out.glob("*.png")))
    if n_png == 0:
        raise SystemExit(f"no plottable rows found in {files}")
    print(f"wrote {n_png} figures to {out} "
          f"({n_runs} run history file(s), {n_abl} ablation history file(s))")


if __name__ == "__main__":
    main()
