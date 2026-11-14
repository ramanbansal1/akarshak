"""Export coverage-alpha frontier + n_cal->coverage data — ticket 08 figures.

Reads a resplit_runner CSV (oracle or real) and emits, per selector:
  - frontier CSV: rows (selector, n_cal, alpha) -> mean_test_coverage,
    cert_rate, violation_rate  -> F2 (coverage vs alpha frontier)
  - ncal CSV: rows (selector, alpha) -> coverage vs n_cal -> F3
  - optional PNGs via matplotlib (--figs)

Usage:
  python experiments/export_frontier.py results/real_resplits.csv \
      --out results/frontier --figs
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.resplit_runner import read_csv


def args():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", required=True, help="output prefix/dir")
    p.add_argument("--figs", action="store_true")
    p.add_argument("--selectors", nargs="*", default=None,
                   help="subset of selector names (default: all certified)")
    return p.parse_args()


def pivot(rows):
    """(selector, n_cal, alpha, grid) -> record; keeps grid=first per cell."""
    d = {}
    for r in rows:
        key = (r["selector"], float(r["n_cal"]), float(r["alpha"]),
               float(r["grid"]))
        d.setdefault(key, r)
    return d


def main():
    a = args()
    rows = read_csv(a.csv)
    cells = pivot(rows)
    selectors = a.selectors or sorted({k[0] for k in cells})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # F2: coverage vs alpha, one panel per (selector, n_cal, grid)
    frontier_path = out / "frontier.csv"
    with frontier_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["selector", "n_cal", "grid", "alpha", "cert_rate",
                    "mean_test_coverage", "test_coverage_std",
                    "violation_rate", "mean_test_error"])
        for (sel, nc, al, gr), r in sorted(cells.items()):
            if sel in selectors:
                w.writerow([sel, int(nc), int(gr), al, r["cert_rate"],
                            r["mean_test_coverage"], r["test_coverage_std"],
                            r["violation_rate"], r["mean_test_error"]])

    # F3: coverage vs n_cal at fixed alpha, per selector+grid
    ncal_path = out / "ncal.csv"
    with ncal_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["selector", "grid", "alpha", "n_cal", "cert_rate",
                    "mean_test_coverage", "violation_rate"])
        for (sel, nc, al, gr), r in sorted(cells.items()):
            if sel in selectors:
                w.writerow([sel, int(gr), al, int(nc), r["cert_rate"],
                            r["mean_test_coverage"], r["violation_rate"]])
    print(f"frontier -> {frontier_path}\nncal -> {ncal_path}")

    if a.figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # F2: alpha -> coverage, panels = selector x n_cal (grid=first)
        grids = sorted({k[3] for k in cells})
        for gr in grids:
            for sel in selectors:
                ncs = sorted({k[1] for k in cells if k[0] == sel and k[3] == gr})
                alphas = sorted({k[2] for k in cells
                                 if k[0] == sel and k[3] == gr})
                fig, ax = plt.subplots(figsize=(5, 3.5))
                for nc in ncs:
                    xs, ys, cs = [], [], []
                    for al in alphas:
                        r = cells.get((sel, nc, al, gr))
                        if r and r["cert_rate"] > 0:
                            xs.append(al); ys.append(r["mean_test_coverage"])
                            cs.append(r["cert_rate"])
                    if xs:
                        ax.plot(xs, ys, "o-", ms=3, label=f"n_cal={int(nc)}")
                ax.set_xlabel("alpha"); ax.set_ylabel("mean test coverage")
                ax.set_title(f"{sel} (grid={int(gr)})")
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(out / f"F2_{sel}_g{int(gr)}.png", dpi=150)
                plt.close(fig)
        # F3: n_cal -> coverage at each alpha (primary selector only panels)
        for sel in selectors:
            for gr in grids:
                alphas = sorted({k[2] for k in cells
                                 if k[0] == sel and k[3] == gr})
                fig, ax = plt.subplots(figsize=(5, 3.5))
                for al in alphas:
                    xs, ys = [], []
                    for nc in sorted({k[1] for k in cells
                                      if k[0] == sel and k[3] == gr}):
                        r = cells.get((sel, nc, al, gr))
                        if r and r["cert_rate"] > 0:
                            xs.append(nc); ys.append(r["mean_test_coverage"])
                    if xs:
                        ax.plot(xs, ys, "o-", ms=3, label=f"alpha={al}")
                ax.set_xlabel("n_cal"); ax.set_ylabel("mean test coverage")
                ax.set_title(f"{sel} (grid={int(gr)})")
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(out / f"F3_{sel}_g{int(gr)}.png", dpi=150)
                plt.close(fig)
        print(f"figs -> {out}/F2_*.png, F3_*.png")


if __name__ == "__main__":
    main()
