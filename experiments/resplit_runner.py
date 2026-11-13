"""Resplit protocol runner — ticket 04, the core WICE experiment.

Pools WICE dev+test parent instances (707 parents / 1,907 claims) and runs
N random parent-level calibration/test resplits. Per resplit, sweeps
n_cal x alpha x |T| and applies every selector; optionally compares a
learned scorer against the oracle floor u_j = y_j.

Modes:
  --mode oracle      u_j = y_j (perfect scorer; feasibility floor, no GPU)
  --mode noisy       u_j = clip(y_j + sigma*eps) — figure variant
  --mode real        u_j from eval_pipeline --scores_out npz files
  --mode both        oracle + real in one pass (real needs npz inputs)

Per cell (n_cal, alpha, grid, bound, selector) the runner aggregates over
resplits: certification rate, mean cal/test coverage, violation rate
(frac of resplits where test accepted-error > alpha), mean test error.

Gate (--compare oracle.csv real.csv): three-region classification per
cell — oracle fails => n-limited; oracle certifies but real doesn't =>
verifier-limited; both certify => report coverage_real / coverage_oracle
(where oracle coverage > 0).

Usage (repo root):
  python experiments/resplit_runner.py --mode oracle --n-resplits 1000 \
      --out results/oracle_resplit.csv
  python experiments/resplit_runner.py --mode real \
      --scores results/scores_dev.npz results/scores_test.npz \
      --out results/real_resplit.csv
  python experiments/resplit_runner.py --compare results/oracle_resplit.csv \
      results/real_resplit.csv --out results/gate_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.entailment_network.calibration import (select_tau_hoeffding,
                                              select_tau_rcps,
                                              select_tau_support)
from src.entailment_network.claimscore import read_npz
from src.entailment_network.splits import M_MAX

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
N_CALS = [100, 200, 300, 500]
GRIDS = [50, 100]
DELTA = 0.10
SEED = 0
WICE_DIR = Path(__file__).resolve().parents[1] / "data" / "wice"


# ----------------------------------------------------------------- arrays
def oracle_arrays(wice_dir: Path = WICE_DIR) -> dict:
    """Pooled dev+test parents -> {scores=u=y, correct=y, instances, ids}."""
    parents = defaultdict(list)
    for split in ("dev", "test"):
        for line in (wice_dir / f"subclaim_{split}.jsonl").open():
            r = json.loads(line)
            pid = r["meta"]["id"].rsplit("-", 1)[0]
            y = 1 if r["label"] == "supported" else 0
            parents[pid].append(y)
    return _parents_to_arrays(parents)


def noisy_oracle_arrays(sigma: float, seed: int = 0,
                        wice_dir: Path = WICE_DIR) -> dict:
    """u_j = clip(y_j + sigma*eps, 0, 1); correct stays strict y."""
    arr = oracle_arrays(wice_dir)
    rng = np.random.default_rng(seed)
    arr["scores"] = np.clip(arr["scores"] + sigma * rng.standard_normal(
        len(arr["scores"])), 0.0, 1.0)
    return arr


def real_arrays(npz_paths: list[str]) -> dict:
    """Merge eval_pipeline ClaimScore npz files into one array dict.

    instance_ids must be unique across files (true for dev/test parents).
    """
    parts = [read_npz(p) for p in npz_paths]
    ids = sorted({i for p in parts for i in p["instance_ids"]})
    pos = {p: i for i, p in enumerate(ids)}
    scores, correct, inst = [], [], []
    for p in parts:
        remap = np.array([pos[i] for i in p["instance_ids"]])
        scores.append(p["scores"])
        correct.append(p["correct"])
        inst.append(remap[p["instances"]])
    return dict(scores=np.concatenate(scores),
                correct=np.concatenate(correct).astype(bool),
                instances=np.concatenate(inst).astype(int),
                instance_ids=ids)


def _parents_to_arrays(parents: dict[str, list[int]]) -> dict:
    ids = sorted(parents)
    pos = {p: i for i, p in enumerate(ids)}
    scores, correct, inst = [], [], []
    for pid in ids:
        for y in parents[pid]:
            scores.append(float(y))
            correct.append(bool(y))
            inst.append(pos[pid])
    scores = np.array(scores)
    return dict(scores=scores, correct=np.array(correct, dtype=bool),
                instances=np.array(inst, dtype=int), instance_ids=ids)


# ----------------------------------------------------------------- engine
SELECTORS = ["empirical", "rcps",
             "bernstein_micro", "bernstein_macro",
             "hoeffding_micro", "hoeffding_macro"]


def _run_selector(name, scores, correct, inst, alpha, delta, grid, m_max):
    if name == "empirical":
        return select_tau_support(scores, correct, alpha)
    if name == "rcps":
        return select_tau_rcps(scores, correct, alpha=alpha, delta=delta,
                               grid=grid)
    bound, mode = name.split("_", 1)
    return select_tau_hoeffding(scores, correct, inst, alpha=alpha,
                                delta=delta, grid=grid, m_max=m_max,
                                bound=bound, mode=mode)


def _apply_tau(scores, correct, tau):
    acc = scores >= tau
    n = int(acc.sum())
    return dict(accepted=n,
                coverage=float(acc.mean()) if len(scores) else 0.0,
                error=(float((~correct[acc]).mean()) if n else None))


def run_resplits(arr: dict, n_resplits: int, seed: int = SEED,
                 n_cals=N_CALS, alphas=ALPHAS, grids=GRIDS, delta=DELTA,
                 m_max: int | None = M_MAX, mode_name: str = "run",
                 selectors=SELECTORS):
    """Run the resplit protocol; returns per-cell aggregate records.

    Each record: one (n_cal, alpha, grid, bound, selector) cell with
    certification rate, coverage and violation stats over resplits.
    """
    scores, correct = arr["scores"], arr["correct"]
    inst = arr["instances"]
    ids = arr["instance_ids"]
    n_inst = len(ids)
    rows_of = [np.where(inst == i)[0] for i in range(n_inst)]
    rng = np.random.default_rng(seed)

    # cell -> lists of per-resplit outcomes
    cells = defaultdict(lambda: dict(cert=[], cov_cal=[], cov_test=[],
                                     viol=[], err_test=[]))
    for _ in range(n_resplits):
        perm = rng.permutation(n_inst)
        for n_cal in n_cals:
            nc = min(int(n_cal), n_inst - 1)
            cal_rows = np.concatenate([rows_of[i] for i in perm[:nc]])
            test_rows = np.concatenate([rows_of[i] for i in perm[nc:]])
            cs, cc, ci = scores[cal_rows], correct[cal_rows], inst[cal_rows]
            ts, tc = scores[test_rows], correct[test_rows]
            for alpha in alphas:
                for grid in grids:
                    for sel in selectors:
                        if sel == "empirical" and grid != grids[0]:
                            continue      # empirical tau ignores |T|
                        bound = sel.split("_")[0] if "_" in sel else "-"
                        sel_mode = sel.split("_")[1] if "_" in sel else "-"
                        key = (nc, alpha, grid, bound, sel_mode, sel)
                        res = _run_selector(sel, cs, cc, ci, alpha, delta,
                                            grid, m_max)
                        tau = res["tau"]
                        cell = cells[key]
                        cell["cert"].append(tau != float("inf"))
                        cell["cov_cal"].append(res.get("coverage", 0.0))
                        if tau == float("inf"):
                            cell["cov_test"].append(0.0)
                            cell["viol"].append(0.0)
                            cell["err_test"].append(np.nan)
                        else:
                            t = _apply_tau(ts, tc, tau)
                            cell["cov_test"].append(t["coverage"])
                            cell["err_test"].append(
                                t["error"] if t["error"] is not None
                                else np.nan)
                            cell["viol"].append(
                                int(t["error"] is not None and t["error"] > alpha))

    records = []
    for (nc, alpha, grid, bound, sel_mode, sel), c in sorted(
            cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][5])):
        err = np.array(c["err_test"], dtype=float)
        records.append(dict(
            mode=mode_name, n_cal=nc, alpha=alpha, grid=grid,
            bound=bound, sel_mode=sel_mode, selector=sel,
            cert_rate=float(np.mean(c["cert"])),
            mean_cal_coverage=float(np.mean(c["cov_cal"])),
            mean_test_coverage=float(np.mean(c["cov_test"])),
            test_coverage_std=float(np.std(c["cov_test"])),
            violation_rate=float(np.mean(c["viol"])),
            mean_test_error=float(np.nanmean(err)) if np.isfinite(err).any()
            else np.nan,
            n_resplits=n_resplits))
    return records


CSV_FIELDS = ["mode", "n_cal", "alpha", "grid", "bound", "sel_mode",
              "selector", "cert_rate", "mean_cal_coverage",
              "mean_test_coverage", "test_coverage_std", "violation_rate",
              "mean_test_error", "n_resplits"]


def write_csv(records: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(records)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [{k: (float(v) if k not in ("mode", "bound", "sel_mode",
                                          "selector") and v != ""
                     else v) for k, v in r.items()}
                for r in csv.DictReader(f)]


# ----------------------------------------------------------------- gate
def gate_summary(oracle_csv: str | Path, real_csv: str | Path,
                 out_path: str | Path | None = None) -> str:
    """Three-region gate: n-limited / verifier-limited / coverage ratio."""
    oracle = {(r["n_cal"], r["alpha"], r["grid"], r["bound"], r["sel_mode"],
               r["selector"]): r for r in read_csv(oracle_csv)}
    real = {(r["n_cal"], r["alpha"], r["grid"], r["bound"], r["sel_mode"],
             r["selector"]): r for r in read_csv(real_csv)}
    lines = ["cell(n_cal,alpha,grid,bound,mode,selector) | region | "
             "oracle_cert real_cert | cov_ratio"]
    for key in sorted(set(oracle) & set(real)):
        o, r = oracle[key], real[key]
        oc, rc = o["cert_rate"], r["cert_rate"]
        if oc < 0.5:
            region = "n-limited"
        elif rc < 0.5:
            region = "verifier-limited"
        else:
            region = "both-certify"
        ratio = (r["mean_test_coverage"] / o["mean_test_coverage"]
                 if o["mean_test_coverage"] > 0 else float("nan"))
        lines.append(f"{key} | {region} | {oc:.2f} {rc:.2f} | "
                     f"{ratio:.3f}" if region == "both-certify"
                     else f"{key} | {region} | {oc:.2f} {rc:.2f} | -")
    text = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(text + "\n")
    return text


# ----------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "noisy", "real", "both"],
                    default="oracle")
    ap.add_argument("--sigma", type=float, default=0.2,
                    help="noise level for --mode noisy")
    ap.add_argument("--scores", nargs="*", default=[],
                    help="eval_pipeline ClaimScore npz files (real mode)")
    ap.add_argument("--n-resplits", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-cals", type=int, nargs="*", default=N_CALS)
    ap.add_argument("--alphas", type=float, nargs="*", default=ALPHAS)
    ap.add_argument("--grids", type=int, nargs="*", default=GRIDS)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--selectors", nargs="*", default=SELECTORS)
    ap.add_argument("--tag", default="", help="suffix for output mode name")
    ap.add_argument("--out", default=None, help="CSV output path")
    ap.add_argument("--compare", nargs=2, metavar=("ORACLE_CSV", "REAL_CSV"),
                    help="write three-region gate summary instead of running")
    a = ap.parse_args()

    if a.compare:
        print(gate_summary(a.compare[0], a.compare[1]))
        return

    arrays = {}
    if a.mode in ("oracle", "both"):
        arrays["oracle"] = oracle_arrays()
    if a.mode == "noisy":
        arrays[f"noisy_s{a.sigma}"] = noisy_oracle_arrays(a.sigma, a.seed)
    if a.mode in ("real", "both"):
        if not a.scores:
            raise SystemExit("--mode real/both requires --scores npz files")
        arrays["real"] = real_arrays(a.scores)

    all_records = []
    for name, arr in arrays.items():
        n_inst = len(arr["instance_ids"])
        print(f"[{name}] instances={n_inst} claims={len(arr['scores'])} "
              f"base_err={1 - arr['correct'].mean():.3f}")
        recs = run_resplits(arr, a.n_resplits, seed=a.seed,
                            n_cals=a.n_cals, alphas=a.alphas,
                            grids=a.grids, delta=a.delta, m_max=M_MAX,
                            mode_name=name + a.tag,
                            selectors=a.selectors)
        all_records.extend(recs)
        out = a.out or f"results/resplit_{name}{a.tag}.csv"
        write_csv(recs, out)
        print(f"[{name}] wrote {len(recs)} cells -> {out}")


if __name__ == "__main__":
    main()
