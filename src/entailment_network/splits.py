"""Deterministic WICE parent-level pool split (experiment protocol, ticket 01).

Pools (disjoint, parent = WICE claim id with the trailing `-<subclaim>`
stripped, e.g. `train41647-0` -> `train41647`):

  P1  ~1,060 train parents -> verifier-head training; a ~10% internal fold
      (P1-val) is carved out for early stopping / threshold tuning.
  P2  ~200 train parents  -> aggregator / LightGBM tuning ONLY.
  dev+test (707 parents)  -> untouched; pooled later for cal/test resplits.

Manifests are written to data/wice/{p1,p1_val,p2,p1_train}_parents.txt and a
filtered universal dataset to data/universal/wice_p1p2/{train,dev,test}.jsonl
so the existing `uni:` loaders work unchanged:

  train.jsonl = P1 minus P1-val rows   (head training)
  dev.jsonl    = P1-val rows       (early stopping / tune_thresholds)
  test.jsonl   = P2 rows           (aggregator / LightGBM tuning)

CLI: `python -m src.entailment_network.splits` (writes + verifies).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNIVERSAL_WICE = ROOT / "data" / "universal" / "wice"
MANIFEST_DIR = ROOT / "data" / "wice"
POOLED_DIR = ROOT / "data" / "universal" / "wice_p1p2"

# Structural cap: max subclaims per parent claim, verified across WICE
# train/dev/test (ticket 01). Frozen constant — never estimated per resplit.
M_MAX = 6

SEED = 20240613
P2_TARGET = 200          # ~200 of 1,260 train parents
VAL_FRAC = 0.10          # ~10% of P1 parents -> early-stopping fold

_SUB_SUFFIX = re.compile(r"-\d+$")


def parent_of(row: dict) -> str:
    """WICE parent id for a universal row: claim_meta.id minus `-<n>` suffix."""
    rid = (row.get("claim_meta") or {}).get("id") or row.get("id", "")
    rid = rid.split(":", 1)[-1] if ":" in rid else rid
    return _SUB_SUFFIX.sub("", rid)


def wice_label(row: dict) -> str:
    return ((row.get("meta") or {}).get("raw") or {}).get("label", "")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def parent_label_mix(rows: list[dict]) -> dict[str, list[str]]:
    """parent_id -> list of WICE labels for its subclaim rows."""
    mix: dict[str, list[str]] = {}
    for r in rows:
        mix.setdefault(parent_of(r), []).append(wice_label(r))
    return mix


def _stratum(mix: list[str]) -> float:
    """Label-mix stratum: fraction of `supported` subclaims, bucketed to 0.2."""
    frac = mix.count("supported") / len(mix)
    return math.floor(frac * 5 + 0.5) / 5   # half-up buckets of 0.2


def _allocate(strata: dict[float, list[str]], total: int) -> dict[float, int]:
    """Largest-remainder proportional allocation of `total` over strata."""
    n = sum(len(v) for v in strata.values())
    quota, rem = {}, []
    for k, members in strata.items():
        exact = len(members) * total / n
        quota[k] = int(exact)
        rem.append((exact - int(exact), k))
    for _, k in sorted(rem, reverse=True)[: total - sum(quota.values())]:
        quota[k] += 1
    return quota


def build_split(rows: list[dict], seed: int = SEED,
                p2_target: int = P2_TARGET, val_frac: float = VAL_FRAC):
    """Return (p1_train, p1_val, p2) parent-id sets. Deterministic in `seed`.

    Stratified by each parent's supported-fraction bucket; within a stratum
    members are shuffled with `seed` before quotas are taken.
    """
    mix = parent_label_mix(rows)
    strata: dict[float, list[str]] = {}
    for pid, labels in mix.items():
        strata.setdefault(_stratum(labels), []).append(pid)
    for k, members in strata.items():
        random.Random(f"{seed}:{k}").shuffle(members)

    p2_quota = _allocate(strata, p2_target)
    p2 = {p for k, q in p2_quota.items() for p in strata[k][:q]}

    p1_strata = {k: [p for p in v if p not in p2] for k, v in strata.items()}
    p1 = {p for v in p1_strata.values() for p in v}
    n_val = round(len(p1) * val_frac)
    val_quota = _allocate(p1_strata, n_val)
    p1_val = {p for k, q in val_quota.items() for p in p1_strata[k][:q]}
    p1_train = p1 - p1_val
    return p1_train, p1_val, p2


def max_subclaims(rows: list[dict]) -> int:
    mix = parent_label_mix(rows)
    return max(len(v) for v in mix.values())


def verify_m_max() -> dict[str, int]:
    """Max subclaims/parent per WICE split; must all be <= M_MAX."""
    out = {}
    for split in ("train", "dev", "test"):
        rows = load_jsonl(UNIVERSAL_WICE / f"{split}.jsonl")
        out[split] = max_subclaims(rows)
    return out


def write_manifests(p1_train, p1_val, p2, out_dir: Path = MANIFEST_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = p1_train | p1_val
    for name, s in (("p1_parents", p1), ("p1_train_parents", p1_train),
                    ("p1_val_parents", p1_val), ("p2_parents", p2)):
        (out_dir / f"{name}.txt").write_text(
            "\n".join(sorted(s)) + "\n", encoding="utf-8")


def write_pooled_dataset(p1_train, p1_val, p2,
                         src_dir: Path = UNIVERSAL_WICE,
                         out_dir: Path = POOLED_DIR):
    """Filtered uni:wice_p1p2/{train,dev,test}.jsonl for the extract pipeline."""
    # all three pools are carved from the train split's rows
    train_rows = load_jsonl(src_dir / "train.jsonl")
    want = {"train": p1_train, "dev": p1_val, "test": p2}
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, keep in want.items():
        rows = [r for r in train_rows if parent_of(r) in keep]
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = (len(rows), len({parent_of(r) for r in rows}))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true",
                    help="verify manifests/dataset match a rebuild; write nothing")
    a = ap.parse_args()

    rows = load_jsonl(UNIVERSAL_WICE / "train.jsonl")
    p1_train, p1_val, p2 = build_split(rows, seed=a.seed)
    p1 = p1_train | p1_val
    assert not (p1 & p2) and len(p1) + len(p2) == len(parent_label_mix(rows))

    mm = verify_m_max()
    print("m_max per split:", mm, "| frozen M_MAX =", M_MAX)
    assert all(v <= M_MAX for v in mm.values()), "m_max exceeded — revisit cap"

    counts = {"P1": len(p1), "P1-train": len(p1_train),
              "P1-val": len(p1_val), "P2": len(p2)}
    print("parents:", counts)
    if a.check:
        ok = True
        for name, s in (("p1_parents", p1), ("p1_train_parents", p1_train),
                        ("p1_val_parents", p1_val), ("p2_parents", p2)):
            f = MANIFEST_DIR / f"{name}.txt"
            same = f.exists() and set(f.read_text().split()) == s
            ok &= same
            print(f"  {name}: {'OK' if same else 'MISMATCH/MISSING'}")
        for split, keep in (("train", p1_train), ("dev", p1_val),
                            ("test", p2)):
            f = POOLED_DIR / f"{split}.jsonl"
            same = (f.exists() and {parent_of(r) for r in load_jsonl(f)} == keep)
            ok &= same
            print(f"  wice_p1p2/{split}.jsonl: {'OK' if same else 'MISMATCH/MISSING'}")
        raise SystemExit(0 if ok else 1)
    write_manifests(p1_train, p1_val, p2)
    print("pooled dataset rows/parents:", write_pooled_dataset(p1_train, p1_val, p2))
    print("wrote manifests ->", MANIFEST_DIR)
    print("wrote dataset   ->", POOLED_DIR)


if __name__ == "__main__":
    main()
