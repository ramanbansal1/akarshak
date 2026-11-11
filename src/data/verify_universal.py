"""Verify data/universal rows: schema, NaN scan, and sampled side-by-side
comparison against the preserved original record (meta.raw)."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
UNI = DATA / "universal"

REQUIRED = {"id", "dataset", "claim", "claim_meta", "evidence", "label",
            "split", "meta"}


def _has_nan(obj) -> bool:
    if isinstance(obj, float) and math.isnan(obj):
        return True
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_nan(v) for v in obj)
    return False


def check_file(path: Path) -> tuple[int, list[str]]:
    problems = []
    n = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {i}: bad json {e}")
                continue
            miss = REQUIRED - r.keys()
            if miss:
                problems.append(f"line {i}: missing keys {miss}")
            if r.get("label", {}).get("channel") not in ("s", "r", "n"):
                problems.append(f"line {i}: bad channel {r.get('label')}")
            if not isinstance(r.get("label", {}).get("conflict"), bool):
                problems.append(f"line {i}: conflict not bool")
            if _has_nan(r):
                problems.append(f"line {i}: NaN value present")
            if r.get("split") != path.stem:
                problems.append(f"line {i}: split={r.get('split')} != {path.stem}")
            if r.get("dataset") != path.parent.name:
                problems.append(f"line {i}: dataset mismatch")
            for ev in r.get("evidence", []):
                for k in ("eid", "source_ref", "sentences"):
                    if k not in ev:
                        problems.append(f"line {i}: evidence missing {k}")
    return n, problems


def sample_rows(path: Path, k: int = 10, seed: int = 0):
    lines = path.read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(lines)), min(k, len(lines))))
    return [json.loads(lines[i]) for i in idx]


def show(r: dict) -> None:
    raw = r.get("meta", {}).get("raw", {})
    raw_label = raw.get("label", raw.get("Label", ""))
    print(f"  id={r['id']}  split={r['split']}")
    print(f"  claim: {r['claim'][:140]}")
    print(f"  raw_label={raw_label!r} -> channel={r['label']['channel']} "
          f"conflict={r['label']['conflict']}")
    evs = r.get("evidence", [])
    n_sent = sum(len(e.get("sentences", [])) for e in evs)
    print(f"  evidence: {len(evs)} doc(s), {n_sent} sentence(s), "
          f"first_ref={evs[0]['source_ref'][:60] if evs else ''!r}")
    keys = sorted(r.get("claim_meta", {}).keys())
    print(f"  claim_meta keys: {keys}")


def main() -> None:
    all_problems = []
    files = sorted(UNI.glob("*/*.jsonl"))
    if not files:
        print("no universal files found"); sys.exit(1)
    for p in files:
        n, probs = check_file(p)
        tag = p.relative_to(UNI)
        print(f"{tag}: {n} rows, {len(probs)} problems")
        for pr in probs[:5]:
            print("   ", pr)
        all_problems += [f"{tag}:{pr}" for pr in probs]

    print("\n--- random sample of 10 rows per dataset (first file each) ---")
    seen = set()
    for p in files:
        ds = p.parent.name
        if ds in seen:
            continue
        seen.add(ds)
        print(f"\n## {ds} ({p.relative_to(UNI)})")
        for r in sample_rows(p):
            show(r)

    print(f"\nTOTAL PROBLEMS: {len(all_problems)}")
    sys.exit(1 if all_problems else 0)


if __name__ == "__main__":
    main()
