"""ClaimScore record + parent-instance eval schema (ticket 02).

Every downstream consumer of per-claim scores — calibration selectors, the
resplit runner, metrics — reads this schema instead of raw pipeline rows.

Record: (u_j, y_j, instance_id, action?, span?, claim_id?)
  u_j         certified score in [0,1]; primary = support * 1{coverage == 1}
              (method.tex: only fully-covered claims may be SUPPORTED)
  y_j         strict binary correctness: WICE `supported` -> 1,
              `partially_supported` / `not_supported` -> 0
  instance_id WICE parent claim id (the i.i.d. unit for certification)
  action      controller action string, when available (audit trail)
  span        source span in the answer, when available
  claim_id    row-level id for traceability back to the dataset row

IO: JSONL (human-readable audit trail) and NPZ (compact arrays for the
resplit runner). `to_instance_arrays` produces exactly the (scores, correct,
instances) triple `calibration.select_tau_hoeffding` consumes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .splits import parent_of
from .verifier_common import parse_label


@dataclass
class ClaimScore:
    u: float
    y: int
    instance_id: str
    action: str | None = None
    span: tuple | None = None
    claim_id: str = ""
    meta: dict = field(default_factory=dict)


def strict_y(label) -> int:
    """Strict binary correctness: only `supported` is correct (y=1).

    Accepts WICE label strings, parse_label ints (0=supported), or bools.
    """
    if isinstance(label, str):
        return int(label.strip().lower() == "supported")
    return int(parse_label(label) == 0)


def u_from_quantities(support: float, coverage: float) -> float:
    """Certified per-claim score u_j = support * 1{coverage == 1}."""
    return float(support) if float(coverage) == 1.0 else 0.0


def instance_id_of(row: dict) -> str:
    """Parent instance id for a pipeline row (claim_meta.id / meta.id / id)."""
    return parent_of(row)


def record_from_row(row: dict, q, action: str | None = None) -> ClaimScore:
    """Build a ClaimScore from a pipeline row + aggregator ClaimQuantities."""
    return ClaimScore(
        u=u_from_quantities(q.support, q.coverage),
        y=strict_y(row.get("label", "")),
        instance_id=instance_id_of(row),
        action=action,
        claim_id=str(row.get("id", "")),
    )


def to_instance_arrays(records: list[ClaimScore]) -> dict:
    """(scores, correct, instances, instance_ids) for select_tau_hoeffding."""
    ids = sorted({r.instance_id for r in records})
    pos = {p: i for i, p in enumerate(ids)}
    return dict(
        scores=np.array([r.u for r in records], dtype=np.float64),
        correct=np.array([r.y == 1 for r in records], dtype=bool),
        instances=np.array([pos[r.instance_id] for r in records], dtype=int),
        instance_ids=ids,
    )


def group_by_instance(records: list[ClaimScore]) -> dict[str, list[ClaimScore]]:
    groups: dict[str, list[ClaimScore]] = {}
    for r in records:
        groups.setdefault(r.instance_id, []).append(r)
    return groups


# ----------------------------------------------------------------- IO
def write_jsonl(records: list[ClaimScore], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            d = asdict(r)
            if d.get("span") is not None:
                d["span"] = list(d["span"])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[ClaimScore]:
    out = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("span") is not None:
            d["span"] = tuple(d["span"])
        out.append(ClaimScore(**{k: v for k, v in d.items()
                                 if k in ClaimScore.__dataclass_fields__}))
    return out


def write_npz(records: list[ClaimScore], path: str | Path):
    """Compact arrays; instance_ids kept as a JSON string array."""
    arr = to_instance_arrays(records)
    np.savez_compressed(
        path,
        u=arr["scores"], y=arr["correct"].astype(np.int8),
        inst_idx=arr["instances"],
        instance_ids=np.array(arr["instance_ids"]),
        claim_id=np.array([r.claim_id for r in records]),
        action=np.array([r.action or "" for r in records]),
    )


def read_npz(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    ids = z["instance_ids"].astype(str)
    return dict(
        scores=z["u"].astype(np.float64),
        correct=z["y"].astype(bool),
        instances=z["inst_idx"].astype(int),
        instance_ids=[str(i) for i in ids],
        claim_id=[str(c) for c in z["claim_id"]],
        action=[str(a) for a in z["action"]],
    )
