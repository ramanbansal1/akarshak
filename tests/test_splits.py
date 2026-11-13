"""CPU tests for the deterministic WICE parent-level pool split (ticket 01)."""

import json

from src.entailment_network.splits import (M_MAX, _allocate, _stratum,
                                           build_split, max_subclaims,
                                           parent_label_mix, parent_of)


def _row(pid: str, sub: int, label: str = "supported") -> dict:
    return {"id": f"wice:{pid}-{sub}",
            "claim_meta": {"id": f"{pid}-{sub}"},
            "meta": {"raw": {"label": label}}, "claim": "c", "evidence": []}


def _synthetic_rows(n_parents=60, subs=(2, 3)):
    rows = []
    for i in range(n_parents):
        pid = f"train{i:05d}"
        labels = (["supported", "partially_supported"] if i % 3 == 0
                  else ["supported", "not_supported"])
        for s in range(subs[i % len(subs)]):
            rows.append(_row(pid, s, labels[s % len(labels)]))
    return rows


def test_parent_of_strips_subclaim_suffix():
    assert parent_of(_row("test00561", 0)) == "test00561"
    assert parent_of({"id": "wice:train41647-12"}) == "train41647"
    # no suffix -> id unchanged (dataset prefix still stripped)
    assert parent_of({"id": "wice:lonely"}) == "lonely"


def test_stratum_buckets_supported_fraction():
    assert _stratum(["supported"] * 3) == 1.0
    assert _stratum(["supported", "not_supported"]) == 0.6  # 0.5 ties up
    assert _stratum(["not_supported"] * 2) == 0.0


def test_build_split_is_deterministic_and_partitions():
    rows = _synthetic_rows()
    a = build_split(rows, seed=7, p2_target=12, val_frac=0.25)
    b = build_split(rows, seed=7, p2_target=12, val_frac=0.25)
    assert a == b
    p1_train, p1_val, p2 = a
    all_parents = set(parent_label_mix(rows))
    assert p1_train.isdisjoint(p1_val) and p1_train.isdisjoint(p2)
    assert p1_val.isdisjoint(p2)
    assert p1_train | p1_val | p2 == all_parents
    assert len(p2) == 12
    assert len(p1_val) == round((len(all_parents) - 12) * 0.25)


def test_build_split_stratifies_label_mix():
    rows = _synthetic_rows()
    _, _, p2 = build_split(rows, seed=1, p2_target=12)
    mix = parent_label_mix(rows)
    # every stratum contributes roughly proportionally (synthetic has 2 strata)
    strata = {}
    for pid, labels in mix.items():
        strata.setdefault(_stratum(labels), set()).add(pid)
    for members in strata.values():
        got = len(members & p2)
        assert 0 < got < len(members) or len(members) <= 2


def test_allocate_hits_total():
    strata = {0.0: [f"a{i}" for i in range(10)],
              1.0: [f"b{i}" for i in range(30)]}
    q = _allocate(strata, 8)
    assert sum(q.values()) == 8
    assert q[0.0] == 2 and q[1.0] == 6          # proportional 10:30 -> 2:6


def test_max_subclaims_and_m_max_constant():
    rows = []
    for i in range(5):
        for s in range(i + 1):                  # sizes 1..5
            rows.append(_row(f"p{i}", s))
    assert max_subclaims(rows) == 5
    assert M_MAX == 6                          # frozen WICE structural cap
