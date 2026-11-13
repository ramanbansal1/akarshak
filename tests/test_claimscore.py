"""CPU tests for the ClaimScore schema + parent-instance grouping (ticket 02)."""

import json

import numpy as np
import pytest

from src.entailment_network.claimscore import (ClaimScore, group_by_instance,
                                               instance_id_of, read_jsonl,
                                               read_npz, record_from_row,
                                               strict_y, to_instance_arrays,
                                               u_from_quantities, write_jsonl,
                                               write_npz)
from src.entailment_network.calibration import select_tau_hoeffding


def _row(rid: str, label: str) -> dict:
    return {"id": f"wice:{rid}", "claim": "c", "evidence": [], "label": label}


class _Q:
    def __init__(self, support, coverage):
        self.support, self.coverage = support, coverage


def test_strict_y_only_supported_is_correct():
    assert strict_y("supported") == 1
    assert strict_y("partially_supported") == 0
    assert strict_y("not_supported") == 0
    assert strict_y(" Supported ") == 1
    assert strict_y(0) == 1      # parse_label int: 0 = supported
    assert strict_y(1) == 0
    assert strict_y(2) == 0


def test_u_from_quantities_requires_full_coverage():
    assert u_from_quantities(0.9, 1.0) == pytest.approx(0.9)
    assert u_from_quantities(0.9, 0.5) == 0.0
    assert u_from_quantities(0.9, 0.0) == 0.0


def test_instance_id_of_parent_claim():
    assert instance_id_of(_row("train41647-0", "supported")) == "train41647"
    assert instance_id_of(_row("test00561-3", "not_supported")) == "test00561"


def test_record_from_row():
    r = record_from_row(_row("train1-0", "supported"), _Q(0.8, 1.0),
                        action="SUPPORTED")
    assert r.u == pytest.approx(0.8) and r.y == 1
    assert r.instance_id == "train1" and r.action == "SUPPORTED"
    r2 = record_from_row(_row("train1-1", "partially_supported"), _Q(0.7, 1.0))
    assert r2.y == 0 and r2.u == pytest.approx(0.7)


def test_to_instance_arrays_groups_by_parent():
    recs = [record_from_row(_row(f"p{i}-0", "supported"), _Q(0.9, 1.0))
            for i in range(3)]
    recs += [record_from_row(_row("p0-1", "not_supported"), _Q(0.9, 1.0))]
    arr = to_instance_arrays(recs)
    assert len(arr["instance_ids"]) == 3
    assert arr["scores"].shape == (4,) and arr["correct"].shape == (4,)
    assert arr["instances"][3] == arr["instances"][0]   # p0-1 shares p0's slot
    assert not arr["correct"][3]


def test_arrays_feed_select_tau_hoeffding():
    # two parents x two claims; p0 all correct high-score, p1 mixed
    recs = []
    for i in range(30):
        recs.append(record_from_row(_row(f"pa{i}-0", "supported"), _Q(0.9, 1.0)))
        recs.append(record_from_row(_row(f"pb{i}-0", "supported"), _Q(0.9, 1.0)))
        recs.append(record_from_row(_row(f"pb{i}-1", "not_supported"), _Q(0.1, 1.0)))
    arr = to_instance_arrays(recs)
    res = select_tau_hoeffding(arr["scores"], arr["correct"], arr["instances"],
                               alpha=0.35, delta=0.10, grid=50, m_max=6,
                               mode="macro")
    assert res["n_instances"] == 60


def test_jsonl_roundtrip(tmp_path):
    recs = [ClaimScore(u=0.5, y=1, instance_id="p1", action="SUPPORTED",
                       span=(0, 5), claim_id="wice:p1-0", meta={"k": 1}),
            ClaimScore(u=0.0, y=0, instance_id="p1", action="ABSTAIN")]
    f = tmp_path / "scores.jsonl"
    write_jsonl(recs, f)
    back = read_jsonl(f)
    assert back == recs
    assert json.loads(f.read_text().splitlines()[0])["span"] == [0, 5]


def test_npz_roundtrip(tmp_path):
    recs = [record_from_row(_row(f"p{i}-0", "supported"), _Q(0.9, 1.0))
            for i in range(4)]
    recs.append(record_from_row(_row("p0-1", "not_supported"), _Q(0.2, 1.0),
                                action="ABSTAIN"))
    f = tmp_path / "scores.npz"
    write_npz(recs, f)
    d = read_npz(f)
    assert d["scores"].shape == (5,) and d["correct"].tolist() == \
        [True, True, True, True, False]
    assert d["instances"][4] == 0        # p0-1 shares p0's instance slot
    assert d["instance_ids"] == ["p0", "p1", "p2", "p3"]
    assert d["action"][4] == "ABSTAIN"


def test_group_by_instance():
    recs = [ClaimScore(u=1, y=1, instance_id="a"),
            ClaimScore(u=0, y=0, instance_id="b"),
            ClaimScore(u=1, y=0, instance_id="a")]
    g = group_by_instance(recs)
    assert sorted(g) == ["a", "b"] and len(g["a"]) == 2
