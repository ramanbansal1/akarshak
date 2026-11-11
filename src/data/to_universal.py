"""Convert local raw datasets into the universal claim-verification format.

Universal row (one JSON object per line, split into per-split files):

    {"id", "dataset", "claim", "claim_meta",
     "evidence": [{"eid", "source_ref", "sentences": [{"sidx", "text"}]}],
     "label": {"channel": "s"|"r"|"n", "conflict": bool},
     "split": "train"|"dev"|"test",
     "meta": {"raw": {...original record...}}}

Splits are physical files: data/universal/<dataset>/<split>.jsonl, and each
row also carries its "split" field. The original record is preserved verbatim
under meta.raw. Datasets whose evidence is not locally resolvable to text
(FEVER doc/sent IDs, FEVEROUS table-cell IDs, AVeriTeC archive URLs) emit
evidence entries with source_ref provenance and sentences=[] — resolution is
a downstream step. Rows are only emitted for datasets that carry labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = DATA / "universal"

VERIFICATION_LABELS = {"s", "r", "n"}


# ---------------------------------------------------------------------------
# label mapping
# ---------------------------------------------------------------------------

def _chan(label: str) -> tuple[str, bool]:
    """Map any dataset label onto (channel, conflict).

    channel: s = supported, r = refuted, n = insufficient/unclear.
    conflict: evidence/labels are two-sided or explicitly disputed.
    """
    l = str(label).strip().lower()
    l = re.sub(r"\s+", " ", l)

    # --- explicit per-dataset vocabularies -------------------------------
    if l in ("supported", "supports", "supported "):
        return "s", False
    if l in ("partially_supported",):
        return "n", False  # partial support does not establish the claim
    if l in ("not_supported",):
        return "n", False  # WICE conflates refute+insufficient (see AGENTS.md)
    if l in ("refutes", "refuted"):
        return "r", False
    if l == "noinfo":
        return "n", False
    if l in ("conflicting evidence/cherrypicking", "conflicting"):
        return "n", True
    if l in ("not enough evidence", "not enough info", "nei"):
        return "n", False
    if l in ("sup",):  # scifact-style shorthand, defensive
        return "s", False

    # --- MultiFC (union of ~15 fact-checking sites) ------------------------
    _MF_S = {
        "true", "truth!", "mostly true", "mostly-true", "mostly_true",
        "mostly-correct", "mostly truth!", "promise kept", "verified", "fact",
        "accurate", "correct", "correct attribution", "correct attribution!",
        "authorship confirmed!", "confirmed authorship!", "in-the-green",
        "in the green", "grass roots movement!", "verdict: true",
        "determination: true", "determination: mostly true",
        "conclusion: accurate", "true messages", "factscan score: true",
        "previously truth! now resolved!",
    }
    _MF_R = {
        "false", "mostly false", "mostly_false", "mostly false ",
        "pants on fire!", "pants-fire", "pantsfire", "fiction!", "fiction",
        "fake", "fake news", "scam", "scam!", "bogus warning",
        "facebook scams", "full flop", "flop", "incorrect", "promise broken",
        "in-the-red", "in the red", "verdict: false", "determination: false",
        "conclusion: false", "rating: false", "factscan score: false",
        "we rate this claim false", "a lot of baloney", "a little baloney",
        "some baloney", "misleading", "misleading!", "misleading recommendations",
        "determination: misleading", "factscan score: misleading",
        "distorts the facts", "spins the facts", "exaggerated", "exaggerates",
        "miscaptioned", "misattributed", "inaccurate attribution!",
        "incorrect attribution!", "cherry picks", "3 pinnochios",
        "4 pinnochios", "determination: barely true",
        "determination: a stretch", "determination: huckster propaganda",
        "fiction! & satire!",
    }
    _MF_N = {
        "half-true", "half true", "half flip", "no flip", "none", "",
        "partially true", "partly true", "2 pinnochios",
        "not yet rated", "stalled", "in the works", "unproven", "unproven!",
        "unverified", "unsupported", "no evidence", "unobservable",
        "opinion!", "commentary!", "satire",
        "outdated", "outdated!", "conclusion: unclear",
        "verdict: unsubstantiated", "unsubstantiated messages",
        "statirical reports", "legend", "needs context", "not the whole story",
        "understated", "virus!",
    }
    _MF_CONFLICT = {
        "truth! & fiction!", "truth! & misleading!", "truth! & outdated!",
        "truth! & unproven!", "truth! & disputed!", "disputed!", "compromise",
        "mixture", "in-between",
    }
    if l in _MF_CONFLICT:
        return "n", True
    if l in _MF_S:
        return "s", False
    if l in _MF_R:
        return "r", False
    if l in _MF_N:
        return "n", False
    if l.isdigit():  # GossipCop 0-10 truth scale: low=false, high=true
        v = int(l)
        return ("r" if v <= 2 else "s" if v >= 8 else "n"), False
    if "&" in l:  # compound site ratings mixing truth and falsehood
        return "n", True

    # --- generic fallback --------------------------------------------------
    if "false" in l or "refut" in l:
        return "r", False
    if any(k in l for k in ("scam", "fake", "hoax", "bogus", "flop", "fiction")):
        return "r", False
    if "true" in l or "accurate" in l or "correct" in l or "support" in l:
        return "s", False
    return "n", False


def _clean_str(v) -> str:
    """No NaN/None leaks into structured fields."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _meta3(speaker=None, date=None, source_url=None, **extra) -> dict:
    meta = {
        "speaker": _clean_str(speaker),
        "date": _clean_str(date),
        "source_url": _clean_str(source_url),
    }
    meta.update({k: v for k, v in extra.items()})
    return meta


# ---------------------------------------------------------------------------
# adapters — each yields universal row dicts
# ---------------------------------------------------------------------------

def _wice(path: Path, dataset: str):
    df = pd.read_json(path, lines=True)
    for i, r in df.iterrows():
        meta = r.get("meta") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        ev_texts = r.get("evidence") or []
        evidence = []
        if isinstance(ev_texts, list) and ev_texts:
            evidence = [{
                "eid": "ev0",
                "source_ref": _clean_str(meta.get("doc") or meta.get("claim_title")),
                "sentences": [
                    {"sidx": j, "text": _clean_str(s)}
                    for j, s in enumerate(ev_texts)
                ],
            }]
        ch, cf = _chan(r["label"])
        claim_meta = dict(meta) if isinstance(meta, dict) else {}
        if r.get("supporting_sentences") is not None:
            claim_meta["supporting_sentences"] = r["supporting_sentences"]
        uid = _clean_str(meta.get("id")) or f"{path.stem}:{i}"
        yield {
            "id": f"{dataset}:{uid}",
            "dataset": dataset,
            "claim": _clean_str(r["claim"]),
            "claim_meta": claim_meta,
            "evidence": evidence,
            "label": {"channel": ch, "conflict": cf},
            "split": path.stem.split("_")[-1],
            "meta": {"raw": r.to_dict()},
        }


def _multi_fc(path: Path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for _, r in df.iterrows():
        if not _clean_str(r.get("label")):
            continue  # unlabeled rows (e.g. test split) carry no supervision
        ch, cf = _chan(r["label"])
        yield {
            "id": f"multi_fc:{_clean_str(r['claimID'])}",
            "dataset": "multi_fc",
            "claim": _clean_str(r["claim"]),
            "claim_meta": _meta3(
                speaker=r.get("speaker"),
                date=r.get("publish date"),
                source_url=r.get("claimURL"),
                checker=r.get("checker"),
                categories=r.get("categories"),
            ),
            "evidence": [],
            "label": {"channel": ch, "conflict": cf},
            "split": path.stem,
            "meta": {"raw": r.to_dict()},
        }


def _fever(path: Path):
    df = pd.read_json(path, lines=True)
    for _, r in df.iterrows():
        ch, cf = _chan(r["label"])
        # evidence = [[page_id, sent_id, page_title, sent_idx], ...] per group;
        # keep group-level provenance, sentences resolved downstream.
        evidence = []
        for gi, group in enumerate(r.get("evidence") or []):
            titles = []
            for tup in group:
                t = _clean_str(tup[2]) if len(tup) > 2 else ""
                if t and t not in titles:
                    titles.append(t)
            evidence.append({
                "eid": f"g{gi}",
                "source_ref": " | ".join(titles),
                "sentences": [],
            })
        yield {
            "id": f"fever:{_clean_str(r['id'])}",
            "dataset": "fever",
            "claim": _clean_str(r["claim"]),
            "claim_meta": _meta3(verifiable=r.get("verifiable")),
            "evidence": evidence,
            "label": {"channel": ch, "conflict": cf},
            "split": path.stem,
            "meta": {"raw": r.to_dict()},
        }


_CELL_PREFIX = re.compile(r"^(.*?)_(?:cell|section|header|title|table|sentence)(?:_|$)")


def _page_of(cell_id: str) -> str:
    m = _CELL_PREFIX.match(cell_id)
    return m.group(1) if m else cell_id


def _feverous(path: Path):
    df = pd.read_json(path, lines=True)
    for i, r in df.iterrows():
        if not _clean_str(r.get("claim")):
            continue  # header row
        ch, cf = _chan(r["label"])
        # evidence items: {"content": [cell_ids], "context": {cell_id: [ids]}}
        evidence = []
        for ei, item in enumerate(r.get("evidence") or []):
            cells = item.get("content") or []
            pages = []
            for c in cells:
                p = _page_of(_clean_str(c))
                if p and p not in pages:
                    pages.append(p)
            evidence.append({
                "eid": f"e{ei}",
                "source_ref": " | ".join(pages),
                "sentences": [],  # table cells resolved downstream
            })
        yield {
            "id": f"feverous:{_clean_str(r.get('id')) or i}",
            "dataset": "feverous",
            "claim": _clean_str(r["claim"]),
            "claim_meta": _meta3(challenge=r.get("challenge")),
            "evidence": evidence,
            "label": {"channel": ch, "conflict": cf},
            "split": "train",
            "meta": {"raw": r.to_dict()},
        }


def _averitec(path: Path):
    df = pd.read_json(path)  # top-level JSON array despite .jsonl name
    for i, r in df.iterrows():
        if not _clean_str(r.get("label")):
            continue  # unlabeled test claims
        ch, cf = _chan(r["label"])
        # answers carry archive.org source URLs — provenance stubs only
        evidence = []
        for qi, q in enumerate(r.get("questions") or []):
            for ai, a in enumerate(q.get("answers") or []):
                if isinstance(a, str):
                    a = {"answer": a}
                evidence.append({
                    "eid": f"q{qi}a{ai}",
                    "source_ref": _clean_str(
                        a.get("cached_source_url") or a.get("source_url")
                    ),
                    "sentences": [],
                })
        yield {
            "id": f"averitec:{i}:{_clean_str(r.get('claim_date'))}",
            "dataset": "averitec",
            "claim": _clean_str(r["claim"]),
            "claim_meta": _meta3(
                speaker=r.get("speaker"),
                date=r.get("claim_date"),
                source_url=r.get("original_claim_url")
                or r.get("fact_checking_article"),
                reporting_source=r.get("reporting_source"),
                location=r.get("location_ISO_code"),
                claim_types=r.get("claim_types"),
                n_questions=len(r.get("questions") or []),
            ),
            "evidence": evidence,
            "label": {"channel": ch, "conflict": cf},
            "split": "train" if "train" in path.name else "test",
            "meta": {"raw": r.to_dict()},
        }


# ---------------------------------------------------------------------------

def _denan(obj):
    """Recursively convert float NaN to None so JSON stays clean."""
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if isinstance(obj, dict):
        return {k: _denan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_denan(v) for v in obj]
    return obj


def _write(rows, ds: str, split: str, counts: dict) -> None:
    d = OUT / ds
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with (d / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            ch = r["label"]["channel"]
            assert ch in VERIFICATION_LABELS, f"bad channel {ch} in {r['id']}"
            f.write(json.dumps(_denan(r), ensure_ascii=False) + "\n")
            n += 1
    if n:
        counts[ds] = counts.get(ds, 0) + n


def main() -> None:
    import shutil
    if OUT.exists():
        shutil.rmtree(OUT)  # converter owns this directory; always regenerate
    counts: dict[str, int] = {}

    w = DATA / "wice"
    for split in ("train", "dev", "test"):
        _write(_wice(w / f"subclaim_{split}.jsonl", "wice"), "wice", split, counts)
        _write(_wice(w / f"claim_{split}.jsonl", "wice_claim"), "wice_claim", split, counts)
    for split in ("train", "dev"):
        # multi_fc test.csv is unlabeled -> skipped by adapter
        _write(_multi_fc(DATA / "multi_fc" / f"{split}.csv"), "multi_fc", split, counts)
    _write(_fever(DATA / "fever" / "train.jsonl"), "fever", "train", counts)
    _write(_feverous(DATA / "feverous" / "feverous_train_challenges.jsonl"),
           "feverous", "train", counts)
    _write(_averitec(DATA / "AVeriTeC" / "train.jsonl"), "averitec", "train", counts)

    for ds, n in sorted(counts.items()):
        print(f"{ds}: {n}")
    print(f"total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
