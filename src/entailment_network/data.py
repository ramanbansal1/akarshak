from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataCfg:
    claim_end: str = "<|claim_end|>"
    ev_end: str = "<|evidence_end|>"
    verify_end: str = "<|im_end|>"   # global readout token; marks[0] -> hq slot
    max_claim_tokens: int = 128
    chunk_tokens: int = 128        # target tokens per evidence unit
    window_tokens: int = 6144      # evidence tokens per sequence (keep head+window < max_model_len)


@dataclass
class Seq:
    row: int
    ids: list
    marks: list        # [verify_end_pos, chunk_end_pos, ...]
    spans: list        # (sent_start, sent_end) per chunk, end exclusive
    pos: list          # chunk overlaps a supporting sentence (training only)


class RowEncoder:
    """(claim, sentence list) -> windows of: claim<claim_end> chunk<ev_end> chunk<ev_end> ..."""

    def __init__(self, tokenizer, cfg: DataCfg):
        self.tok, self.cfg = tokenizer, cfg
        enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
        self.enc = enc
        self.claim_end_ids, self.ev_end_ids, self.nl = enc(cfg.claim_end), enc(cfg.ev_end), enc("\n")
        self.verify_end_ids = enc(cfg.verify_end)
        self.prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    def encode_row(self, row: int, claim: str, sentences, supporting=frozenset()):
        c = self.cfg
        head = self.prefix + self.enc(claim)[: c.max_claim_tokens] + self.claim_end_ids
        sent_ids = self.tok(list(sentences), add_special_tokens=False)["input_ids"] if len(sentences) else []

        chunks, cur, start = [], [], 0                      # greedy sentence packing
        for j, s in enumerate(sent_ids):
            s = s[: c.chunk_tokens]
            if cur and len(cur) + len(s) + len(self.nl) > c.chunk_tokens:
                chunks.append((start, j, cur)); cur, start = [], j
            cur += s + self.nl
        if cur:
            chunks.append((start, len(sent_ids), cur))

        # each window ends with verify_end; marks[0] = its position (global state)
        ids, cmarks, spans, pos = list(head), [], [], []

        def finish():
            ids.extend(self.verify_end_ids)
            return Seq(row, list(ids), [len(ids) - 1] + cmarks, spans, pos)

        for a, b, toks in chunks:
            add = toks + self.ev_end_ids
            if spans and len(ids) - len(head) + len(add) + len(self.verify_end_ids) > c.window_tokens:
                yield finish()
                ids, cmarks, spans, pos = list(head), [], [], []
            ids += add
            cmarks.append(len(ids) - 1); spans.append((a, b))
            pos.append(any(a <= i < b for i in supporting))
        yield finish()


def supporting_set(row) -> frozenset:
    return frozenset(i for s in row["supporting_sentences"] for i in s)   # union over annotators


def stream(enc: RowEncoder, ds, meta: list):
    """Lazy (ids, marks) stream for the extractor; per-sequence metadata is appended to `meta`."""
    for r, row in enumerate(ds):
        for s in enc.encode_row(r, row["claim"], row["evidence"], supporting_set(row)):
            meta.append((s.row, s.pos, s.spans))
            yield s.ids, s.marks


# ----------------------------------------------------------------- universal JSONL source
class UniversalSplit:
    """List-backed split exposing the HF-Dataset surface the pipeline uses
    (__len__, __iter__, __getitem__ incl. column access, .select)."""

    def __init__(self, rows: list):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, k):
        if isinstance(k, str):                       # column access, like datasets.Dataset
            return [r[k] for r in self.rows]
        return self.rows[k]

    def select(self, idx):
        return UniversalSplit([self.rows[i] for i in idx])


_UNI_SPLIT = {"train": "train", "dev": "dev", "validation": "dev", "test": "test"}


def universal_row(r: dict) -> dict:
    """Universal JSONL row -> pipeline row {claim, evidence, supporting_sentences, label}.

    `evidence` is flattened to a list of sentence strings; `label` is the
    dataset-native label from meta.raw (e.g. WICE supported/partially_supported/
    not_supported), not the universal s/r/n channel — the head's K=3 classes
    are dataset-native.
    """
    sents = [s["text"] for e in r.get("evidence", []) for s in e.get("sentences", [])]
    return {
        "claim": r["claim"],
        "evidence": sents,
        "supporting_sentences": (r.get("claim_meta") or {}).get("supporting_sentences") or [],
        "label": (r.get("meta", {}).get("raw") or {}).get("label", ""),
    }


def load_universal(name: str, split: str) -> UniversalSplit:
    """Load data/universal/<name>/<split>.jsonl ('validation' maps to 'dev')."""
    path = (Path(__file__).resolve().parents[2] / "data" / "universal" / name
            / f"{_UNI_SPLIT[split]}.jsonl")
    rows = [universal_row(json.loads(l)) for l in path.open(encoding="utf-8") if l.strip()]
    return UniversalSplit(rows)
