"""Gemma-APS adapter: proposition-segmentation output -> ExtractionResult.

Gemma-2b-APS (google/gemma-2b-aps-it) emits atomic propositions as plain text
(one per line in practice). This adapter wraps each proposition in an
AtomicClaim and reuses RuleBasedExtractor's slot-filling and edge inference so
the downstream schema (entities, dates, numbers, dependency edges) is populated
without an LLM call. Deferred module: not wired into the WICE path.
"""

from __future__ import annotations

import re
from pathlib import Path

from .atomic_claim_extractor import (AtomicClaim, DependencyEdge, ExtractionResult,
                                     RuleBasedExtractor)

_PROPOSITION_SPLIT = re.compile(r"\n+|(?<=[.!?])\s{2,}")


def parse_propositions(text: str) -> list[str]:
    """Split raw APS output into proposition strings, deduped, order preserved."""
    props, seen = [], set()
    for part in _PROPOSITION_SPLIT.split(text):
        p = part.strip().strip("-*• ")
        if len(p.split()) >= 3 and p.lower() not in seen:
            seen.add(p.lower())
            props.append(p)
    return props


class GemmaAPSAdapter:
    """extract(text) -> ExtractionResult, same interface as other extractors."""

    def __init__(self, nlp=None):
        self._rules = RuleBasedExtractor(nlp=nlp)

    def extract(self, text: str, propositions: str | None = None) -> ExtractionResult:
        """text: original response (for spans); propositions: raw APS output.
        If propositions is None, treats `text` itself as APS output."""
        props = parse_propositions(propositions if propositions is not None else text)
        if not props:
            return ExtractionResult(claims=[AtomicClaim(text=text, source_span=text)],
                                    raw_response=text)
        claims = []
        for p in props:
            c = AtomicClaim(text=p, source_span=p if p in text else "")
            claims.append(self._rules._fill_slots(c))
        edges = self._rules._infer_edges(claims)
        return ExtractionResult(claims=claims, edges=edges, raw_response=text)


def from_aps_jsonl(path: str | Path) -> list[ExtractionResult]:
    """Load run_aps_simple.py output JSONL -> ExtractionResults."""
    import json
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(GemmaAPSAdapter().extract(
                d.get("original_response", ""), d.get("extracted_propositions", "")))
    return out
