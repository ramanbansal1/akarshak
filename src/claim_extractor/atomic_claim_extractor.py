"""
atomic_claim_extractor.py
─────────────────────────
Atomic Claim and Dependency Extractor (TRACE-RAG Component 1)

Design principles (grounded in the PDF literature review):
  • Each claim is atomic, decontextualized, and self-contained.
  • Structured JSON output: claim text + entity list + relation + date/quantity
    slots + source span + claim type (Luo 2024, Wang et al._1 2026).
  • Conjunctions, multiple predicates, multiple numbers/dates split recursively
    until each claim satisfies atomicity (Wang et al._1, 2026).
  • Dependency edges: entity_coreference, temporal_constraint,
    numerical_constraint, causal_dependency, semantic_entailment,
    semantic_conflict (Wang et al._1, 2026).
  • One batch API call for the full response — not per-sentence (PLAN.md goal).
  • Backends: rule-based fallback OR Ollama (local LLM).
"""

from __future__ import annotations

import json
import re
import textwrap
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

CLAIM_TYPES = frozenset({
    "biographical_fact", "numerical_value", "temporal_statement",
    "definition", "causal_relation", "general_fact",
})

DEP_TYPES = frozenset({
    "entity_coreference", "temporal_constraint", "numerical_constraint",
    "causal_dependency", "semantic_entailment", "semantic_conflict",
})


@dataclass
class AtomicClaim:
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    source_span: str = ""       # verbatim substring from original response
    entities: list = field(default_factory=list)
    subject: str = ""
    relation: str = ""          # primary predicate / verb phrase
    obj: str = ""
    temporal_expr: str = ""     # date / time expression if present
    numerical_value: str = ""   # quantity if present
    modality: str = ""          # negation, hypothetical, etc.
    claim_type: str = "general_fact"
    prob_fact: float = 1.0      # verifiability score 0-1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DependencyEdge:
    source_id: str
    target_id: str
    dep_type: str   # one of DEP_TYPES
    note: str = ""


@dataclass
class ExtractionResult:
    claims: list = field(default_factory=list)   # list[AtomicClaim]
    edges: list = field(default_factory=list)    # list[DependencyEdge]
    raw_response: str = ""
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "edges": [asdict(e) for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def claim_by_id(self, cid: str) -> Optional[AtomicClaim]:
        return next((c for c in self.claims if c.claim_id == cid), None)


# ─────────────────────────────────────────────────────────────────────────────
# Compact LLM prompt  (~20 lines as per PLAN.md goal)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
You are an atomic claim extractor for a fact-verification pipeline. Rules:
1. Each claim = exactly ONE factual proposition (atomic).
2. Resolve pronouns so each claim is self-contained (decontextualized).
3. Split claims that join independent predicates, multiple dates, or multiple quantities.
4. Do NOT add information absent from the source (faithful).
5. Preserve original order. After claims, identify dependency edges.

Output ONLY valid JSON, no markdown fences:
{"claims":[{"claim_id":"<id>","text":"<sentence>","source_span":"<verbatim>","entities":[],"subject":"","relation":"","obj":"","temporal_expr":"","numerical_value":"","modality":"","claim_type":"<biographical_fact|numerical_value|temporal_statement|definition|causal_relation|general_fact>","prob_fact":1.0}],"edges":[{"source_id":"","target_id":"","dep_type":"<entity_coreference|temporal_constraint|numerical_constraint|causal_dependency|semantic_entailment|semantic_conflict>","note":""}]}
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama client  (local LLM, no API key)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaExtractor:
    """
    Single-call extractor using a locally running Ollama model.
    One request for the entire response (not per-sentence).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        if not _REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def extract(self, text: str) -> ExtractionResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = _requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        return self._parse(raw, text, tokens)

    def _parse(self, raw: str, original: str, tokens: int) -> ExtractionResult:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.M)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.M)
        try:
            d = json.loads(cleaned)
        except json.JSONDecodeError:
            fallback = AtomicClaim(text=original, source_span=original)
            return ExtractionResult(claims=[fallback], raw_response=raw, tokens_used=tokens)
        claims = [AtomicClaim(**c) for c in d.get("claims", [])]
        edges = [DependencyEdge(**e) for e in d.get("edges", [])]
        return ExtractionResult(claims=claims, edges=edges, raw_response=raw, tokens_used=tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback (no LLM, no API key)
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedExtractor:
    """
    Sentence-split → conjunction-split → NER (spaCy optional) → slot-fill.
    Transparent baseline; use as comparison or when Ollama is unavailable.
    """

    _CONJ = re.compile(r"\band\b|\bbut\b|\bwhile\b|\bwhereas\b|\balso\b", re.I)
    _DATE = re.compile(
        r"\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s*\d{4})\b", re.I
    )
    _NUM = re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|percent|%))?)\b"
    )

    def __init__(self, nlp=None):
        self.nlp = nlp

    def extract(self, text: str) -> ExtractionResult:
        sents = self._split_sentences(text)
        raw: list[AtomicClaim] = []
        for s in sents:
            raw.extend(self._split_conjunctions(s))
        claims = [self._fill_slots(c) for c in raw]
        edges = self._infer_edges(claims)
        return ExtractionResult(claims=claims, edges=edges, raw_response=text)

    def _split_sentences(self, text: str) -> list[str]:
        if self.nlp:
            doc = self.nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _split_conjunctions(self, sent: str) -> list[AtomicClaim]:
        parts = [p.strip().rstrip(",;") for p in self._CONJ.split(sent)]
        parts = [p for p in parts if len(p.split()) >= 3]
        return [AtomicClaim(text=p, source_span=p) for p in parts] or \
               [AtomicClaim(text=sent, source_span=sent)]

    def _fill_slots(self, c: AtomicClaim) -> AtomicClaim:
        dm = self._DATE.search(c.text)
        if dm:
            c.temporal_expr = dm.group()
            c.claim_type = "temporal_statement"
        nm = self._NUM.search(c.text)
        if nm and not dm:
            c.numerical_value = nm.group()
            c.claim_type = "numerical_value"
        if self.nlp:
            doc = self.nlp(c.text)
            c.entities = list({e.text for e in doc.ents})
            for tok in doc:
                if tok.dep_ == "nsubj" and not c.subject:
                    c.subject = tok.text
                if tok.dep_ in ("dobj", "attr", "pobj") and not c.obj:
                    c.obj = tok.text
                if tok.pos_ == "VERB" and not c.relation:
                    c.relation = tok.lemma_
        return c

    def _infer_edges(self, claims: list[AtomicClaim]) -> list[DependencyEdge]:
        edges = []
        for i, a in enumerate(claims):
            for j, b in enumerate(claims):
                if i >= j:
                    continue
                shared = set(a.entities) & set(b.entities)
                if shared:
                    edges.append(DependencyEdge(
                        a.claim_id, b.claim_id, "entity_coreference",
                        note=f"shared: {', '.join(shared)}"
                    ))
                if a.temporal_expr and b.temporal_expr and a.temporal_expr != b.temporal_expr:
                    edges.append(DependencyEdge(
                        a.claim_id, b.claim_id, "temporal_constraint"
                    ))
        return edges


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_extractor(
    backend: str = "rule",
    **kwargs,
) -> RuleBasedExtractor | OllamaExtractor:
    """
    backend : "rule"   -> RuleBasedExtractor (no external deps)
              "ollama" -> OllamaExtractor (requires running Ollama)
    **kwargs forwarded to the chosen class constructor.
    """
    if backend == "ollama":
        return OllamaExtractor(**kwargs)
    nlp = None
    if _SPACY_AVAILABLE:
        try:
            nlp = _spacy.load("en_core_web_sm")
        except OSError:
            pass
    return RuleBasedExtractor(nlp=nlp)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EXAMPLE = (
        "The Golden Gate Bridge was constructed in 1937 and was designed "
        "by engineer Joseph Strauss. It spans 2,737 metres and was the "
        "longest suspension bridge in the world at the time of its completion."
    )
    ext = build_extractor("rule")
    res = ext.extract(EXAMPLE)
    print(res.to_json())
