"""Granite-APS extractor hook (deferred — weights do not exist yet).

The proposal's end state is a Granite-8B model fine-tuned (distilled) into an
abstractive proposition segmenter, sharing the verifier backbone. This stub
fixes the interface now so the fine-tuned checkpoint drops in without touching
downstream code: same extract(text) -> ExtractionResult contract as
OllamaExtractor / GemmaAPSAdapter / RuleBasedExtractor.

Intended implementation: load the fine-tuned checkpoint, prompt with the APS
format used during distillation, parse propositions via
gemma_aps_adapter.parse_propositions, then slot-fill/edge-infer as in
GemmaAPSAdapter.
"""

from __future__ import annotations

from .atomic_claim_extractor import ExtractionResult


class GraniteAPSExtractor:
    """Placeholder for the fine-tuned Granite APS extractor."""

    def __init__(self, checkpoint: str | None = None, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs

    def extract(self, text: str) -> ExtractionResult:
        raise NotImplementedError(
            "Granite APS weights not trained yet. Train the APS-distilled "
            "checkpoint first, then wire it here (see module docstring)."
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: str, **kwargs) -> "GraniteAPSExtractor":
        return cls(checkpoint=checkpoint, **kwargs)
