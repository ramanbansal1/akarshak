import pytest

from src.claim_extractor.atomic_claim_extractor import (AtomicClaim,
                                                        ExtractionResult)
from src.claim_extractor.gemma_aps_adapter import (GemmaAPSAdapter,
                                                   parse_propositions)
from src.claim_extractor.granite_aps import GraniteAPSExtractor


class TestParsePropositions:
    def test_splits_lines_and_dedupes(self):
        raw = "The bridge was built in 1985.\nThe bridge was built in 1985.\n\nIt spans 2km."
        props = parse_propositions(raw)
        assert props == ["The bridge was built in 1985.", "It spans 2km."]

    def test_drops_too_short(self):
        assert parse_propositions("Yes.\nNo.\nIt happened.") == []


class TestGemmaAPSAdapter:
    def test_extract_produces_claims_with_slots(self):
        ad = GemmaAPSAdapter()
        res = ad.extract(
            "The bridge was built in 1985 and spans 2.5 km.",
            propositions="The bridge was built in 1985.\nThe bridge spans 2.5 km.")
        assert isinstance(res, ExtractionResult)
        assert len(res.claims) == 2
        assert all(isinstance(c, AtomicClaim) for c in res.claims)
        assert res.claims[0].temporal_expr == "1985"
        assert res.claims[1].numerical_value == "2.5"

    def test_empty_propositions_falls_back(self):
        res = GemmaAPSAdapter().extract("Some original response text here.")
        assert len(res.claims) == 1
        assert res.claims[0].source_span == "Some original response text here."


class TestGraniteAPSStub:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            GraniteAPSExtractor().extract("anything")
