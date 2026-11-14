"""WiCE App. E.2 partition-label rule (wice_chunk_labels)."""
from src.entailment_network.verifier_common import wice_chunk_labels


def test_no_support_is_not_supported():
    # chunk covers sentences 0-1, supporting sentence is 5 -> NOT_SUP
    assert wice_chunk_labels([(0, 2)], [frozenset({5})], 0) == [2]


def test_all_support_plus_supported_row_is_supported():
    # single chunk contains every supporting sentence, row supported -> SUPPORTED
    assert wice_chunk_labels([(0, 4)], [frozenset({1, 3})], 0) == [0]


def test_any_complete_annotator_set_suffices():
    # two valid annotator sets; chunk covers set B fully -> SUPPORTED
    sets = [frozenset({1, 3}), frozenset({2})]
    assert wice_chunk_labels([(0, 3)], sets, 0) == [0]


def test_partial_coverage_on_supported_row_is_partial():
    # chunk has one of two supporting sentences -> PARTIALLY_SUPPORTED
    assert wice_chunk_labels([(0, 2)], [frozenset({1, 5})], 0) == [1]


def test_supporting_chunk_on_partial_row_is_partial():
    # even full coverage is only PARTIAL when the row is partially_supported
    assert wice_chunk_labels([(0, 4)], [frozenset({1, 3})], 1) == [1]


def test_supporting_chunk_on_not_supported_row_is_partial():
    assert wice_chunk_labels([(0, 4)], [frozenset({1, 3})], 2) == [1]


def test_mixed_chunks():
    # chunk0 no support -> 2; supporting split across chunk1/chunk2 so
    # neither holds a complete set -> both PARTIAL even on a supported row
    spans = [(0, 2), (2, 6), (6, 8)]
    assert wice_chunk_labels(spans, [frozenset({3, 7})], 0) == [2, 1, 1]
    # same spans, supporting fully inside chunk1 -> [2, 0, 2]
    assert wice_chunk_labels(spans, [frozenset({3, 4})], 0) == [2, 0, 2]


def test_empty_supporting_all_not_supported():
    assert wice_chunk_labels([(0, 3), (3, 5)], [], 0) == [2, 2]
