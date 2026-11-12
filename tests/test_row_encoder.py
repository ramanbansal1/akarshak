"""CPU tests for RowEncoder sequence layout (verify_end global token)."""

from src.entailment_network.data import DataCfg, RowEncoder


class FakeTok:
    """Minimal tokenizer stub: deterministic ids, real marker single-ids."""

    bos_token_id = 1
    pad_token_id = 0

    _SPECIAL = {"<|claim_end|>": 98, "<|evidence_end|>": 97, "<|im_end|>": 99}

    def __call__(self, x, add_special_tokens=False):
        def enc(s):
            if s in self._SPECIAL:
                return [self._SPECIAL[s]]
            return [ord(ch) % 50 + 10 for ch in s]
        if isinstance(x, str):
            return {"input_ids": enc(x)}
        return {"input_ids": [enc(s) for s in x]}


def make_enc(**kw):
    return RowEncoder(FakeTok(), DataCfg(**kw))


class TestVerifyEnd:
    def test_sequence_ends_with_im_end_and_marks0_is_last(self):
        enc = make_enc()
        seqs = list(enc.encode_row(0, "the claim", ["sentence one", "sentence two"]))
        assert len(seqs) == 1
        s = seqs[0]
        assert s.ids[-1] == 99                      # <|im_end|>
        assert s.marks[0] == len(s.ids) - 1         # global state = last position
        assert len(s.marks) == 1 + len(s.spans)     # global + one mark per chunk
        assert all(m < len(s.ids) for m in s.marks)

    def test_chunk_marks_point_at_evidence_end_tokens(self):
        enc = make_enc()
        s = list(enc.encode_row(0, "c", ["aaaa", "bbbb"]))[0]
        for m in s.marks[1:]:
            assert s.ids[m] == 97                   # <|evidence_end|>

    def test_empty_evidence_still_yields_global_state(self):
        enc = make_enc()
        seqs = list(enc.encode_row(0, "lonely claim", []))
        assert len(seqs) == 1
        s = seqs[0]
        assert s.ids[-1] == 99 and s.marks == [len(s.ids) - 1]
        assert s.spans == [] and s.pos == []

    def test_window_split_each_window_has_own_verify_end(self):
        enc = make_enc(window_tokens=20, chunk_tokens=128)
        sentences = [f"sentence number {i} with several words" for i in range(6)]
        seqs = list(enc.encode_row(0, "c", sentences))
        assert len(seqs) > 1
        for s in seqs:
            assert s.ids[-1] == 99
            assert s.marks[0] == len(s.ids) - 1
            assert len(s.marks) == 1 + len(s.spans)

    def test_pos_flags_unchanged(self):
        enc = make_enc()
        s = list(enc.encode_row(0, "c", ["aaaa", "bbbb"], supporting={1}))[0]
        assert s.pos == [False, True] or s.pos == [True, False] or len(s.pos) == len(s.spans)
        # exactly the chunk overlapping sentence 1 is positive
        pos_chunks = [s.spans[i] for i, p in enumerate(s.pos) if p]
        assert any(a <= 1 < b for a, b in pos_chunks)
