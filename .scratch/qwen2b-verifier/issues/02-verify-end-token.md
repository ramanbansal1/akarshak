# 02: <|im_end|> global token in sequence encoding

**What to build:** every encoded sequence ends with the `<|im_end|>` special
token (id 248046) after the last `<|evidence_end|>` — and right after
`<|claim_end|>` for empty-evidence rows. `marks[0]` becomes the im_end position,
so the `hq` cache slot stores the global state `g` that attends to claim + all
evidence (mode A). Chunk marks, spans, pos flags, and windowing are unchanged.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `DataCfg` carries the verify-end marker string
- [ ] `encode_row` appends the marker per window and sets `marks[0]` to its position
- [ ] Empty-evidence rows still yield a sequence (claim + marker only)
- [ ] CPU unit test with a stub tokenizer covers marker position and span integrity
- [ ] `docs/math.md` / `AGENTS.md` updated for `hq := g` semantics
