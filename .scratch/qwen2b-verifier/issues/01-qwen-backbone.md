# 01: Qwen3.5-2B text-backbone support in hidden-state extractor

**What to build:** `HiddenStateExtractor(backend="hf")` can initialize
`Qwen/Qwen3.5-2B` — loads the conditional-generation checkpoint, runs forward
passes through the text decoder (`language_model`), reads layer count/dim from
the nested `text_config`, and returns per-mark hidden states `[M, L, d]`.
Verified by a smoke forward showing correct output shape and hidden dim.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `default_layers("Qwen/Qwen3.5-2B")` returns the text model's last layer index
- [ ] HF backend init unwraps VLM wrapper to the text model (no vision-tower forward)
- [ ] `extract()` yields `[M, L, d]` tensors for a short token sequence on CPU
- [ ] Existing Granite path unchanged (tests still pass)
