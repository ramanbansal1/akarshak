# 05: Docs sync — backbone swap + hq := g semantics

**What to build:** `docs/math.md` and `AGENTS.md` reflect the Qwen3.5-2B
backbone, the `<|im_end|>` global token, and the changed meaning of the `hq`
cache slot (global verify_end state, not claim_end state).

**Blocked by:** 02-verify-end-token

**Status:** ready-for-agent

- [ ] math.md sequence-layout and hidden-state sections describe im_end marker
- [ ] AGENTS.md commands/model references updated to Qwen3.5-2B path
- [ ] Known-approximation notes still accurate
