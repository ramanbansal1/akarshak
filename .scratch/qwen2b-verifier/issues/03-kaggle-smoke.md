# 03: Kaggle smoke — extract 100 WICE rows on Qwen3.5-2B

**What to build:** run `src.pre_computing.extract` on Kaggle (2×T4, HF backend,
float16, reduced window/batch flags) with `--model Qwen/Qwen3.5-2B --limit 100`
and produce `feats_{train,validation,test}.pt` smoke caches whose tensor
shapes and mark counts are sane (hidden dim = Qwen text dim, chunks+1 marks
per sequence).

**Blocked by:** 01-qwen-backbone, 02-verify-end-token

**Status:** ready-for-agent

- [ ] Smoke extraction completes without OOM on 2×T4
- [ ] Cache files exist for all three splits with expected shapes
- [ ] Backend actually used is confirmed in logs (HF, not silent vLLM fallback)
