# 04: Full WICE re-extraction + head train + eval on Qwen3.5-2B

**What to build:** full `data/universal/wice` extraction into `runs/qwen2b/`,
bilinear head training, and dev/test evaluation — establishing the Qwen
backbone baseline metrics (row/chunk macro-F1, calibrated thresholds).

**Blocked by:** 03-kaggle-smoke

**Status:** ready-for-agent

- [ ] `runs/qwen2b/feats_{train,validation,test}.pt` cover all splits
- [ ] `runs/qwen2b/head.pt` trains to completion with logged per-epoch metrics
- [ ] Test-split metrics recorded for the paper's results table
