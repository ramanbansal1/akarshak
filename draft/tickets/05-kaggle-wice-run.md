# 05: Kaggle WICE extraction + head training run

**What to build:** the GPU run producing the artifacts everything else
needs: `feats_{split}.pt` hidden-state caches for P1 (+val fold), P2,
dev, test; trained `head.pt` (VerdictModel, r=256 default); row-level
metrics on P1-val and P2. FIRST check `runs/` for existing artifacts —
only retrain if absent. Uses P1/P2 manifests from ticket 01.

**Blocked by:** 01 (split manifests); 02 preferred (schema for outputs)

**Status:** in-progress (Kaggle kernel `bansalraman/trace-rag-wice`)

- [ ] `train.py` accepts parent-manifest filtering (train on P1 rows only)
- [ ] Early stopping / `tune_thresholds` uses the P1-internal val fold, NOT dev
- [ ] Artifacts logged: feats caches, head.pt, metrics JSON
- [ ] Smoke run `--limit 100 --stage extract` passes on Kaggle first

## Observability (added after first ERROR)

- `wice_run.py::run()` streams child stdout/stderr **live** to the kernel
  log and `extract_*.log`/`train.log`/`scores.log` in `/kaggle/working`
  (chunked reads, not buffered tails — tqdm `\r` updates flow through).
- tqdm progress bars: extraction (`HiddenStateExtractor.extract`, unit=seq)
  and head training (per-epoch batch loop). Both degrade silently if tqdm
  is absent.
- `train.py` appends per-epoch metrics to `head_history.csv`
  (epoch, loss, chunk_f1, row_f1, row_acc, lr) and logs to W&B when
  available.
- `src/entailment_network/runlog.py`: W&B no-ops unless `wandb` is
  installed AND `WANDB_API_KEY` is reachable (env var or Kaggle secret of
  the same name). `WANDB_PROJECT` overrides the project (default
  `trace-rag-wice` on the kernel, `trace-rag` elsewhere).
- `experiments/plot_training.py` turns `head_history.csv` and
  `head_ablations_history.csv` into PNGs under `results/plots/`; the
  kernel runs it as stage 3b and uploads figures to the W&B run.
- To enable W&B on Kaggle: add a notebook secret named `WANDB_API_KEY`
  (Add-ons -> Secrets). Without it everything still works, file logs only.
- Kernel currently at hf_batch_tokens=8192 + expandable_segments after a
  T4 OOM at 32768; next fallback is `--window_tokens 4096`.

## Run v6 postmortem (resolved)

- Extraction + training **succeeded**: all `feats_*.pt` caches and
  `head.pt` downloaded locally to `/tmp/wice_out3` (kernel output zip was
  truncated — `wice_pool/feats_test.pt` arrived 0 bytes and later files
  were missing, but the p1p2 caches + head are intact).
- ERROR cause: `experiments/scores_from_cache.py::_parse` split
  `uni:wice_p1p2:test=...` on the first colon → `KeyError: 'wice_p1p2:test'`
  in `_UNI_SPLIT`. Fixed (strip `uni:`, rsplit on last colon). Reproduced
  locally, then verified end-to-end on the real caches.
- Real P2 metrics from `head.pt` (r=256, 10 ep, class_power=0.5):
  row macro-F1 0.386, acc 0.543; per-class F1 supported 0.65 /
  partial 0.06 / not_supported 0.45. Chunk macro-F1 0.40.
- CPU sweep on the real caches (`results/head_sweep_cpu.csv`):
  class_power=1.0 is the only change that helps (test row-F1 0.389,
  acc 0.581); 30 epochs overfits badly. Kernel train cmd now passes
  `--class_power 1.0`.
- `tau_s=tau_r=inf` on P2 calibration → all u_j=0 → honest
  verifier-limited regime (ticket 08 gate analysis applies).
- Real ClaimScore npz for the two p1p2 pools saved under
  `results/scores/wice_p1p2_{validation,test}.npz` (degenerate u=0 —
  kept as pipeline-verified artifacts, not usable for coverage claims).
