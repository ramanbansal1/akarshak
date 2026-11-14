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
