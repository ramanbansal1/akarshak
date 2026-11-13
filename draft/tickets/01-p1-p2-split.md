# 01: Deterministic P1/P2 parent split for WICE train

**What to build:** a fixed, seeded split of WICE `subclaim_train` parent
claims into P1 (~1,060 parents, head training + ~10% internal val fold for
early stopping) and P2 (~200 parents, aggregator/LightGBM tuning only).
Dev+test are NOT touched — they stay pooled for cal/test resplits.
Writes `data/wice/p1_parents.txt`, `data/wice/p2_parents.txt`,
`data/wice/p1_val_parents.txt` (or equivalent manifest) and a small loader
that filters `subclaim_train.jsonl` rows by parent id.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Split is deterministic (fixed seed 20240613), stratified by parent
      supported-fraction bucket (0.2 width, largest-remainder allocation)
- [x] P1+P2 partition covers all 1,260 train parents exactly once
      (P1=1060: 954 train + 106 val; P2=200; verified by `--check`)
- [x] Loader: `src/entailment_network/splits.py` + filtered universal dataset
      `data/universal/wice_p1p2/{train,dev,test}.jsonl` (works with existing
      `uni:` loaders unchanged); unit tests in `tests/test_splits.py`
- [x] `m_max=6` verified as structural cap across all splits (train/dev/test
      all = 6), frozen as `splits.M_MAX`; `calibration.py` callers should
      pass it explicitly (currently defaults to empirical max)

Notes: manifests at `data/wice/{p1,p1_train,p1_val,p2}_parents.txt`.
P1-val is a subset of P1 (early stopping only — never aggregator tuning).
`python -m src.entailment_network.splits --check` verifies reproducibility.
