# 07: Head capacity ablations: rank sweep + baselines

**What to build:** on cached P1/P2 feats (CPU once caches exist):
VerdictModel rank r ∈ {1,2,4,8,16,32,64,128,256,512} (tiny ranks added —
r=1 is the ~12k-param probe regime); BlockDiagVerdictModel block-size
sweep b ∈ {1,2,4,8,16} (K·d·b+K params; b=1 ~6k, b=2 ~12k at d=2048);
nearest-centroid baseline; layer sweep (last vs middle hidden layers —
needs feats re-extraction or multi-layer caching, flag if infeasible);
LightGBM on the 5 ClaimQuantities as alternative u_j (trained on P2, CV
within P2). Answers: does the head earn its parameters vs. centroid,
and does a ~10k-param probe suffice?

**Blocked by:** 05 (feats caches); 06 for the quantities

**Status:** pipeline-ready (dummy inputs); real run blocked on ticket 05

- [x] Rank sweep table: row macro-F1 vs r
      -> experiments/head_ablations.py --mode rank (ranks flag; trains on
         feats_train, early-stops on feats_validation, reports feats_test
         chunk_f1/row_f1/row_acc + n_params per rank -> head_ablations_rank.csv)
- [x] Block-diagonal head arm (~10k-param regime)
      -> src/entailment_network/verifier_common.py::BlockDiagVerdictModel
         (W_k block-diag, K*d*b+K params) + --mode block --blocks 1,2,4,8,16
         -> head_ablations_block.csv / _block_history.csv
- [x] Smoke on REAL caches (results/head_ablations_smoke/):
      lowrank r=1 (12,291 params): val row-F1 0.403, test row-F1 0.376,
        acc 0.436 | blockdiag b=2 (12,291 params): val 0.358, test 0.357,
        acc 0.548 — both retain most of the 3.1M head's signal at ~400x
        fewer params; full sweep still pending
- [x] Centroid baseline row in same table
      -> --mode centroid: nearest-centroid in he last-layer space through
         the same row_scores/decide path -> head_ablations_centroid.csv
- [x] LightGBM u_j variant feeds ticket 04's protocol as ablation
      -> --mode lgbm: 5 ClaimQuantities -> P(strict_y), GroupKFold over
         parent_id within P2 (honest OOF AUC), scores extra pools into
         results/scores_lgbm.*.npz (resplit_runner --mode real format)
- [ ] Layer sweep: INFEASIBLE — caches store last hidden layer only
      (verifier_common.default_layers); needs --layers change + GPU re-extract
- [ ] REAL run on downloaded runs/wice_p1p2 feats caches
