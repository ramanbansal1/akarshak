# 07: Head capacity ablations: rank sweep + baselines

**What to build:** on cached P1/P2 feats (CPU once caches exist):
VerdictModel rank r ∈ {32,64,128,256,512}; nearest-centroid baseline;
layer sweep (last vs middle hidden layers — needs feats re-extraction or
multi-layer caching, flag if infeasible); LightGBM on the 5 ClaimQuantities
as alternative u_j (trained on P2, CV within P2). Answers: does the 3.1M
head earn its parameters vs. centroid?

**Blocked by:** 05 (feats caches); 06 for the quantities

**Status:** pipeline-ready (dummy inputs); real run blocked on ticket 05

- [x] Rank sweep table: row macro-F1 vs r
      -> experiments/head_ablations.py --mode rank (ranks flag; trains on
         feats_train, early-stops on feats_validation, reports feats_test
         chunk_f1/row_f1/row_acc + n_params per rank -> head_ablations_rank.csv)
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
