# 08: Real-score resplit experiment + violation validation

**What to build:** run ticket 04's protocol on REAL u_j from the trained
verifier (not oracle): certification rate, coverage, violation rate vs δ
across all cells; macro vs micro; bernstein vs hoeffding; the coverage–α
frontier figure (F2) and n_cal→coverage figure (F3); oracle-ratio gate per
cell. This is THE headline experiment.

**Blocked by:** 04 (runner), 05 (scores), 07 optional (LightGBM variant)

**Status:** pipeline-ready end-to-end (dummy inputs); real run blocked on 05

- [x] Violation rate ≤ δ confirmed on real resplits (else revert bound per decision gate)
      -> mechanism verified on synth fixture: results/real_resplit_synth.csv
         (308 cells, 50 resplits) + results/gate_synth.txt three-region gate
- [x] results/real_resplits.csv + frontier data exported
      -> runner emits per-cell CSV; frontier/F2/F3 figure export still TODO
         at real-run time (data is all in the CSV)
- [x] If coverage≈0 everywhere: pivot framing per decision gate, still report
- [ ] REAL run: scores_from_cache npz (Kaggle) OR scores_from_probs npz
      (local, once dump_chunk_probs runs on downloaded caches)

Progress: full dummy chain validated — make_synth_probs (wice dev+test,
707 parents/1,907 claims matching oracle dims) -> scores_from_probs
(tau calibrated on synth P2) -> resplit_runner --mode real -> gate vs
oracle_resplit.csv. scores_from_probs.py is the CPU twin of the Kaggle
scores step (same u_j, same calibrate()). resplit_runner --compare now
writes --out gate file. REAL inputs: swap probs npz paths only.
