# 06: Aggregation + decision ablations on P2

**What to build:** sweep on P2 parents only (never dev/test):
γ ∈ {0,0.25,0.5,0.75,1.0}; weakest-link vs arithmetic mean; ancestor gate
on/off (subclaim groupings); k=1 vs k>1 facets; τ_s/τ_n/ε around defaults;
row macro-F1 + claim-level risk–coverage per setting. Outputs deltas for
paper T4. Challenge slices (agree-yet-incomplete, ancestor-blocked) run on
synthetic data only, labelled as such.

**Blocked by:** 05 (needs trained head + feats on P2)

**Status:** pipeline-ready (dummy inputs); real run blocked on ticket 05

- [x] One runner script, all sweeps, deterministic seeds
      -> experiments/aggregation_ablations.py (gamma x pool x tau_factors x
         eps = 270 settings; --synthetic_facets K + --ancestor_gate for the
         synthetic-only k>1/ancestor slices, tagged synthetic=1)
- [x] Results in results/ablation_*.csv
      -> results/ablation_synth_p2.csv (FIXTURE — synthetic probs, not a result)
- [x] No dev/test rows touched (assert in code)
      -> load_probs() hard-fails unless every parent_id is in p2_parents.txt
- [x] Enabler: experiments/dump_chunk_probs.py — feats cache + head.pt ->
      probs_{name}_{split}.npz (chunk probs + labels + parent ids); runs on
      CPU once ticket 05's runs/ artifacts are downloaded
- [x] Fixture: experiments/make_synth_probs.py — real P2 ids/labels +
      label-correlated synthetic probs -> results/probs_synth_p2.npz
- [x] aggregator.aggregate() gained pool={geomean,mean} + eps params
      (backward compatible; geomean+EPS default unchanged)
- [ ] REAL run: dump_chunk_probs on downloaded feats caches -> ablations
