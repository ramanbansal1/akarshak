# 04: Resplit protocol runner (cal/test resamples, n_cal & α sweeps, oracle gate)

**What to build:** the core experiment: pool dev+test parents (707), run
N random parent-level cal/test resplits; per resplit sweep
n_cal ∈ {100,200,300,500}, α ∈ {0.05..0.35}, |T| ∈ {50,100}; apply all
selectors; ALSO run oracle `u_j=y_j` per cell (no GPU). Outputs per cell:
certification rate, mean coverage, violation rate vs δ, and the three-region
gate classification (n-limited / verifier-limited / coverage ratio).
Reuses `experiments/oracle_feasibility.py` structure.

**Blocked by:** 03 (needs real u_j scores on pooled parents) — oracle half
can start immediately on labels alone

**Status:** ready-for-agent

- [x] Oracle-only mode runs today (extends oracle_feasibility.py to 1000 resplits)
- [ ] Real-score mode consumes eval_pipeline ClaimScore output
- [x] Emits results/*.csv + summary table for paper T3/F2/F3
- [x] Violation rate = fraction of resplits where test accepted-error > α

Progress: experiments/resplit_runner.py implemented — modes
oracle/noisy/real/both, n_cal x alpha x |T| sweeps, all 6 selectors,
per-cell cert_rate/coverage/violation_rate aggregation, gate_summary()
three-region classifier, CSV IO. select_tau_rcps vectorized via
scipy.special.betaincinv (116ms -> 2.7ms/call; identical CP bounds).
Oracle 1000-resplit run in progress -> results/oracle_resplit.csv.
Real mode consumes eval_pipeline --scores_out npz (needs ticket 05).
Tests: tests/test_resplit_runner.py (5 tests).

Oracle DONE: results/oracle_resplit.csv (308 cells, 1000 resplits).
Headline at alpha=0.20/grid50: macro certifies at ALL n_cal
(100-500) with ~55.5% coverage ceiling (= correct-claim fraction) and
0% violation; micro certifies only at n_cal=500 (bernstein) —
confirms macro-FDP as practical primary. Violation rate 0.0 in all
94 cells with cert_rate>0.5 (delta=0.10 sanity OK).
