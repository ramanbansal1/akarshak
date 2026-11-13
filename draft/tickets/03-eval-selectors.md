# 03: Wire certified selectors into eval_pipeline with instance ids

**What to build:** `eval_pipeline --split test` emits, per run:
empirical τ, claim-level rcps τ, and `select_tau_hoeffding` τ for
`bound ∈ {bernstein, hoeffding}` × `mode ∈ {micro, macro}` on the
fixed 50-point grid, `m_max=6`. Output includes τ̂, coverage, error rate,
U(τ), n_instances — the audit trail for the paper.

**Blocked by:** 02

**Status:** done

- [x] eval_pipeline consumes ClaimScore records, passes parent ids as `instances`
- [x] All 4 selector variants reported side-by-side in output JSON
- [x] Unit test on cached/synthetic scores (no GPU)

Notes: eval_pipeline.py gains claim_records() (aggregate -> ClaimScore per
row, k=1 path, action from controller.decide), certified_block() emitting
baselines {empirical, rcps} + certified {bernstein,hoeffding}x{micro,macro}
with calib+test tau/coverage/error_rate/U/Lbar/n_instances, wired into
main() under metrics["certified"]; m_max=M_MAX(6) only for *wice* datasets.
--scores_out writes calib/test ClaimScore npz for the resplit runner.
Tests: tests/test_eval_certified.py (4 tests, no GPU).
