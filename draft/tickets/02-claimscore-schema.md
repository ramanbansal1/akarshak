# 02: ClaimScore record + parent-instance eval schema

**What to build:** a single per-claim record `(u_j, y_j, instance_id,
action, source_span?)` where `instance_id` = WICE parent claim id, plus a
loader that groups subclaim rows into instances. Every downstream consumer
(calibration selectors, resplit runner, metrics) reads this schema instead
of raw rows. Strict labels: `supported`→y=1, else y=0. `u_j` computed as
`support·1{coverage=1}` per method.tex.

**Blocked by:** 01 (needs the parent manifest to define instance ids)

**Status:** done

- [x] Record type + JSONL/npz writer/reader in `src/entailment_network/`
- [x] Groups claims by parent id; exposes `instances` array for selectors
- [x] Unit test: synthetic parents → correct grouping, strict-y mapping

Notes: `src/entailment_network/claimscore.py` — ClaimScore dataclass
(u, y, instance_id, action, span, claim_id, meta), strict_y (supported->1
else 0, accepts WICE strings and parse_label ints), u_from_quantities
(support * 1{coverage==1}), record_from_row, to_instance_arrays (feeds
select_tau_hoeffding directly), JSONL+NPZ IO. `data.py:universal_row` now
passes through row `id` so instance_id_of resolves parent ids on pipeline
rows. Tests: tests/test_claimscore.py (9 tests).
