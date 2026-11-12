# TRACE-RAG — Pipeline Mathematics

Complete math of the WICE verification path, stage by stage, as implemented in
`src/entailment_network/` + `src/pre_computing/`.

## 1. Data → sequences (`data.py`, `RowEncoder`)

Universal row → flattened `(claim, evidence=[s₁…s_N], supporting_sentences, label)`.

**Sentence packing** (greedy): sentence token lists `s_j` are appended to the
current chunk while `|cur| + |s_j| + 1 ≤ chunk_tokens` (128); else a new chunk
starts. Each chunk is a contiguous sentence span `(a, b)`.

**Sequence layout** per window:

```
[BOS] claim_tokens[:128] <|claim_end|>  chunk₁ <|evidence_end|>  ...  chunkₙ <|evidence_end|>  <|im_end|>
```

- `marks[0]` = position of `<|im_end|>` → **global state `g`**: with causal
  attention it is the only position that has seen the claim AND every chunk,
  so it fills the `hq` (claim-side) slot of the bilinear head — mode A
- `marks[1:]` = positions of each `<|evidence_end|>` → chunk states `he`
- `<|claim_end|>` is kept in the text for structure but its state is unused
- Windows split when evidence tokens exceed `window_tokens` (6144); the head
  (claim + marker) repeats per window, and each window gets its own `<|im_end|>`

**Chunk label** (training): `y_i = row_label` if chunk span `(a,b)` overlaps any
supporting sentence index, else `not_supported` (2). Supporting set = union over
annotator groups.

## 2. Hidden states (`pre_computing/hs_extract.py`)

Backbone `f_θ` (Qwen3.5-2B text decoder, frozen; was granite-4.1-8b). For each
sequence, collect the last-layer hidden state at each mark:

```
h_q   = f_θ(x)[marks[0]]    ∈ R^d     (global state g: claim + all evidence)
h_e,i = f_θ(x)[marks[i]]    ∈ R^d     (chunk i state)
```

`save_cache` stores `hq [S,L,d]` (S sequences), `he [M,L,d]` (M chunks),
`n_ev`, `y`, `seq_row` (chunk→row), `row_y`.

## 3. Bilinear verdict head (`verifier_common.VerdictModel`)

Low-rank factorization `W_k ≈ U_k V_kᵀ`, rank `r=256`, K=3 classes:

```
qu[k,b,r] = Σ_d hq[b,d] · U[k,d,r]
qe[k,b,n,r] = Σ_d he[b,n,d] · V[k,d,r]

z_k(b,n) = Σ_{r=1..R} qu[k,b,r] · qe[k,b,n,r] + bias_k     logits [B,N,K]
p_i = softmax(z_i) ∈ Δ²      per-chunk verdict probs (s, m, r)
```

Only the **last cached layer** is used (`hq[:, -1, :]`, `he[:, :, -1, :]`);
`U,V ~ N(0, 1/√d)` init.

**Training loss** (chunk-level weighted cross-entropy):

```
L = (1/|B|) Σ_b Σ_n w_{y_bn} · CE(softmax(z_bn), y_bn)
w_c = freq_c^{-class_power} / mean(freq^{-class_power})     class_power = 0.5
```

AdamW (lr=1e-3, wd=0.01), cosine schedule with ~5% warmup, grad clip 1.0,
10 epochs; checkpoint = best **row macro-F1** on validation. Checkpoint stores
state_dict, model_cfg, labels, thresholds, backbone/layers/data_cfg.

## 4. Chunk → row scores (`row_scores`, `decide`, `tune_thresholds`)

Row score per class = **max over chunks** (existential evidence — one good
chunk suffices):

```
s_row[c] = max_{i ∈ row} p_i[c],    c ∈ {supported, partial, not_supported}
```

Head-level decision (used for checkpoint selection):

```
pred = not_supported
if s_row[1] ≥ t_part: pred = partially_supported
if s_row[0] ≥ t_sup:  pred = supported        # supported checked last, overrides
```

`t_sup, t_part` tuned on dev by grid search `linspace(0.05, 0.95, 19)²`
maximizing row macro-F1.

## 5. Aggregator — §2.3 evidence algebra (`aggregator.py`)

Per claim, chunk probs `p_i = (s_i, m_i, r_i)` renormalized. On WICE k=1
(claim = own facet), all chunks in one facet, `weights=1`.

**Facet pooling** (weighted means over the facet's chunks):

```
S_f = Σ w_i s_i / Σ w_i ,   M_f = Σ w_i m_i / Σ w_i ,   R_f = Σ w_i r_i / Σ w_i
eff_f = S_f + γ·M_f ,   γ = 0.5          (partial support discounted)
```

**Facet verdicts:**

```
contradicted_f  ⟺  R_f ≥ τ_r  AND  R_f > S_f
established_f   ⟺  eff_f ≥ τ_s  AND  ¬contradicted_f  AND  n_chunks ≥ 1
                   AND  all ancestors established    (dependency edges; vacuous at k=1)
```

**Claim quantities:**

```
support       = ( Π_f max(eff_f, ε) )^{1/k}     geometric mean — weakest link, ε=1e-6
coverage      = (1/k) Σ_f established_f         fraction of established facets
contradiction = max_f R_f
S_tot, R_tot  = mean_f S_f, mean_f R_f
conflict      = 2·S_tot·R_tot / (S_tot + R_tot)  two-sided mass (0 if both 0)
agreement     = 1 − H(argmax-label distribution)/ln 3    source agreement via label entropy
missing_facets = {f : ¬established_f}
```

Core design principle: `eff_f = 0` when a facet has **no chunks** — missing
evidence is not weak support; and `partially_supported` contributes only
`γ·M`, never full credit.

## 6. Calibration — §2.4 (`calibration.py`, on validation split)

**τ̂_r** (refute threshold) — smallest τ on `R = mean_f R_f` s.t.
false-rejection rate ≤ β:

```
τ̂_r = min { τ :  P(¬refuted | R ≥ τ) ≤ β }        β = 0.10
```

**τ̂_s** (support threshold) — smallest τ on `eff` s.t. accepted-error ≤ α,
computed on non-contradicted claims (`R < τ̂_r or R ≤ S`):

```
τ̂_s = min { τ :  P(label ≠ supported | eff ≥ τ, ¬contradicted) ≤ α }   α = 0.10
```

If no τ satisfies α → `τ = ∞` (reject everything — honest abstention).

**ECE** (10 confidence bins):

```
ECE = Σ_bins (|B_m|/N) · |acc(B_m) − conf(B_m)|
```

**Risk–coverage**: sort claims by confidence `q.support` desc; at each prefix
fraction report `(coverage = k/N, risk = 1 − mean(correct[:k]))`.

## 7. Controller — decision table (`controller.py` + `decide_action`)

```
if ∃ facet f: contradicted_f ∧ S_tot > 0     → QUALIFY_CONFLICT
elif coverage == 1.0 ∧ support ≥ τ̂_s        → SUPPORTED
else                                          → ABSTAIN
```

`ABSTAIN` is the honest bucket: retrieve-more and correct-claim collapse into
it in the skeleton (retrieval budget = 0); `Decision.report` keeps the full
audit trail (per-facet S/R/M/eff/established/blocked_by, all claim quantities)
for a real controller later.

In `eval_pipeline`, actions map to WICE labels for scoring:
`SUPPORTED→0, QUALIFY_CONFLICT→1 (≈partially_supported), ABSTAIN→2 (≈not_supported)`
— documented approximation.

## 8. Metrics

- **Chunk macro-F1**: per-class `F1_c = 2TP/(2TP+FP+FN)`, mean over 3 classes,
  on chunk argmax vs chunk labels.
- **Row/action macro-F1**: same formula on decided actions vs row labels.
- **Row accuracy**, **abstain/qualify rates**,
  **accepted-error rate** = `P(label ≠ supported | action = SUPPORTED)` checked
  against α, **ECE**, **risk–coverage curve**.

## The one-line version

Frozen Granite states → low-rank bilinear scorer gives per-chunk `(s,m,r)` →
max-pool to row / weighted-mean to facets → `eff = S + 0.5M` with
geometric-mean weakest-link support → contradiction gate `R ≥ τ_r` →
establishment needs coverage 1.0 → calibrated τ's guarantee accepted-error ≤ α
on validation → three-way action with full audit report.

## Known approximations

- WICE `not_supported` conflates refute + insufficient evidence; the r-channel
  is really "not established". The refute/unclear split is deferred to
  FEVER/SciFact-style data.
- k=1 facets on WICE (claim = own facet); the ancestor/dependency gate is
  implemented but untested on real multi-facet data.
- Action→label mapping in eval is approximate (QUALIFY_CONFLICT ≈
  partially_supported, ABSTAIN ≈ not_supported).
- Controller's retrieve/correct actions are stubbed (ABSTAIN only).
