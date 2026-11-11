# TRACE-RAG — working notes

Research proposal in `context.md` (source of truth). PRD for the current
milestone: `issues/prd.md`.

## Layout

- `src/entailment_network/` — WICE verification path (wired end-to-end):
  `train.py` (extract+train), `infer.py` (checkpoint → verdicts),
  `eval_pipeline.py` (verifier → aggregator → controller → metrics),
  `verifier_common.py`
  (cache, VerdictModel, metrics), `data.py` (claim+evidence → token windows),
  `aggregator.py` (§2.3 math, numpy), `calibration.py` (τ̂_s/τ_r, ECE,
  risk-coverage), `controller.py` (3 actions).
- `src/pre_computing/` — hidden-state extraction (`hs_extract.py`,
  `HiddenStateExtractor`: vLLM `pooling_task="all"` + HF fallback).
- `src/claim_extractor/` — deferred modules behind flags (not in WICE path):
  `gemma_aps_adapter.py`, `edge_finder.py` (RRF+BM25+dense+rules),
  `granite_aps.py` (stub for future fine-tune), `atomic_claim_extractor.py`.
- `tests/` — CPU-only pytest suite (no GPU/datasets/transformers needed).

## Commands

- Tests: `python3 -m pytest tests/ -q`
- Entrypoints (repo root, `python -m`):
  - `python -m src.entailment_network.train --limit 100 --stage extract` (smoke)
  - `python -m src.entailment_network.train` (full: extract → train head)
  - `python -m src.entailment_network.infer --ckpt runs/wice/head.pt`
  - `python -m src.entailment_network.eval_pipeline --ckpt runs/wice/head.pt --split test`
- Kaggle: 2×T4, `--tp 2 --dtype float16 --backend vllm` (HF fallback automatic).
- `datasets`/`transformers`/`vllm` imports are lazy — only needed for the
  stages that use them; pure modules and tests run without them.

## Known limitations (skeleton)

- WICE `not_supported` conflates refute+insufficient (verified vs paper
  arXiv:2303.01432); refute/unclear split deferred to FEVER/SciFact.
- k=1 facets on WICE (claim = own facet); ancestor gate untested on real data.
- vLLM per-position hidden states use pooling `"all"` task — validate on Kaggle;
  HF backend is the guaranteed fallback.
- Controller emits {SUPPORTED, QUALIFY_CONFLICT, ABSTAIN}; retrieve/correct stubbed.
