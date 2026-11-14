"""Phase 1: hidden-state extraction -> feature caches (feats_{split}.pt).

GPU-heavy stage. Run once per (dataset, backbone, data_cfg); caches are reused
by `python -m src.entailment_network.train` unless --force is passed.

  python -m src.pre_computing.extract --dataset uni:wice \
      --tp 2 --dtype float16 --backend vllm
"""

import argparse, os
from dataclasses import asdict

import torch

from src.entailment_network.data import (DataCfg, RowEncoder, load_universal,
                                         stream, supporting_sets)
from src.entailment_network.verifier_common import (
    NOT_SUP, default_layers, parse_label, save_cache, wice_chunk_labels)
from .hs_extract import HiddenStateExtractor


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="uni:wice",
                   help="HF dataset id[/config] or uni:<name> for data/universal")
    p.add_argument("--config", default="subclaim",
                   help="WICE config for HF datasets (ignored for uni:)")
    p.add_argument("--model", default="ibm-granite/granite-4.1-8b")
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--claim_end", default="<|claim_end|>")
    p.add_argument("--ev_end", default="<|evidence_end|>")
    p.add_argument("--chunk_tokens", type=int, default=128)
    p.add_argument("--window_tokens", type=int, default=6144)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--shm_tokens", type=int, default=150_000)
    p.add_argument("--dtype", default="float16", help="float16 on T4 (no bf16)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--shm_root", default="/dev/shm/hs_extract")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--hf_batch_tokens", type=int, default=65_536,
                   help="HF backend: max tokens per forward batch (lower on small VRAM)")
    p.add_argument("--force", action="store_true", help="re-extract even if cache exists")
    p.add_argument("--limit", type=int, default=None, help="first N rows per split (smoke test)")
    p.add_argument("--splits", nargs="*", default=["train", "validation", "test"],
                   choices=["train", "validation", "test"],
                   help="subset of cache splits to (re)build")
    p.add_argument("--out_dir", default="runs/wice")
    return p.parse_args()


def build_cache(ext, enc, ds, path, meta_cfg):
    meta, states = [], []
    for st in ext.extract(stream(enc, ds, meta)):
        states.append(st)
    row_y = torch.tensor([parse_label(l) for l in ds["label"]])
    # WiCE App. E.2: partition labels from supporting-sentence coverage,
    # not the row label (see wice_chunk_labels).
    sup = []
    for r in ds:
        try:
            sup.append(supporting_sets(r))
        except Exception:                                # noqa: BLE001 - no annotation -> empty
            sup.append([])
    labels = [wice_chunk_labels(m[2], sup[m[0]], int(row_y[m[0]])) for m in meta]
    seq_row = torch.tensor([m[0] for m in meta])
    save_cache(path, states, labels, meta_cfg, extra=dict(row_y=row_y, seq_row=seq_row))
    print(f"{path}: {len(ds)} rows, {len(states)} sequences, {sum(map(len, labels))} chunks")


def main():
    a = args(); os.makedirs(a.out_dir, exist_ok=True)
    dcfg = DataCfg(a.claim_end, a.ev_end, chunk_tokens=a.chunk_tokens,
                  window_tokens=a.window_tokens)
    wanted = set(a.splits)
    cp = {s: f"{a.out_dir}/feats_{s}.pt" for s in ("train", "validation", "test")
          if s in wanted}

    from transformers import AutoTokenizer
    layers = a.layers or default_layers(a.model)
    enc = RowEncoder(AutoTokenizer.from_pretrained(a.model), dcfg)
    ext = HiddenStateExtractor(a.model, layers, a.max_model_len, gpu_mem=a.gpu_mem,
                               max_chunk_tokens=a.shm_tokens,
                               dtype=a.dtype, tp=a.tp, shm_root=a.shm_root,
                               backend=a.backend, hf_batch_tokens=a.hf_batch_tokens)
    meta_cfg = dict(backbone=a.model, layers=layers, data_cfg=asdict(dcfg))
    try:
        for split in cp:
            if os.path.exists(cp[split]) and not a.force:
                print('skip (cached):', cp[split]); continue
            if a.dataset.startswith("uni:"):
                ds = load_universal(a.dataset[4:], split)
            else:
                from datasets import load_dataset
                ds = load_dataset(a.dataset, a.config, split=split)
            if a.limit:
                ds = ds.select(range(min(a.limit, len(ds))))
            build_cache(ext, enc, ds, cp[split], meta_cfg)
    finally:
        ext.close()


if __name__ == "__main__":
    main()
