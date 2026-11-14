"""Joint LoRA + block-diagonal probe training on WICE — ticket 05/07 arm.

Pipeline (single GPU, QLoRA 4-bit preferred, HF+peft fp16 fallback):
  1. load Qwen3.5-2B with LoRA adapters (r=16, all-linear targets)
  2. stream P1-train rows -> (ids, marks) windows via data.stream/RowEncoder
  3. per window: backbone forward -> last_hidden_state at mark positions
     -> hq (verify_end) + he (chunk ends) -> BlockDiagVerdictModel -> CE
  4. AdamW over LoRA+probe (param groups), cosine schedule, grad ckpt
  5. each epoch: eval on P1-val (chunk_f1 + row_f1 via tune_thresholds),
     keep best joint state on val row_f1 -> out_dir/joint.pt
  6. eval best on P2 test (+ optional --pairs pools) -> row metrics,
     chunk probs per row -> ClaimScore npz via calibrate/aggregate
  7. history -> out_dir/joint_history.csv (plot_training.py compatible)
     + W&B via runlog when credentials exist

Usage (repo root, data/universal must be populated):
  python experiments/lora_joint.py --out_dir runs/joint --blocks 2 \
      --pairs uni:wice:validation uni:wice:test
"""

from __future__ import annotations

import argparse, csv, math, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.entailment_network.data import (DataCfg, RowEncoder, load_universal,
                                         stream, supporting_set)
from src.entailment_network.runlog import finish as wb_finish, get_wandb, log as wb_log
from src.entailment_network.verifier_common import (LABELS, BlockDiagVerdictModel,
                                                    macro_f1, parse_label)

try:
    from tqdm import tqdm
except Exception:                                    # noqa: BLE001
    tqdm = None


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--out_dir", default="runs/joint")
    p.add_argument("--blocks", type=int, default=2,
                   help="BlockDiagVerdictModel block size (1,2,4)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_lr", type=float, default=2e-4)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=4, help="windows per micro-step")
    p.add_argument("--accum", type=int, default=8, help="grad accumulation steps")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--window_tokens", type=int, default=6144)
    p.add_argument("--chunk_tokens", type=int, default=128)
    p.add_argument("--claim_end", default="<|claim_end|>")
    p.add_argument("--ev_end", default="<|evidence_end|>")
    p.add_argument("--train", default="wice_p1p2", help="uni: dataset name")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val", default="wice_p1p2")
    p.add_argument("--val_split", default="validation")   # maps to dev.jsonl
    p.add_argument("--test", default="wice_p1p2")
    p.add_argument("--test_split", default="test")
    p.add_argument("--pairs", nargs="*", default=[],
                   help="extra uni:name:split pools to score with best model")
    p.add_argument("--scores_out", default="results/scores_joint")
    p.add_argument("--limit", type=int, default=None, help="rows per split (smoke)")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------ model
def load_joint_model(a, dev):
    """QLoRA-preferred loader. Returns (model, tokenizer, lora_params, adapted)."""
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=a.model, max_seq_length=a.max_model_len,
            load_in_4bit=True, load_in_16bit=False,
            full_finetuning=False)
        model = FastLanguageModel.get_peft_model(
            model, r=a.lora_r, lora_alpha=a.lora_alpha,
            lora_dropout=0.0, bias="none",
            target_modules="all-linear", use_gradient_checkpointing="unsloth",
            random_state=a.seed)
        return model, tok, True
    except Exception as e:
        print(f"unsloth path failed ({e}); HF+peft fallback", flush=True)

    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(torch_dtype=torch.float16)
    try:
        m = AutoModel.from_pretrained(a.model, **kw)
    except Exception:
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    base = getattr(m, "model", m)
    base = getattr(base, "language_model", base)
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0,
                     bias="none", task_type="FEATURE_EXTRACTION",
                     target_modules="all-linear")
    model = get_peft_model(base, cfg)
    if dev == "cuda":
        model = model.to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tok, False


def last_hidden(model, ids, attn):
    """[B,T,d] last hidden state, differentiable."""
    out = model(input_ids=ids, attention_mask=attn)
    hs = getattr(out, "last_hidden_state", None)
    if hs is None:
        hs = out.hidden_states[-1]
    return hs


def lora_params(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def adapted_modules(model):
    mods = set()
    for n, _ in model.named_modules():
        sd = getattr(_, "state_dict", lambda: {})()
        if any("lora_" in k for k in sd):
            mods.add(n)
    return sorted(mods)


# ------------------------------------------------------------------ data
def encode_split(model_tok, ds, meta_cfg, a, dev):
    """Stream rows -> padded micro-batches of (ids, attn, hq_pos, he_pos,
    chunk_y, row_idx). Yields dicts; labels mirror extract.py's rule."""
    enc = RowEncoder(model_tok, DataCfg(a.claim_end, a.ev_end,
                                        chunk_tokens=a.chunk_tokens,
                                        window_tokens=a.window_tokens))
    row_y = torch.tensor([parse_label(l) for l in ds["label"]])
    meta = []
    seqs = []
    for item in stream(enc, ds, meta):
        ids, marks = item
        if len(ids) > a.max_model_len:
            ids, marks = ids[:a.max_model_len], [m for m in marks
                                                 if m < a.max_model_len]
        if not marks:
            continue
        seqs.append((ids, marks))
    # group sequences into rows: meta[i] = (row, pos, spans) per sequence
    # chunk label: row_y if chunk has supporting sentence else NOT_SUP(2)
    batches, cur, cur_len = [], [], 0
    pad_id = model_tok.pad_token_id or model_tok.eos_token_id or 0
    for i, (ids, marks) in enumerate(seqs):
        r, pos, _ = meta[i]
        n_chunks = len(marks) - 1
        y = torch.tensor([int(row_y[r]) if p else 2 for p in pos],
                         dtype=torch.long)
        cur.append(dict(ids=ids, marks=marks, y=y, row=int(r)))
        cur_len = max(cur_len, len(ids))
        if len(cur) >= a.bs:
            yield _pad_batch(cur, pad_id, dev)
            cur, cur_len = [], 0
    if cur:
        yield _pad_batch(cur, pad_id, dev)


def _pad_batch(items, pad_id, dev):
    B = len(items); T = max(len(it["ids"]) for it in items)
    N = max(len(it["marks"]) - 1 for it in items)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    hq_pos = torch.zeros(B, dtype=torch.long)
    he_pos = torch.zeros(B, N, dtype=torch.long)
    he_mask = torch.zeros(B, N, dtype=torch.bool)
    ys, ymask, rows = [], [], []
    for b, it in enumerate(items):
        ids[b, :len(it["ids"])] = torch.tensor(it["ids"])
        attn[b, :len(it["ids"])] = 1
        hq_pos[b] = it["marks"][0]
        n = len(it["marks"]) - 1
        he_pos[b, :n] = torch.tensor(it["marks"][1:])
        he_mask[b, :n] = True
        ys.append(it["y"]); ymask.append(torch.ones(n, dtype=torch.bool))
        rows.append(it["row"])
    y = torch.full((B, N), -100, dtype=torch.long)
    ym = torch.zeros(B, N, dtype=torch.bool)
    for b, (yy, mm) in enumerate(zip(ys, ymask)):
        y[b, :len(yy)] = yy; ym[b] = mm
    return dict(ids=ids.to(dev), attn=attn.to(dev), hq_pos=hq_pos.to(dev),
                he_pos=he_pos.to(dev), he_mask=he_mask.to(dev),
                y=y.to(dev), ymask=ym.to(dev), rows=rows)


@torch.no_grad()
def batch_logits(model, probe, batch):
    """-> chunk logits [B,N,3] float cpu."""
    hs = last_hidden(model, batch["ids"], batch["attn"])      # [B,T,d]
    B = hs.shape[0]
    hq = hs[torch.arange(B, device=hs.device), batch["hq_pos"]]
    hq = hq.unsqueeze(1)                                     # [B,1,d]
    he = hs[torch.arange(B, device=hs.device)[:, None],
            batch["he_pos"].clamp(max=hs.shape[1] - 1)]        # [B,N,d]
    he = he.unsqueeze(2)                                     # [B,N,1,d]
    return probe(hq, he)[:, :, 0]                            # [B,N,3]


def chunk_loss(model, probe, batch):
    logits = batch_logits(model, probe, batch)
    return F.cross_entropy(logits[batch["ymask"]], batch["y"][batch["ymask"]])


# ------------------------------------------------------------------ eval
@torch.no_grad()
def eval_split(model, probe, tok, name, split, a, dev, need_probs=False):
    """Returns (metrics dict, per-row chunk probs list)."""
    ds = load_universal(name, split)
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    row_y = torch.tensor([parse_label(l) for l in ds["label"]])
    n_rows = len(ds)
    probs_rows = [[] for _ in range(n_rows)]
    chunk_y, chunk_pred = [], []
    model.eval(); probe.eval()
    for batch in encode_split(tok, ds, None, a, dev):
        logits = batch_logits(model, probe, batch)
        probs = logits.softmax(-1).cpu()
        for b in range(probs.shape[0]):
            row = batch["rows"][b]
            n = int(batch["ymask"][b].sum())
            probs_rows[row].append(probs[b, :n])
            chunk_pred.append(probs[b, :n].argmax(-1))
            chunk_y.append(batch["y"][b, :n].cpu())
    model.train(); probe.train()
    # flatten chunk-level
    cp = torch.cat(chunk_pred); cy = torch.cat(chunk_y)
    chunk_f1, _ = macro_f1(cp, cy)
    # row scores: max chunk prob per class (same as verifier_common.row_scores)
    s = torch.zeros(n_rows, 3)
    allp = torch.cat([torch.cat(v) if v else torch.zeros(0, 3)
                      for v in probs_rows])
    # rebuild chunk->row map in order
    crow = []
    for r, v in enumerate(probs_rows):
        for t in v:
            crow += [r] * t.shape[0]
    crow = torch.tensor(crow, dtype=torch.long)
    if len(allp):
        s.scatter_reduce_(0, crow[:, None].expand(-1, 3), allp,
                          reduce="amax", include_self=False)
    y = row_y
    best = (-1.0, 0.5, 0.5)
    grid = torch.linspace(0.05, 0.95, 19).tolist()
    for ts in grid:
        for tp in grid:
            pred = torch.full((n_rows,), 2, dtype=torch.long)
            pred[s[:, 1] >= tp] = 1
            pred[s[:, 0] >= ts] = 0
            f1, _ = macro_f1(pred, y)
            if f1 > best[0]:
                best = (f1, ts, tp)
    f1, ts, tp = best
    pred = torch.full((n_rows,), 2, dtype=torch.long)
    pred[s[:, 1] >= tp] = 1; pred[s[:, 0] >= ts] = 0
    rf1, per = macro_f1(pred, y)
    probs_list = [torch.cat(v).numpy() if v else np.zeros((0, 3))
                  for v in probs_rows]
    m = dict(chunk_f1=round(float(chunk_f1), 4), row_f1=round(float(rf1), 4),
             row_acc=round(float((pred == y).float().mean()), 4),
             thr=(round(float(ts), 3), round(float(tp), 3)))
    return m, probs_list, ds


# ------------------------------------------------------------------ main
def main():
    a = args(); torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    out_scores = Path(a.scores_out); out_scores.mkdir(parents=True, exist_ok=True)

    model, tok, used_unsloth = load_joint_model(a, dev)
    mods = adapted_modules(model)
    print(f"backend={'unsloth' if used_unsloth else 'hf+peft'} dev={dev} "
          f"adapted_modules={len(mods)}", flush=True)
    for m in mods[:50]:
        print("  lora on:", m, flush=True)

    d = getattr(getattr(model, "config", None), "hidden_size", None) \
        or getattr(getattr(model, "model", model).config, "hidden_size", 2048)
    L = 1
    probe = BlockDiagVerdictModel(int(d), L, K=3, block=a.blocks).to(dev)
    print(f"probe: block={a.blocks} params={sum(p.numel() for p in probe.parameters())}",
          flush=True)

    wb = get_wandb(name=f"joint-b{a.blocks}",
                   config=vars(a) | dict(backend="unsloth" if used_unsloth else "hf"))

    lp = [p for p in lora_params(model) if p.requires_grad]
    pp = list(probe.parameters())
    opt = torch.optim.AdamW([
        dict(params=lp, lr=a.lora_lr, weight_decay=a.wd),
        dict(params=pp, lr=a.probe_lr, weight_decay=a.wd)], lr=a.lora_lr)

    tr = load_universal(a.train, a.train_split)
    va = load_universal(a.val, a.val_split)
    te = load_universal(a.test, a.test_split)
    if a.limit:
        tr = tr.select(range(min(a.limit, len(tr))))
        va = va.select(range(min(a.limit, len(va))))
        te = te.select(range(min(a.limit, len(te))))

    # fraction-based warmup; total = epochs * est_steps
    est_steps_per_epoch = max(1, len(tr) // (a.bs * a.accum))
    total = a.epochs * est_steps_per_epoch
    warm = max(int(total * a.warmup_frac), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm)
        * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))

    hist_path = Path(a.out_dir) / "joint_history.csv"
    fresh = not hist_path.exists()
    hist = open(hist_path, "a", newline="")
    wr = csv.writer(hist)
    if fresh:
        wr.writerow(["epoch", "loss", "chunk_f1", "row_f1", "row_acc", "lr"])

    best = -1.0; step = 0; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); probe.train()
        ep_loss, nb, micro = 0.0, 0, 0
        pbar = tqdm(encode_split(tok, tr, None, a, dev),
                    desc=f"ep{ep}", unit="batch", leave=False) \
            if tqdm else encode_split(tok, tr, None, a, dev)
        for batch in pbar:
            loss = chunk_loss(model, probe, batch) / a.accum
            loss.backward()
            ep_loss += float(loss) * a.accum; nb += 1; micro += 1
            if micro % a.accum == 0 or micro == nb:
                torch.nn.utils.clip_grad_norm_(
                    lp + pp, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
        opt.zero_grad(set_to_none=True)
        m, _, _ = eval_split(model, probe, tok, a.val, a.val_split, a, dev)
        wr.writerow([ep, f"{ep_loss/max(nb,1):.6f}", m["chunk_f1"],
                     m["row_f1"], m["row_acc"], f"{sched.get_last_lr()[0]:.3e}"])
        hist.flush()
        print(f"ep{ep} loss {ep_loss/max(nb,1):.4f} | val chunk_f1 "
              f"{m['chunk_f1']:.4f} row_f1 {m['row_f1']:.4f} "
              f"acc {m['row_acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        wb_log(dict(epoch=ep, train_loss=ep_loss / max(nb, 1),
                    chunk_f1=m["chunk_f1"], row_f1=m["row_f1"],
                    row_acc=m["row_acc"], lr=sched.get_last_lr()[0]))
        if m["row_f1"] > best:
            best = m["row_f1"]
            torch.save(dict(probe=dict(block=a.blocks, d=int(d), K=3),
                            probe_state=probe.state_dict(),
                            lora_state={k: v.cpu() for k, v in
                                        model.state_dict().items()
                                        if "lora_" in k},
                            adapted_modules=mods,
                            thresholds=m["thr"], args=vars(a)),
                       f"{a.out_dir}/joint.pt")
            if hasattr(model, "save_pretrained"):
                try:
                    model.save_pretrained(f"{a.out_dir}/adapter")
                except Exception:
                    pass
    hist.close()

    # ---- eval best on test + extra pools ----
    ck = torch.load(f"{a.out_dir}/joint.pt", map_location="cpu",
                    weights_only=False)
    probe.load_state_dict(ck["probe_state"])
    ts, tp = ck["thresholds"]
    m_te, probs_te, ds_te = eval_split(model, probe, tok, a.test,
                                       a.test_split, a, dev)
    print("TEST", m_te, flush=True)
    wb_log({f"test/{k}": v for k, v in m_te.items()
            if isinstance(v, (int, float))})

    # ClaimScore npz for test + --pairs pools (calibrated on P2 test)
    from src.entailment_network.aggregator import aggregate
    from src.entailment_network.calibration import select_tau_rcps
    from src.entailment_network.claimscore import (ClaimScore, instance_id_of,
                                                   strict_y, u_from_quantities,
                                                   write_npz)
    from src.entailment_network.controller import decide as cdecide
    from src.entailment_network.eval_pipeline import _facet_mass, calibrate

    def stats_for(probs_list, ds):
        return [dict(probs=np.asarray(p, dtype=np.float64), spans=[],
                     label=parse_label(r["label"]), row=r)
                for p, r in zip(probs_list, ds)]

    calib_stats = stats_for(probs_te, ds_te)
    cal = calibrate(calib_stats, a.alpha, a.beta, a.delta)
    tau_s, tau_r = cal["tau_s"], cal["tau_r"]
    print(f"calib {a.test}:{a.test_split} -> tau_s={tau_s} tau_r={tau_r}",
          flush=True)

    def dump(name, split, probs_list, ds):
        stats = stats_for(probs_list, ds)
        recs = []
        for s in stats:
            p = np.asarray(s["probs"], dtype=np.float64)
            q = aggregate(p, np.zeros(len(p), int), [0],
                          tau_s=tau_s, tau_r=tau_r) if len(p) else \
                aggregate(np.zeros((0, 3)), np.zeros(0, int), [0],
                          tau_s=tau_s, tau_r=tau_r)
            recs.append(ClaimScore(
                u=u_from_quantities(q.support, q.coverage),
                y=strict_y(s["row"].get("label", "")),
                instance_id=instance_id_of(s["row"]),
                action=cdecide(q, tau_s).action,
                claim_id=str(s["row"].get("id", ""))))
        suf = f"{name.replace(':', '_')}_{split}"
        write_npz(recs, f"{out_scores}.{suf}.npz")
        print(f"-> {out_scores}.{suf}.npz ({len(recs)} claims)", flush=True)

    dump(a.test, a.test_split, probs_te, ds_te)
    for spec in a.pairs:
        name, split = spec.rsplit(":", 1)
        m_p, probs_p, ds_p = eval_split(model, probe, tok, name, split,
                                        a, dev)
        print(spec, m_p, flush=True)
        dump(name, split, probs_p, ds_p)

    wb_finish()
    print("== DONE ==", flush=True)


if __name__ == "__main__":
    main()
