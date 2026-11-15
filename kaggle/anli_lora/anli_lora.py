"""ANLI + LoRA probe — single-file Kaggle script.

facebook/anli, 9 configs: {train,dev,test}_{r1,r2,r3}.
label: 0=entailment, 1=neutral, 2=contradiction.

Model: Qwen/Qwen2.5-1.5B + LoRA (r=16, all linear) + linear probe head
on the last hidden state at the <answer> position.
Prompt:  {hypothesis}<hypothesis-end> {premise}<premise-end> <answer>
Loss:    3-class cross-entropy. Metric: accuracy per split.
"""

import json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SPLITS = [f"{s}_{r}" for s in ("train", "dev", "test") for r in ("r1", "r2", "r3")]
MAX_LEN = 1024
EPOCHS = 2
BS = 8
ACCUM = 4                      # effective batch 32
LR = 1e-4
LORA_R, LORA_ALPHA = 16, 32
OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0); random.seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------- data
def load(split):
    return load_dataset("facebook/anli", split)

def prompt(hyp, prem):
    return f"{hyp}<hypothesis-end> {prem}<premise-end> <answer>"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def encode(rows):
    texts = [prompt(r["hypothesis"], r["premise"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    enc = tok(texts, truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    return enc, torch.tensor(labels)

# ------------------------------------------------------------- model
print(f"loading {MODEL_ID} on {dev} ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, attn_implementation="sdpa")
base.gradient_checkpointing_enable()
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                  target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, lora)
hidden = base.config.hidden_size
head = nn.Linear(hidden, 3).to(dev, torch.float16)
model.print_trainable_parameters()

params = [{"params": [p for p in model.parameters() if p.requires_grad]},
          {"params": head.parameters(), "lr": LR * 10}]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

# ------------------------------------------------------------------ train
def batch_iter(rows, shuffle=True):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), BS):
        sel = [rows[j] for j in idx[i:i + BS]]
        yield encode(sel)

@torch.no_grad()
def evaluate(rows):
    model.eval(); head.eval()
    correct = total = 0
    for enc, y in batch_iter(rows, shuffle=False):
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]          # state at <answer>
        pred = head(h.float()).argmax(-1).cpu()
        correct += (pred == y).sum().item(); total += len(y)
    model.train(); head.train()
    return correct / max(total, 1)

train_rows, dev_rows = [], {}
for split in SPLITS:
    ds = load(split)
    rows = [{"hypothesis": r["hypothesis"], "premise": r["premise"],
             "label": r["label"]} for r in ds]
    if split.startswith("train"):
        train_rows += rows
    else:
        dev_rows[split] = rows
    print(f"{split}: {len(rows)} rows", flush=True)

model.train(); head.train()
steps = EPOCHS * math.ceil(len(train_rows) / (BS * ACCUM))
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=[LR, LR * 10], total_steps=steps, pct_start=0.1)
t0 = time.time(); step = 0
for ep in range(EPOCHS):
    for enc, y in batch_iter(train_rows):
        enc = {k: v.to(dev) for k, v in enc.items()}
        y = y.to(dev)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        loss = F.cross_entropy(head(h.float()), y) / ACCUM
        loss.backward()
        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 200 == 0:
            print(f"ep{ep} step{step}/{steps} loss={loss.item()*ACCUM:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ eval
results = {}
for split, rows in dev_rows.items():
    results[split] = evaluate(rows)
    print(f"{split}: acc={results[split]:.4f}", flush=True)
results["train_subset_acc"] = evaluate(train_rows[:2000])
print("mean dev acc:", np.mean([v for k, v in results.items()
                               if k.startswith("dev")]), flush=True)

# --------------------------------------------------------------- save
model.save_pretrained(str(OUT / "lora_adapter"))
tok.save_pretrained(str(OUT / "tokenizer"))
torch.save(head.state_dict(), OUT / "head.pt")
(OUT / "metrics.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT, flush=True)
