"""
Fast Gemma-APS runner for RAGTruth (true batched generation).

- Real batched generate() with left padding
- Length-sorted batches to minimise padding / straggler waste
- Greedy decoding, KV cache, bf16 + SDPA
- Incremental JSONL writes + resume support
"""

import json
import argparse
import warnings
from pathlib import Path

import nltk
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

for pkg in ("punkt", "punkt_tab"):  # punkt_tab needed on newer NLTK
    nltk.download(pkg, quiet=True)

START, END, SEP = "<s>", "</s>", "\n"


def create_propositions_input(text: str) -> str:
    sents = nltk.tokenize.sent_tokenize(text)
    return SEP.join(f"{START} {s} {END}" for s in sents)


def build_prompt(tokenizer, text: str) -> str:
    msgs = [{"role": "user", "content": create_propositions_input(text)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens):
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        add_special_tokens=False,  # chat template already adds <bos>
    ).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2b-aps-it")
    p.add_argument("--output_dir", default="./data/processed")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Only process first N examples per split")
    p.add_argument("--splits", nargs="*", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ds = load_dataset("wandb/RAGTruth-processed")
    splits = args.splits or list(ds.keys())

    if args.dry_run:
        for s in splits:
            print(s, len(ds[s]))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"          # REQUIRED for batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    for split in splits:
        data = ds[split]
        ids, texts = data["id"], data["output"]
        if args.limit:
            ids, texts = ids[:args.limit], texts[:args.limit]

        out_path = Path(args.output_dir) / f"ragtruth_{split}_output.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = load_done_ids(out_path)

        items = [(i, t) for i, t in zip(ids, texts) if i not in done]
        # Build prompts once, sort by token length (longest first -> OOM surfaces early)
        prompts = [build_prompt(tokenizer, t) for _, t in items]
        lengths = [len(tokenizer(pr, add_special_tokens=False)["input_ids"]) for pr in prompts]
        order = sorted(range(len(items)), key=lambda k: -lengths[k])

        print(f"[{split}] {len(items)} to do ({len(done)} already done)")

        with open(out_path, "a", encoding="utf-8") as f:
            for b in tqdm(range(0, len(order), args.batch_size), desc=split):
                idx = order[b:b + args.batch_size]
                outs = generate_batch(model, tokenizer, [prompts[k] for k in idx], args.max_new_tokens)
                for k, props in zip(idx, outs):
                    f.write(json.dumps({
                        "id": items[k][0],
                        "original_response": items[k][1],
                        "extracted_propositions": props,
                    }) + "\n")
                f.flush()

    print("✓ Done")


if __name__ == "__main__":
    main()
