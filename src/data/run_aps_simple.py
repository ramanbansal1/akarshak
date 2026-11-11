"""
Simple Gemma-APS runner for RAGTruth using transformers pipeline.
"""

import json
import argparse
from pathlib import Path

import nltk
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

nltk.download('punkt', quiet=True)

start_marker = '<s>'
end_marker = '</s>'
separator = '\n'


def create_propositions_input(text: str) -> str:
    input_sents = nltk.tokenize.sent_tokenize(text)
    propositions_input = ''
    for sent in input_sents:
        propositions_input += f'{start_marker} ' + sent + f' {end_marker}{separator}'
    return propositions_input.strip(f'{separator}')


def run_gemaps_batch(examples, generator, max_new_tokens=256):
    results = []
    for item in examples:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get('output', item.get('response', item.get('text', '')))
        else:
            continue

        formatted_input = create_propositions_input(text)
        messages = [{'role': 'user', 'content': formatted_input}]
        output = generator(messages, return_full_text=False)[0]['generated_text']

        results.append({
            'id': item.get('id', None) if isinstance(item, dict) else None,
            'original_response': text,
            'extracted_propositions': output,
            'model': generator.model.name_or_path,
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2b-aps-it")
    p.add_argument("--output_dir", default="./data/processed")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ds = load_dataset("wandb/RAGTruth-processed")

    if args.dry_run:
        print(f"Train: {len(ds['train'])} examples")
        print(f"Test: {len(ds['test'])} examples")
        print(f"Batch size: {args.batch_size}")
        print(f"Model: {args.model}")
        return

    generator = pipeline(
        'text-generation',
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'test']:
        data = ds[split]
        total = len(data)
        output_path = output_dir / f"ragtruth_{split}_output.jsonl"

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, total, args.batch_size), desc=split):
                batch = data[i:i + args.batch_size]
                results = run_gemaps_batch(batch, generator, args.max_new_tokens)
                for result in results:
                    f.write(json.dumps(result) + '\n')
                f.flush()

    print("✓ Done")


if __name__ == "__main__":
    main()
