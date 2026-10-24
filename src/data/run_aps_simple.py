"""
Simple Gemma-APS Runner for RAGTruth from HuggingFace

Uses pipeline API with NLTK sentence tokenization and <s></s> markers.
"""

import logging
from transformers import logging as hf_logging

# Mute Hugging Face specific logs/warnings
hf_logging.set_verbosity_error()

# (Optional) Mute standard Python warnings if the warning persists
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # or just "ignore"


import json
import argparse
import nltk
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline
import torch
from tqdm import trange

# Download NLTK data
nltk.download('punkt', quiet=True)

start_marker = '<s>'
end_marker = '</s>'
separator = '\n'


def create_propositions_input(text: str) -> str:
    """Format text with <s></s> markers for Gemma-APS."""
    input_sents = nltk.tokenize.sent_tokenize(text)
    propositions_input = ''
    for sent in input_sents:
        propositions_input += f'{start_marker} ' + sent + f' {end_marker}{separator}'
    propositions_input = propositions_input.strip(f'{separator}')
    return propositions_input


def load_ragtruth():
    """Load RAGTruth dataset from HuggingFace."""
    print("Loading RAGTruth dataset from HuggingFace...")
    ds = load_dataset("wandb/RAGTruth-processed")
    print(f"✓ Loaded dataset with splits: {list(ds.keys())}")
    return ds


def run_gemaps_batch(
    examples,
    generator,
    max_new_tokens=256
):
    """Run Gemma-APS on a batch of examples using pipeline."""
    results = []
    ids, outputs = examples['id'], examples['output']
    for text, item_id in zip(ids, outputs):
        # Format with <s></s> markers
        formatted_input = create_propositions_input(text)
        messages = [{'role': 'user', 'content': formatted_input}]

        # Generate
        output = generator(messages, return_full_text=False)[0]['generated_text']
        propositions = output.strip()

        result = {
            "id": item_id,
            "original_response": text,
            "extracted_propositions": propositions,
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple Gemma-APS Runner")
    parser.add_argument("--model", type=str, default="google/gemma-2b-aps-it")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dry_run", action="store_true", help="Dry run without actual inference")

    args = parser.parse_args()

    # Load dataset
    ds = load_ragtruth()

    if args.dry_run:
        print("DRY RUN MODE - no actual inference")
        print(f"Would process {len(ds['train'])} train examples and {len(ds['test'])} test examples")
        print(f"Batch size: {args.batch_size}")
        print(f"Model: {args.model}")
        return

    # Load pipeline once
    print(f"Loading pipeline: {args.model}...")
    generator = pipeline('text-generation', args.model, device_map='auto', torch_dtype=torch.bfloat16)
    print("✓ Pipeline loaded")

    # Process each split
    for split_name in ds.keys():
        print(f"\n{'='*60}")
        print(f"Processing split: {split_name}")
        print(f"{'='*60}")

        examples = ds[split_name]
        total = len(examples)
        print(f"Total examples: {total}")

        # Process in batches
        batch_size = args.batch_size
        all_results = []

        for i in trange(0, total, batch_size):
            batch = examples[i:i+batch_size]
            print(f"\nBatch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}: {len(batch)} examples")

            results = run_gemaps_batch(
                batch,
                generator,
                args.max_new_tokens
            )
            all_results.extend(results)
            break

        # Save all results for this split
        output_path = Path(args.output_dir) / f"ragtruth_{split_name}_output.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
        print(f"✓ Saved {len(all_results)} results to {output_path}")

    print(f"\n{'='*60}")
    print("✓ All splits processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
