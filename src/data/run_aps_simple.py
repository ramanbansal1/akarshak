"""
Simple Gemma-APS Runner for RAGTruth from HuggingFace

Loads dataset from wandb/RAGTruth-processed and runs Gemma-APS on all rows.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_ragtruth():
    """Load RAGTruth dataset from HuggingFace."""
    print("Loading RAGTruth dataset from HuggingFace...")
    ds = load_dataset("wandb/RAGTruth-processed")
    print(f"✓ Loaded dataset with splits: {list(ds.keys())}")
    return ds


def run_gemaps_batch(
    examples,
    model,
    tokenizer,
    device,
    max_new_tokens=256
):
    """Run Gemma-APS on a batch of examples (model loaded externally)."""
    results = []
    for item in tqdm(examples, desc="Processing"):
        # Handle different dataset formats
        if isinstance(item, str):
            text = item
            item_id = ""
            item_model = ""
            item_labels = []
        else:
            text = item.get('response', item.get('text', ''))
            item_id = item.get('id', '')
            item_model = item.get('model', '')
            item_labels = item.get('labels', [])

        prompt = f"Extract atomic propositions from the following text:\n\n{text}\n\nPropositions:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        propositions = generated_text[len(prompt):].strip()

        result = {
            "id": item_id,
            "original_response": text,
            "extracted_propositions": propositions,
            "model": item_model,
            "labels": item_labels
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple Gemma-APS Runner")
    parser.add_argument("--model", type=str, default="google/gemma-2b-aps-it")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    # Load dataset
    ds = load_ragtruth()

    # Load model ONCE (to avoid OOM)
    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None
    )

    if args.device == "cpu":
        model = model.to(args.device)

    model.eval()
    print("✓ Model loaded (will be reused for all batches)")

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

        for i in range(0, total, batch_size):
            batch = examples[i:i+batch_size]
            print(f"\nBatch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}: {len(batch)} examples")

            results = run_gemaps_batch(
                batch,
                model,
                tokenizer,
                args.device,
                args.max_new_tokens
            )
            all_results.extend(results)

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
