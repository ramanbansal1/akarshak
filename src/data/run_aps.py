"""
Gemma-APS Runner for Kaggle

This script provides command-line interface to:
1. Download RAGTruth data
2. Process data for Gemma-APS
3. Run Gemma-APS inference
4. Save extracted propositions

Usage:
    python run_aps.py download
    python run_aps.py process --sample_size 100
    python run_aps.py run --model google/gemma-2b-aps-it --max_examples 10
    python run_aps.py full --sample_size 1000 --max_examples 100
"""

import json
import argparse
import requests
from pathlib import Path
from typing import List, Dict
import sys


def download_ragtruth(data_dir: str = "./data/ragtruth"):
    """Download RAGTruth dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading RAGTruth dataset...")

    response_url = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
    source_url = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"

    # Download response.jsonl
    response_path = data_dir / "response.jsonl"
    print(f"Downloading {response_url}...")
    response = requests.get(response_url)
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"✓ Saved to {response_path}")

    # Download source_info.jsonl
    source_path = data_dir / "source_info.jsonl"
    print(f"Downloading {source_url}...")
    response = requests.get(source_url)
    with open(source_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"✓ Saved to {source_path}")

    print("\n✓ Download completed!")


def process_ragtruth(
    data_dir: str = "./data/ragtruth",
    output_dir: str = "./data/processed",
    sample_size: int = None,
    split_ratio: float = 0.8
):
    """Process RAGTruth data for Gemma-APS."""
    import random
    from collections import Counter

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response_path = data_dir / "response.jsonl"

    if not response_path.exists():
        print("RAGTruth data not found. Run 'download' first.")
        return

    print(f"Loading data from {response_path}...")

    # Load data
    examples = []
    with open(response_path, 'r', encoding='utf-8') as f:
        for line in f:
            if sample_size and len(examples) >= sample_size:
                break
            data = json.loads(line.strip())
            examples.append(data)

    print(f"✓ Loaded {len(examples)} examples")

    # Shuffle
    random.shuffle(examples)

    # Split
    split_idx = int(len(examples) * split_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Format
    train_data = []
    for ex in train_examples:
        train_data.append({
            "input": ex['response'],
            "id": ex['id'],
            "model": ex['model'],
            "split": ex['split'],
            "quality": ex['quality'],
            "labels": ex['labels']
        })

    val_data = []
    for ex in val_examples:
        val_data.append({
            "input": ex['response'],
            "id": ex['id'],
            "model": ex['model'],
            "split": ex['split'],
            "quality": ex['quality'],
            "labels": ex['labels']
        })

    # Save
    train_path = output_dir / "ragtruth_train.jsonl"
    val_path = output_dir / "ragtruth_val.jsonl"

    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Saved {len(train_data)} training examples to {train_path}")
    print(f"✓ Saved {len(val_data)} validation examples to {val_path}")

    # Print stats
    print(f"\nStatistics:")
    print(f"  Model distribution: {Counter([ex['model'] for ex in examples]).most_common(5)}")
    print(f"  Split distribution: {Counter([ex['split'] for ex in examples])}")
    print(f"  Quality distribution: {Counter([ex['quality'] for ex in examples])}")


def run_gemaps(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-2b-aps-it",
    max_examples: int = None,
    device: str = "cuda"
):
    """Run Gemma-APS inference."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("Error: torch and transformers required.")
        print("Install with: pip install torch transformers")
        sys.exit(1)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    print("✓ Model loaded")

    # Load data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if max_examples:
        data = data[:max_examples]

    print(f"Processing {len(data)} examples...")

    # Process
    results = []
    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Processing example {item['id']}...")

        # Extract propositions
        text = item['input']
        prompt = f"Extract atomic propositions from the following text:\n\n{text}\n\nPropositions:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        propositions = generated_text[len(prompt):].strip()

        result = {
            "id": item['id'],
            "original_response": text,
            "extracted_propositions": propositions,
            "model": item.get('model', ''),
            "labels": item.get('labels', [])
        }
        results.append(result)

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Saved results to {output_path}")

    # Print sample
    print(f"\nSample result:")
    if results:
        print(f"  ID: {results[0]['id']}")
        print(f"  Original: {results[0]['original_response'][:100]}...")
        print(f"  Extracted: {results[0]['extracted_propositions']}")


def full_pipeline(
    sample_size: int = 100,
    max_examples: int = 10,
    model_name: str = "google/gemma-2b-aps-it",
    device: str = "cuda"
):
    """Run full pipeline: download, process, run Gemma-APS."""
    print("="*60)
    print("FULL PIPELINE")
    print("="*60)

    # Download
    print("\n[1/3] Downloading RAGTruth...")
    download_ragtruth()

    # Process
    print("\n[2/3] Processing data...")
    process_ragtruth(sample_size=sample_size)

    # Run Gemma-APS
    print("\n[3/3] Running Gemma-APS...")
    run_gemaps(
        input_path="./data/processed/ragtruth_train.jsonl",
        output_path="./data/processed/ragtruth_train_output.jsonl",
        model_name=model_name,
        max_examples=max_examples,
        device=device
    )

    print("\n" + "="*60)
    print("✓ Pipeline completed!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Gemma-APS Runner for Kaggle")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download RAGTruth dataset')
    download_parser.add_argument('--data_dir', type=str, default="./data/ragtruth")

    # Process command
    process_parser = subparsers.add_parser('process', help='Process RAGTruth data')
    process_parser.add_argument('--data_dir', type=str, default="./data/ragtruth")
    process_parser.add_argument('--output_dir', type=str, default="./data/processed")
    process_parser.add_argument('--sample_size', type=int, default=None)
    process_parser.add_argument('--split_ratio', type=float, default=0.8)

    # Run command
    run_parser = subparsers.add_parser('run', help='Run Gemma-APS')
    run_parser.add_argument('--input', type=str, default="./data/processed/ragtruth_train.jsonl")
    run_parser.add_argument('--output', type=str, default="./data/processed/ragtruth_train_output.jsonl")
    run_parser.add_argument('--model', type=str, default="google/gemma-2b-aps-it")
    run_parser.add_argument('--max_examples', type=int, default=10)
    run_parser.add_argument('--device', type=str, default="cuda")

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--sample_size', type=int, default=100)
    full_parser.add_argument('--max_examples', type=int, default=10)
    full_parser.add_argument('--model', type=str, default="google/gemma-2b-aps-it")
    full_parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    if args.command == 'download':
        download_ragtruth(args.data_dir)
    elif args.command == 'process':
        process_ragtruth(args.data_dir, args.output_dir, args.sample_size, args.split_ratio)
    elif args.command == 'run':
        run_gemaps(args.input, args.output, args.model, args.max_examples, args.device)
    elif args.command == 'full':
        full_pipeline(args.sample_size, args.max_examples, args.model, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
