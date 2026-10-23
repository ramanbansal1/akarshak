"""
RAGTruth Data Pipeline for Gemma-APS Training

This script:
1. Downloads/loads RAGTruth dataset
2. Uses RAGTruth responses as input for Gemma-APS
3. Gemma-APS will extract atomic propositions from responses
4. Prepares (response → extracted claims) pairs for training our model
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter
import requests


@dataclass
class RAGTruthExample:
    """Structured representation of a RAGTruth example."""
    id: str
    source_id: str
    model: str
    temperature: float
    labels: List[Dict]
    split: str
    quality: str
    response: str
    source_info: Dict = None


class RAGTruthPipeline:
    """Pipeline for processing RAGTruth dataset for Gemma-APS training."""

    def __init__(self, data_dir: str = "./data/ragtruth", output_dir: str = "./data/processed"):
        """
        Initialize the pipeline.

        Args:
            data_dir: Directory to store RAGTruth data
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.examples: List[RAGTruthExample] = []
        self.stats = {}

    def download_ragtruth(self):
        """Download RAGTruth dataset from GitHub."""
        print("Downloading RAGTruth dataset...")

        # RAGTruth GitHub URLs
        response_url = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
        source_url = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"

        # Download response.jsonl
        response_path = self.data_dir / "response.jsonl"
        print(f"Downloading {response_url}...")
        response = requests.get(response_url)
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved to {response_path}")

        # Download source_info.jsonl
        source_path = self.data_dir / "source_info.jsonl"
        print(f"Downloading {source_url}...")
        response = requests.get(source_url)
        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved to {source_path}")

        print("Download completed!")

    def load_data(self, sample_size: int = None) -> List[RAGTruthExample]:
        """
        Load RAGTruth data from JSONL file.

        Args:
            sample_size: If provided, only load this many examples (for testing)

        Returns:
            List of RAGTruthExample objects
        """
        response_path = self.data_dir / "response.jsonl"

        if not response_path.exists():
            print("RAGTruth data not found. Downloading...")
            self.download_ragtruth()

        examples = []
        with open(response_path, 'r', encoding='utf-8') as f:
            for line in f:
                if sample_size and len(examples) >= sample_size:
                    break
                data = json.loads(line.strip())
                example = RAGTruthExample(
                    id=data['id'],
                    source_id=data['source_id'],
                    model=data['model'],
                    temperature=data['temperature'],
                    labels=data['labels'],
                    split=data['split'],
                    quality=data['quality'],
                    response=data['response']
                )
                examples.append(example)

        self.examples = examples
        print(f"Loaded {len(examples)} examples from {response_path}")
        return examples

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data and compute statistics.

        Returns:
            Dictionary with validation statistics
        """
        if not self.examples:
            raise ValueError("No data loaded. Call load_data() first.")

        stats = {
            'total_examples': len(self.examples),
            'split_distribution': Counter(),
            'model_distribution': Counter(),
            'quality_distribution': Counter(),
            'response_length_stats': [],
            'label_stats': {
                'with_labels': 0,
                'without_labels': 0,
                'total_labels': 0
            },
            'invalid_responses': []
        }

        for ex in self.examples:
            # Split distribution
            stats['split_distribution'][ex.split] += 1
            stats['model_distribution'][ex.model] += 1
            stats['quality_distribution'][ex.quality] += 1

            # Response length
            response_len = len(ex.response.split())
            stats['response_length_stats'].append(response_len)

            # Label stats
            if ex.labels:
                stats['label_stats']['with_labels'] += 1
                stats['label_stats']['total_labels'] += len(ex.labels)
            else:
                stats['label_stats']['without_labels'] += 1

            # Invalid responses (empty or too short)
            if not ex.response or len(ex.response.strip()) < 20:
                stats['invalid_responses'].append(ex.id)

        # Compute averages
        if stats['response_length_stats']:
            stats['avg_response_length'] = sum(stats['response_length_stats']) / len(stats['response_length_stats'])
            stats['min_response_length'] = min(stats['response_length_stats'])
            stats['max_response_length'] = max(stats['response_length_stats'])

        self.stats = stats
        return stats

    def clean_data(self) -> List[RAGTruthExample]:
        """
        Clean the data by filtering invalid examples.

        Returns:
            List of cleaned RAGTruthExample objects
        """
        if not self.stats:
            self.validate_data()

        cleaned = []
        removed = 0

        for ex in self.examples:
            # Remove invalid responses
            if ex.id in self.stats['invalid_responses']:
                removed += 1
                continue

            # Remove if quality is not "good" (optional)
            # if ex.quality != "good":
            #     removed += 1
            #     continue

            cleaned.append(ex)

        print(f"Cleaned data: removed {removed} invalid examples")
        print(f"Remaining: {len(cleaned)} examples")
        self.examples = cleaned
        return cleaned

    def sample_examples(self, n: int = 5) -> List[RAGTruthExample]:
        """
        Sample random examples for inspection.

        Args:
            n: Number of examples to sample

        Returns:
            List of sampled RAGTruthExample objects
        """
        if not self.examples:
            raise ValueError("No data loaded. Call load_data() first.")

        sampled = random.sample(self.examples, min(n, len(self.examples)))
        return sampled

    def prepare_for_gemaps(self, split_ratio: float = 0.8) -> Dict[str, List[Dict]]:
        """
        Prepare data for Gemma-APS training.

        Format:
        {
            "input": RAG response (longer text with multiple facts),
            "output": extracted atomic propositions (to be generated by Gemma-APS)
        }

        Args:
            split_ratio: Train/validation split ratio

        Returns:
            Dictionary with 'train' and 'validation' splits
        """
        if not self.examples:
            raise ValueError("No data loaded. Call load_data() first.")

        # Shuffle data
        random.shuffle(self.examples)

        # Split
        split_idx = int(len(self.examples) * split_ratio)
        train_examples = self.examples[:split_idx]
        val_examples = self.examples[split_idx:]

        # Format for Gemma-APS
        # Input: The RAG response (contains multiple facts)
        # Output: Will be generated by Gemma-APS (atomic propositions)
        train_data = []
        for ex in train_examples:
            train_data.append({
                "input": ex.response,  # The RAG response
                "id": ex.id,
                "model": ex.model,
                "split": ex.split,
                "quality": ex.quality,
                "labels": ex.labels  # Ground truth hallucination labels
            })

        val_data = []
        for ex in val_examples:
            val_data.append({
                "input": ex.response,
                "id": ex.id,
                "model": ex.model,
                "split": ex.split,
                "quality": ex.quality,
                "labels": ex.labels
            })

        output = {
            'train': train_data,
            'validation': val_data
        }

        # Save to files
        train_path = self.output_dir / "ragtruth_train.jsonl"
        val_path = self.output_dir / "ragtruth_val.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')

        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')

        print(f"Saved {len(train_data)} training examples to {train_path}")
        print(f"Saved {len(val_data)} validation examples to {val_path}")

        return output

    def print_statistics(self):
        """Print validation statistics."""
        if not self.stats:
            self.validate_data()

        print("\n" + "="*60)
        print("RAGTRUTH DATASET STATISTICS")
        print("="*60)
        print(f"Total examples: {self.stats['total_examples']}")
        print(f"\nSplit distribution:")
        for split, count in self.stats['split_distribution'].most_common():
            print(f"  {split}: {count} ({count/self.stats['total_examples']*100:.1f}%)")
        print(f"\nModel distribution (top 5):")
        for model, count in self.stats['model_distribution'].most_common(5):
            print(f"  {model}: {count} ({count/self.stats['total_examples']*100:.1f}%)")
        print(f"\nQuality distribution:")
        for quality, count in self.stats['quality_distribution'].most_common():
            print(f"  {quality}: {count} ({count/self.stats['total_examples']*100:.1f}%)")
        print(f"\nResponse length stats:")
        print(f"  Average: {self.stats['avg_response_length']:.1f} words")
        print(f"  Min: {self.stats['min_response_length']} words")
        print(f"  Max: {self.stats['max_response_length']} words")
        print(f"\nLabel stats:")
        print(f"  With labels: {self.stats['label_stats']['with_labels']}")
        print(f"  Without labels: {self.stats['label_stats']['without_labels']}")
        print(f"  Total labels: {self.stats['label_stats']['total_labels']}")
        print(f"\nInvalid responses: {len(self.stats['invalid_responses'])}")
        print("="*60 + "\n")

    def print_sample(self, n: int = 2):
        """Print sample examples."""
        samples = self.sample_examples(n)

        print("\n" + "="*60)
        print(f"SAMPLE EXAMPLES (n={n})")
        print("="*60)
        for i, ex in enumerate(samples, 1):
            print(f"\nExample {i}:")
            print(f"  ID: {ex.id}")
            print(f"  Model: {ex.model}")
            print(f"  Split: {ex.split}")
            print(f"  Quality: {ex.quality}")
            print(f"  Response (truncated): {ex.response[:200]}...")
            if ex.labels:
                print(f"  Labels: {len(ex.labels)} hallucination spans")
            else:
                print(f"  Labels: None")
        print("="*60 + "\n")


def main():
    """Main function to run the pipeline."""
    # Configuration
    DATA_DIR = "/home/raman/Desktop/akarshak/src/data/ragtruth"
    OUTPUT_DIR = "/home/raman/Desktop/akarshak/src/data/processed"
    SAMPLE_SIZE = 100  # Load 100 examples for testing

    # Initialize pipeline
    pipeline = RAGTruthPipeline(DATA_DIR, OUTPUT_DIR)

    # Download and load data
    print("Loading RAGTruth data...")
    pipeline.load_data(sample_size=SAMPLE_SIZE)

    # Validate and print statistics
    print("Validating data...")
    pipeline.validate_data()
    pipeline.print_statistics()

    # Print samples
    pipeline.print_sample(n=2)

    # Clean data
    print("Cleaning data...")
    pipeline.clean_data()

    # Prepare for Gemma-APS
    print("Preparing data for Gemma-APS...")
    pipeline.prepare_for_gemaps(split_ratio=0.8)

    print("\nPipeline completed successfully!")
    print(f"Processed data saved to: {OUTPUT_DIR}")
    print("\nNext step: Run Gemma-APS on this data to extract atomic propositions")


if __name__ == "__main__":
    main()
