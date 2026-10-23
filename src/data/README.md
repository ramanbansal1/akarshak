# RAGTruth + Gemma-APS Pipeline for Kaggle

## Quick Start (Kaggle)

```bash
# Install dependencies
pip install torch transformers requests

# Run full pipeline with 100 samples, process 10 with Gemma-APS
python run_aps.py full --sample_size 100 --max_examples 10

# Or run steps separately:
python run_aps.py download                    # Download RAGTruth
python run_aps.py process --sample_size 100   # Process data
python run_aps.py run --max_examples 10      # Run Gemma-APS
```

## Files

- `run_aps.py`: Main script with command-line interface
- `fever.py`: FEVER dataset loader (not used currently)
- `ragtruth_pipeline.py`: Original pipeline (legacy)
- `train.jsonl`: FEVER data (not used currently)
- `ragtruth/`: RAGTruth downloaded data
- `processed/`: Processed data for Gemma-APS

## Output Format

### Input (ragtruth_train.jsonl)
```json
{"input": "RAG response text...", "id": "123", "model": "gpt-4", "labels": [...]}
```

### Output (ragtruth_train_output.jsonl)
```json
{
  "id": "123",
  "original_response": "RAG response text...",
  "extracted_propositions": "Atomic proposition 1. Atomic proposition 2.",
  "model": "gpt-4",
  "labels": [...]
}
```

## For Kaggle

1. Upload this entire directory to Kaggle
2. Turn on GPU accelerator
3. Run: `python run_aps.py full --sample_size 1000 --max_examples 100`

## Next Steps

After Gemma-APS extracts propositions:
1. Use extracted propositions as training targets for Phi-3/Qwen
2. Train multi-head model with:
   - Head 1: Claim extraction (Gemma-APS style)
   - Head 2: Multi-evidence entailment
   - Head 3: Coverage/conflict aggregator
   - Head 4: Decision controller
