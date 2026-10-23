# PLAN: Compact Atomic Claim Extractor

## Problem Statement

VeriScore's claim extractor has significant overhead:
- **Per-sentence API calls**: 10 sentences = 10 API calls (expensive)
- **Massive prompts**: 100+ line few-shot prompts (thousands of tokens)
- **Context redundancy**: Same context sent repeatedly
- **No dependency tracking**: Claims extracted in isolation
- **Unstructured output**: Plain text without metadata

## Solution Overview

Build a compact, dependency-aware claim extractor that:
- Processes entire response in **ONE batch API call** (90% cost reduction)
- Uses **compact structured prompt** (~20 lines vs 100+)
- Extracts **claim dependency graph** (links related claims)
- Returns **structured JSON** with entities, relations, dates, quantities
- Minimal dependencies (no spaCy, no external models)

## Architecture

### Core Components

1. **CompactClaimExtractor**
   - Single batch extraction for entire text
   - Compact JSON-structured prompt
   - Dependency tracking between claims

2. **Model Client Interface**
   - Abstract interface for LLM API calls
   - Support for OpenAI, Anthropic, or custom clients
   - Token counting for cost tracking

3. **Claim Data Structure**
   - `Claim` class with: text, entities, relations, quantities, dates, depends_on
   - `DependencyGraph` for claim relationships

4. **Output Formats**
   - JSON output for integration
   - Pretty-printed for debugging
   - Dependency graph visualization

## VeriScore Analysis

### Claim Extractor Issues
- **Per-sentence API calls**: 1 call per sentence (expensive)
- **Massive prompts**: 100+ line few-shot templates
- **No dependency tracking**: Claims extracted in isolation
- **Unstructured output**: Plain text lists, no metadata
- **spaCy dependency**: Requires NLP library

### Claim Verifier Issues
- **Few-shot examples**: Loads from demos, adds overhead
- **Per-claim verification**: 1 API call per claim
- **Simple parsing**: String-based result extraction (fragile)
- **No confidence scores**: Binary/trinary only

### Evidence Retrieval
- **Serper API**: Requires external API key
- **Per-claim search**: One search per claim
- **No reranking**: Takes top N results only

## Updated Implementation Plan (Unified Multi-Head Model)

### Phase 1: Data Preparation & Dataset Download
- [ ] Download RAGTruth: response.jsonl + source_info.jsonl
- [ ] Download FEVER: training/dev/test sets
- [ ] Normalize FEVER labels to our format
- [ ] Create training data loaders for RAGTruth and FEVER
- [ ] Construct challenge set for partial support/conflicting evidence (~100-200 examples)

### Phase 2: Base Model Setup
- [ ] Download Phi-3-mini-4k-instruct (3.8B)
- [ ] Set up HuggingFace Transformers integration
- [ ] Configure LoRA adapters for parameter-efficient fine-tuning
- [ ] Implement base model wrapper with multi-head support

### Phase 3: Head 1 - Claim Extraction (Gemma-APS style)
- [ ] Design claim extraction head architecture
- [ ] Implement structured output (entities, relations, quantities, dates, dependencies)
- [ ] Create training data from RAGTruth responses
- [ ] SFT on claim extraction task
- [ ] Validate extraction quality against RAGTruth hallucination spans

### Phase 4: Head 2 - Multi-Evidence Entailment Network
- [ ] Implement last hidden state extraction for claims and evidence
- [ ] Implement weighted cosine similarity computation
- [ ] Design block diagonal classification head
- [ ] Create claim-evidence pairs from FEVER
- [ ] SFT on entailment classification (entailment/contradiction/neutral)
- [ ] Validate on FEVER dev set

### Phase 5: Head 3 - Conflict & Coverage Aggregator
- [ ] Design aggregator architecture (MLP or attention-based)
- [ ] Implement multi-passage signal combination
- [ ] Create coverage and conflict labels from challenge set
- [ ] Train aggregator with supervised labels
- [ ] Validate on partial support examples

### Phase 6: Head 4 - Decision Controller
- [ ] Design decision classification head
- [ ] Implement 5-way classification (accept/reject/qualify/retrieve/abstain)
- [ ] Create decision labels from challenge set
- [ ] Train controller with supervised labels
- [ - Implement calibrated thresholds

### Phase 7: GRPO Joint Fine-Tuning
- [ ] Implement GRPO trainer using HuggingFace TRL
- [ ] Design reward functions:
  - `reward_extraction`: F1 score vs ground truth claims
  - `reward_verification`: Accuracy vs FEVER labels
  - `reward_dependency`: Dependency graph correctness
- [ ] Implement group sampling (G=4-8)
- [ ] Joint training of all 4 heads
- [ ] Monitor reward curves and convergence

### Phase 8: Evaluation & Benchmarking
- [ ] Evaluate on FEVER test set (verification accuracy)
- [ ] Evaluate on RAGTruth test set (claim extraction + verification)
- [ ] Compare against VeriScore (cost, accuracy, performance)
- [ ] Ablation studies:
  - Entailment only
  - Entailment + contradiction
  - Entailment + contradiction + coverage
  - Full model with all components
- [ ] Calibration metrics (ECE, Brier score)
- [ ] Latency and memory benchmarks

### Phase 9: CLI & Tools
- [ ] Command-line interface for inference
- [ ] Batch processing support
- [ ] Output to JSONL format (compatible with VeriScore)
- [ ] Visualization tools for dependency graphs
- [ ] Evaluation comparison tool

### Phase 10: Documentation & Release
- [ ] Training scripts and configs
- [ ] Inference examples
- [ ] Model card documentation
- [ ] Release trained model checkpoints

## Key Design Decisions

### Prompt Strategy
- **Compact prompt**: ~20 lines with clear JSON schema
- **No few-shot examples**: Rely on schema and clear instructions
- **Structured output**: Force JSON format for parsing

### Dependency Tracking
- Claims reference each other by ID
- Example: "Engineer A designed the bridge" depends on claim 0 "The bridge was built in 1985"
- Enables verification in dependency order

### Batch vs Sentence-Level
- VeriScore: 1 API call per sentence
- Our approach: 1 API call for entire response
- Tradeoff: Larger context window needed, but 90% cost reduction

### Model Support
- **Primary**: Local models via Ollama (free, lower accuracy)
- **Model choice**: User will specify later (configurable)
- Fallback: Fail loudly with error on invalid JSON
- Caching: User will specify requirements later

### Evaluation
- Full evaluation suite comparing against VeriScore
- Metrics: accuracy, cost, performance, claim quality
- Output: JSONL format compatible with VeriScore pipeline
- Support both direct comparison and offline comparison

## Unified Multi-Head Model Architecture

### Research Findings: Single Model for Multiple Tasks

**Yes, a single model CAN perform all 4 tasks!** Literature confirms this is feasible:

1. **"One LLM to Train Them All" (2025)**: Single Qwen3-4B model performs claim detection, evidence ranking, and stance detection jointly using multi-task learning with classification heads. Achieved 44%, 54%, 31% relative gains over baselines.

2. **"Simple Compounded-Label Training" (ACL 2020)**: Single sequence matching model trained with compounded labels for both sentence selection and claim verification. Outperformed typical multi-task learning with simpler setup.

3. **"MultiVerS" (2021)**: Multitask verification model predicts fact-checking label and identifies rationales in a multitask fashion based on shared encoding.

### Model Selection (Under 5B Parameters)

| Model | Parameters | Pros | Cons |
|-------|-----------|------|------|
| **Phi-3-mini** | 3.8B | Rivals GPT-3.5, runs on phone, excellent reasoning | Requires PyTorch |
| **Granite-4.0-Micro** | 3B | Apache 2.0, hybrid Mamba-2/transformer, 70% lower memory | Newer architecture |
| **Granite-4.2-3B** | 4B | Traditional transformer, well-supported | Slightly larger |
| **Gemma-2B-APS** | 2B | Pre-trained for proposition segmentation! | Smaller capacity |

**Recommendation**: **Phi-3-mini (3.8B)** - best performance under 5B, rivals much larger models, strong reasoning capabilities essential for claim extraction and verification.

### Gemma-APS Inspiration

**Gemma-APS (Abstractive Proposition Segmentation)**:
- Breaks text into atomic facts, statements, and ideas
- Restates them in full sentences with minimal changes
- Exactly what we need for atomic claim extraction!
- Available in 2B and 7B variants
- Trained using distillation from Gemini Pro/Ultra on multi-domain synthetic data

**Our approach**: Use Gemma-APS architecture as the claim extraction head, integrated into a unified multi-head model.

### GRPO Training Method

**Group Relative Policy Optimization (GRPO)**:
- Eliminates need for critic network (unlike PPO)
- Uses Monte-Carlo samples ("group") to estimate advantage
- Normalizes rewards within group: `A = (r - mean(r)) / std(r)`
- Used in DeepSeek R1 reasoning models
- Works well with verifiable rewards (perfect for fact-checking)
- More memory-efficient than PPO

**Training approach**:
1. Supervised Fine-Tuning (SFT) on FEVER/RAGTruth for initial alignment
2. GRPO fine-tuning with verifiable rewards:
   - Reward 1: Claim extraction quality (vs ground truth)
   - Reward 2: Verification accuracy (vs FEVER labels)
   - Reward 3: Dependency correctness (vs annotated dependencies)

### Proposed Unified Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified Base Model (Phi-3-mini 3.8B)          │
│                  Shared Transformer Encoder/Decoder          │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Head 1:        │  │  Head 2:        │  │  Head 3:        │
│  Claim          │  │  Multi-Evidence │  │  Conflict &    │
│  Extraction     │  │  Entailment     │  │  Coverage       │
│  (Gemma-APS)    │  │  Network        │  │  Aggregator     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  - Entities     │  │  - Entailment   │  │  - Support      │
│  - Relations    │  │  - Contradiction│  │  - Contradiction│
│  - Quantities   │  │  - Neutral      │  │  - Coverage     │
│  - Dates        │  │  - Confidence   │  │  - Uncertainty  │
│  - Dependencies │  │  - Evidence IDs │  │  - Source agree │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Head 4:        │
                    │  Decision       │
                    │  Controller     │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  - Accept      │
                    │  - Reject      │
                    │  - Qualify     │
                    │  - Retrieve    │
                    │  - Abstain     │
                    └─────────────────┘
```

### Detailed Multi-Evidence Entailment Network (User's Specification)

**Architecture**:
1. **Input**: Claim + Retrieved Evidence (multiple passages)
2. **Process**:
   - Pass claim through base model → get last hidden state `h_claim`
   - Pass each evidence passage through base model → get last hidden states `[h_e1, h_e2, ..., h_ek]`
3. **Similarity Computation**:
   - Compute weighted cosine similarity between `h_claim` and each `h_ei`
   - Weights learned during training (attention mechanism)
4. **Classification**:
   - Block diagonal classification head
   - Separate classification paths for each evidence passage
   - Aggregate via learned weights
5. **Output**:
   - Per-passage predictions: entailment, contradiction, neutral
   - Aggregated prediction with confidence

**Mathematical Formulation**:
```
h_claim = Transformer(claim)[:, -1, :]  # Last token hidden state
h_evidence = [Transformer(e_i)[:, -1, :] for e_i in evidence]

# Weighted cosine similarity
sim_i = (w_i * cosine_similarity(h_claim, h_ei))

# Block diagonal classification
logits = BlockDiag([Classifier_i(h_claim, h_ei) for i in range(k)])

# Softmax over categories
prob = softmax(logits / temperature)
```

### Training Strategy

#### Phase 1: SFT on Claim Extraction
- **Dataset**: RAGTruth responses + FEVER claims
- **Task**: Extract atomic claims with metadata
- **Loss**: Cross-entropy on claim tokens + auxiliary losses for entities/relations
- **Head**: Head 1 (Claim Extraction - Gemma-APS style)

#### Phase 2: SFT on Entailment
- **Dataset**: FEVER (claim-evidence pairs)
- **Task**: Classify entailment/contradiction/neutral
- **Loss**: Cross-entropy on classification
- **Head**: Head 2 (Multi-Evidence Entailment)

#### Phase 3: GRPO Fine-Tuning (Joint Training)
- **Dataset**: RAGTruth + FEVER
- **Reward Functions**:
  ```
  reward_extraction = F1_score(extracted_claims, ground_truth)
  reward_verification = accuracy(verification_label, ground_truth)
  reward_dependency = correctness(dependency_graph)
  total_reward = α*reward_extraction + β*reward_verification + γ*reward_dependency
  ```
- **Process**:
  1. Sample G completions per prompt
  2. Compute rewards for each completion
  3. Normalize rewards within group
  4. Update policy toward higher-reward behaviors
- **All heads trained jointly**

#### Phase 4: Aggregator & Controller Training
- **Dataset**: Constructed challenge set (partial support, conflicting evidence)
- **Task**: Learn to aggregate multi-evidence signals and make decisions
- **Loss**: Cross-entropy on final decision
- **Heads**: Head 3 (Aggregator) + Head 4 (Controller)

### Dataset Training Plan

| Phase | Dataset | Task | Head |
|-------|---------|------|------|
| 1 | RAGTruth response.jsonl | Claim extraction | Head 1 |
| 1 | FEVER claims | Claim extraction (supervised) | Head 1 |
| 2 | FEVER (claim+evidence) | Entailment classification | Head 2 |
| 3 | RAGTruth + FEVER | Joint GRPO fine-tuning | All heads |
| 4 | Challenge set | Aggregation + decisions | Head 3+4 |

### Technical Implementation Details

**Base Model**: Phi-3-mini-4k-instruct (3.8B parameters)
- Context window: 4K tokens
- Architecture: Dense decoder-only transformer
- Supports: LoRA adapters for efficient fine-tuning

**Multi-Head Setup**:
- Shared base model frozen during initial head training
- LoRA adapters per head (parameter-efficient)
- Gradual unfreezing during joint training

**GRPO Implementation**:
- Use HuggingFace TRL library `GRPOTrainer`
- Custom reward functions for each task
- Group size G = 4-8 (balance quality vs compute)
- KL penalty to prevent drift from reference policy

**Hardware Requirements**:
- Single A100 (40GB) or RTX 4090 (24GB) sufficient
- LoRA reduces memory significantly
- Gradient checkpointing for longer sequences

## Dataset Analysis: FEVER & RAGTruth

### FEVER Dataset Format
- **Structure**: JSONL with 185,445 claims
- **Labels**: SUPPORTS, REFUTES, NOT ENOUGH INFO
- **Fields**:
  - `id`: Claim ID
  - `label`: Annotated label
  - `claim`: Text of the claim
  - `evidence`: List of evidence sets with Wikipedia URLs and sentence IDs
- **Use case**: Fact verification against Wikipedia
- **Strengths**: Already has claim-level annotations and evidence
- **Limitations**: Claims are pre-extracted (not from RAG responses)

### RAGTruth Dataset Format
- **Structure**: Two JSONL files (response.jsonl, source_info.jsonl)
- **Size**: ~18,000 RAG-generated responses
- **Fields (response.jsonl)**:
  - `id`: Response index
  - `source_id`: Source information index
  - `model`: Model that generated response
  - `temperature`: Generation temperature
  - `labels`: Hallucination spans with start/end positions, label_type, meta
  - `split`: train/test
  - `quality`: Quality flags (incorrect_refusal, truncated)
  - `response`: LLM response text
- **Fields (source_info.jsonl)**:
  - `source_id`: Source index
  - `task_type`: QA, Data2txt, Summary
  - `source`: Original content source (MARCO, Yelp, etc.)
  - `source_info`: Base content (passages for QA, structured data for Data2txt)
  - `prompt`: Prompt used for generation
- **Use case**: Word-level hallucination detection in RAG
- **Strengths**: Real RAG responses, word-level annotations, multiple models
- **Limitations**: Need to extract atomic claims from responses first

### How to Use These Datasets for GOAL.md

#### For Claim Extraction (Phase 1)
1. **RAGTruth response.jsonl**: Use responses as input to our compact claim extractor
2. **Compare with RAGTruth labels**: Map our extracted claims to hallucination spans
3. **FEVER**: Use as additional training data for claim verification (has pre-extracted claims)

#### For Claim Verification (Phase 2-3)
1. **FEVER**: Primary dataset for training entailment/contradiction classifier
   - Already has claim-evidence pairs
   - Labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
   - Use to train DeBERTa-v3 or RoBERTa-MNLI
2. **RAGTruth**: Use source_info.jsonl passages as evidence for response claims
   - Simulate RAG verification scenario
   - Test on real RAG responses with ground truth hallucination labels

#### For Multi-Evidence Aggregation (Phase 4)
1. **FEVER**: 31.75% of claims require multiple sentences, 12.15% from multiple pages
   - Perfect for testing multi-passage evidence aggregation
2. **RAGTruth**: Multiple passages per question in QA tasks
   - Test conflict detection when passages disagree

#### For Evaluation Metrics
1. **Claim-level accuracy**: Compare extracted claims vs RAGTruth hallucination spans
2. **Verification accuracy**: Use FEVER labels as ground truth
3. **Partial support detection**: Create challenge set from FEVER multi-sentence examples
4. **Conflict detection**: Use RAGTruth cases where model hallucinates despite correct context

### Dataset Integration Strategy

#### Step 1: Data Preparation
- Download RAGTruth: response.jsonl + source_info.jsonl
- Download FEVER: training/dev/test sets
- Normalize FEVER labels to our format (supported/contradicted/insufficient)
- Extract claims from RAGTruth responses using our extractor

#### Step 2: Train-Val-Test Splits
- FEVER: Use official splits (already provided)
- RAGTruth: Use train/test splits provided
- Ensure no leakage between related claims or source documents

#### Step 3: Label Mapping
```
FEVER → TRACE-RAG:
- SUPPORTS → supported
- REFUTES → contradicted
- NOT ENOUGH INFO → insufficient

RAGTruth → TRACE-RAG:
- Evident Baseless Info → contradicted
- Implicit True → supported (if info in context)
- Hallucination spans → contradicted
- Non-hallucinated text → supported/insufficient
```

#### Step 4: Challenge Set Construction
- Select FEVER examples requiring multi-sentence evidence
- Select RAGTruth examples with partial support
- Annotate coverage/conflict labels where needed
- ~100-200 examples for testing novelty

## Open Questions

### Decided
- ✅ LLM Provider: Local model (Ollama)
- ✅ Fallback: Fail loudly with error
- ✅ Evaluation: Full comparison suite
- ✅ Caching: User will specify later
- ✅ Model Framework: Ollama only
- ✅ Default Model: User will specify later
- ✅ Comparison Format: Full evaluation suite
- ✅ Datasets: FEVER + RAGTruth

### Pending
- ⏳ Which specific Ollama model to use?
- ⏳ Caching implementation details (in-memory vs disk)?
- ⏳ Specific evaluation metrics to prioritize?
