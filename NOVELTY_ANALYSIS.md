# Critical Analysis: Our Novelty Over "One LLM to Train Them All"

## Their Approach (Baseline)

**Paper**: "One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking" (2025)

- **3 Tasks**: Claim detection (is this check-worthy?), Evidence re-ranking, Stance detection (supports/refutes)
- **Model**: Qwen3-4B (4B parameters)
- **Training**: Standard supervised fine-tuning with QLoRA
- **Heads**: Classification (CLS), Causal LM (CLM), Instruction Tuning (IT)
- **Datasets**: CheckThat! 2024, AVeriTeC, general fact-checker data
- **Output**: Binary/trinary classification
- **Method**: Multi-task learning with loss weighting, task ordering, model scaling analysis

## Our Novel Improvements (Beyond Their Work)

| Aspect | Their Work | Our Novelty | Impact |
|--------|-----------|-------------|---------|
| **Task 1** | Claim detection (binary: check-worthy?) | **Atomic claim extraction** with entities, relations, dates, quantities, dependencies | ✅ Much finer-grained |
| **Architecture** | Generic CLS/CLM heads | **Gemma-APS proposition segmentation** for claim extraction | ✅ Specialized for atomic claims |
| **Multi-Evidence** | Simple stance detection | **Block diagonal classification** with per-passage predictions + weighted cosine similarity | ✅ Explicit multi-passage reasoning |
| **Coverage** | Implicit in partial support | **Explicit coverage detection** (partial support analysis) | ✅ Novel contribution per GOAL.md |
| **Conflict** | Not addressed | **Conflict aggregation** from multiple sources | ✅ Novel contribution per GOAL.md |
| **Training** | Supervised fine-tuning only | **GRPO with verifiable rewards** (reinforcement learning) | ✅ Better for complex reasoning |
| **Decision** | Classification only | **Decision controller** (accept/reject/qualify/retrieve/abstain) | ✅ Actionable per GOAL.md |
| **Dependencies** | None | **Dependency graph tracking** between claims | ✅ Novel for verification ordering |
| **Datasets** | General fact-checking | **RAGTruth** (RAG-specific) + FEVER | ✅ RAG-focused evaluation |
| **Model** | Qwen3-4B (4B) | **Phi-3-mini (3.8B)** - smaller, better reasoning | ✅ More efficient |
| **Calibration** | Not mentioned | **Calibrated thresholds** for decisions | ✅ More reliable decisions |

## Detailed Novelty Breakdown

### 1. Atomic Claim Extraction (They Don't Have This)

**Their approach**: "Is this sentence check-worthy?" (binary classification)
- Input: Text
- Output: True/False (check-worthy or not)
- Simple binary decision

**Our approach**: "Extract all atomic claims with entities, relations, dates, quantities, and dependencies" (structured generation)
- Input: Text (RAG response)
- Output: List of claims with metadata:
  ```json
  {
    "claims": [
      {
        "text": "The bridge was constructed in 1985",
        "entities": ["bridge"],
        "relations": ["constructed"],
        "quantities": [],
        "dates": ["1985"],
        "depends_on": []
      },
      {
        "text": "Engineer A designed the bridge",
        "entities": ["Engineer A", "bridge"],
        "relations": ["designed"],
        "quantities": [],
        "dates": [],
        "depends_on": [0]
      }
    ]
  }
  ```

**Novelty**: Gemma-APS inspired proposition segmentation integrated into multi-head model
**Impact**: Enables fine-grained verification, not just binary check-worthiness

### 2. Multi-Evidence Entailment with Block Diagonal (Their Missing Feature)

**Their approach**: Single stance prediction (supports/refutes/partial)
- Input: Claim + single evidence passage
- Output: Support/Refute/Partial Support classification
- No per-passage analysis

**Our approach**: Per-passage entailment with block diagonal classification + weighted cosine similarity
- Input: Claim + [Evidence1, Evidence2, ..., EvidenceK]
- Process:
  1. `h_claim = Transformer(claim)[:, -1, :]` (last hidden state)
  2. `h_evidence = [Transformer(e_i)[:, -1, :] for e_i in evidence]`
  3. `sim_i = w_i * cosine_similarity(h_claim, h_ei)` (learned weights)
  4. `logits = BlockDiag([Classifier_i(h_claim, h_ei) for i in range(k)])`
  5. `prob = softmax(logits / temperature)`
- Output: Per-passage predictions + aggregated prediction

**Novelty**: Explicit multi-passage reasoning with learned similarity weights
**Impact**: Can handle conflicting evidence, partial support, source disagreement

### 3. Coverage and Conflict Aggregation (Novel Contribution)

**Their approach**: Implicit partial support in stance labels
- Has "Partially Supported" and "Partially Refuted" labels
- But no explicit modeling of coverage
- No conflict detection from multiple sources

**Our approach**: Explicit coverage detection + conflict aggregation from multiple sources
- Coverage V_i: extent to which all claim facets are established
- Conflict K_i: aggregate contradiction strength from multiple passages
- Source agreement: whether evidence sources agree or disagree
- Mathematical formulation:
  ```
  R_i = σ(fθ(S_i, K_i, V_i, U_i))
  ```
  where S_i = support, K_i = conflict, V_i = coverage, U_i = uncertainty

**Novelty**: Directly addresses GOAL.md's research gap on partial support and conflicting evidence
**Impact**: Better detection of partially supported claims (RQ1) and conflicting evidence (RQ2)

### 4. GRPO Training with Verifiable Rewards (Training Innovation)

**Their approach**: Standard supervised fine-tuning with cross-entropy loss
- Loss: `L_total = Σ λ_t * L_t` (weighted sum of task losses)
- Loss weights: equal (1:1:1) or tuned (1:4:2, 4:1:2)
- No reinforcement learning component

**Our approach**: GRPO with programmable rewards
- Reward functions:
  ```
  reward_extraction = F1_score(extracted_claims, ground_truth)
  reward_verification = accuracy(verification_label, ground_truth)
  reward_dependency = correctness(dependency_graph)
  total_reward = α*reward_extraction + β*reward_verification + γ*reward_dependency
  ```
- Process:
  1. Sample G completions per prompt
  2. Compute rewards for each completion
  3. Normalize rewards within group: `A = (r - mean(r)) / std(r)`
  4. Update policy toward higher-reward behaviors
- Eliminates critic network (simpler than PPO)
- Used in DeepSeek R1 reasoning models

**Novelty**: First to apply GRPO to multi-task fact-checking
**Impact**: Better optimization for complex, multi-step reasoning tasks

### 5. Decision Controller (Actionable Output)

**Their approach**: Classification output only
- Output: Support/Refute/Partial Support/Partial Refute
- No actionable recommendations
- No abstention mechanism

**Our approach**: 5-way decision with calibrated thresholds
- Output: Accept, Reject, Qualify, Retrieve more evidence, Abstain
- Calibrated thresholds for each decision
- Actionable per GOAL.md's actionability requirement

**Novelty**: Actionable recommendations
**Impact**: Enables adaptive retrieval and practical deployment

### 6. Dependency Graph (Structural Novelty)

**Their approach**: No claim dependencies
- Claims treated independently
- No structural relationships
- No verification ordering

**Our approach**: Dependency tracking between claims
- Example: "Engineer A designed the bridge" depends on claim 0 "The bridge was built in 1985"
- Enables verification in dependency order
- Better error propagation

**Novelty**: First to model claim dependencies in unified model
**Impact**: Enables verification in dependency order, better error propagation

### 7. RAG-Specific Datasets (Domain Novelty)

**Their approach**: General fact-checking
- CheckThat! 2024: political debates
- AVeriTeC: general web claims
- General fact-checker data

**Our approach**: RAGTruth (RAG-specific) + FEVER
- RAGTruth: 18,000 RAG-generated responses with word-level hallucination annotations
- FEVER: 185,445 fact-verification claims with evidence
- RAG-focused evaluation

**Novelty**: First multi-head model trained on RAG-specific data
**Impact**: Better performance on RAG systems (the target use case)

### 8. Model Selection (Efficiency)

**Their approach**: Qwen3-4B (4B parameters)
- 4B parameters
- Good performance
- Not the most efficient

**Our approach**: Phi-3-mini (3.8B)
- 3.8B parameters (smaller)
- Rivals GPT-3.5 performance (69% MMLU)
- Runs on a phone
- Better reasoning capabilities

**Novelty**: More efficient model with better reasoning
**Impact**: Lower deployment cost, better performance

### 9. Calibration (Reliability)

**Their approach**: Not mentioned
- No calibration metrics
- No uncertainty estimation
- No ECE or Brier score

**Our approach**: Calibrated thresholds
- Expected Calibration Error (ECE)
- Brier score
- Risk-coverage curves
- Reliable confidence estimates

**Novelty**: Explicit calibration for reliable decisions
**Impact**: More reliable verification decisions

## Research Questions We Address (Beyond Their Work)

### Their RQs (We Address These Too)
- RQ1: How does multi-task learning compare with single-task and zero-/few-shot settings? ✅
- RQ2: Which configuration (CLS vs CLM vs IT) achieves best balance? ✅ (We add Gemma-APS)
- RQ3: How do task weighting, training order, and model scale affect optimization? ✅ (We add GRPO)

### Our Additional RQs (From GOAL.md - Novel)
- **RQ1**: Does explicit claim-facet coverage improve detection of partially supported claims? (Novel)
- **RQ2**: Does conflict-aware multi-passage aggregation improve detection when retrieved sources disagree? (Novel)
- **RQ3**: Does uncertainty-aware abstention reduce false verification decisions at a fixed coverage level? (Novel)
- **RQ4**: Does adaptive retrieval improve verification accuracy enough to justify its additional computational cost? (Novel)

## Summary: Our Key Innovations Over "One LLM to Train Them All"

1. **4 Heads vs 3 Tasks**: We add atomic claim extraction with dependency tracking
2. **Gemma-APS Integration**: Specialized architecture for proposition segmentation
3. **Block Diagonal Multi-Evidence**: Per-passage classification with learned similarity
4. **Explicit Coverage/Conflict**: Direct modeling of partial support and conflicts
5. **GRPO Training**: Reinforcement learning with verifiable rewards (not just SFT)
6. **Decision Controller**: Actionable 5-way classification
7. **Dependency Graphs**: Structural claim relationships
8. **RAG-Specific Data**: RAGTruth + FEVER for RAG-focused evaluation
9. **Phi-3-mini**: Smaller, better reasoning model
10. **Calibration**: Reliable confidence estimates

## Bottom Line

**Their work proves**: Multi-task learning works for fact-checking with 3 tasks (claim detection, evidence ranking, stance detection)

**Our work extends this to**:
- **Finer granularity** (atomic claims vs check-worthiness)
- **Multi-evidence reasoning** (block diagonal vs simple classification)
- **Coverage/conflict detection** (explicit vs implicit)
- **RAG-specific scenarios** (their work is general fact-checking)
- **Actionable decisions** (controller vs classification only)
- **Reinforcement learning** (GRPO vs SFT only)
- **Dependency modeling** (structural vs independent)
- **Calibration** (reliable vs uncalibrated)

**Novel contribution**: We're not just replicating their work - we're extending it to address the specific research gaps identified in GOAL.md around partial support, conflicting evidence, and uncertainty-aware decision making in RAG systems.
