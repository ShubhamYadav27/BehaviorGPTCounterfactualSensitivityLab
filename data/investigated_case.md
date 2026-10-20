# In-Depth Case Investigation: case_004
**Archetype:** HOME_DECORATOR | **Mutation Type:** mutation_position_early | **Position:** Index 0 (early)

### 1. Sequence Intervention
- **History Length:** 6 actions
- **Single Changed Action:** Index 0

#### Input Behavioral History:
**Original History:**
  0. [viewItem] product: paB0CPSK1GVB <-- mutated position
  1. [addToCart] product: paB0CRLY5MYP
  2. [viewItem] product: paB0CPSK1GVB
  3. [addToCart] product: paB0CPSK1GVB
  4. [addToCart] product: paB0CPSK1GVB
  5. [viewItem] product: paB0CRLY5MYP

**Mutated History:**
  0. [viewItem] product: paB0D8CSSQ7J <-- mutated position
  1. [addToCart] product: paB0CRLY5MYP
  2. [viewItem] product: paB0CPSK1GVB
  3. [addToCart] product: paB0CPSK1GVB
  4. [addToCart] product: paB0CPSK1GVB
  5. [viewItem] product: paB0CRLY5MYP

### 2. Empirical Model Outputs
#### Original Output (Top 5):
  - [paB0CN9GKZ2W] Timberland Men's 6" Premium Waterproof Boot F | prob: 0.048688 | price: $74.0
  - [paB0CPSK59ST] Timberland Men's 6" Premium Waterproof Boot | prob: 0.037331 | price: $35.0
  - [paB0D6BWY8Q3] Timberland Men's 6in Water Resistant Basic Fa | prob: 0.028622 | price: $69.0
  - [paB0CQ2VRMMS] Timberland Men's 6" Premium Waterproof Boot | prob: 0.021605 | price: $0.0
  - [paB075XJR9X7] Timberland Men's 6" Premium Waterproof Fashio | prob: 0.012701 | price: $0.0

#### Mutated Output (Top 5):
  - [paB0D8CJB151] COACH Tabby Shoulder Bag 20, B4/Maple | prob: 0.038451 | price: $371.0
  - [paB0D8CQ1YD6] COACH Tabby Shoulder Bag 20, B4/Cedar | prob: 0.026427 | price: $37.0
  - [paB0CQFC3SRP] UGG Women's Classic Ultra Mini New Heights Fa | prob: 0.014368 | price: $79.0
  - [paB0D8CJ44D2] COACH Tabby Shoulder Bag 26, B4/Merlot | prob: 0.010677 | price: $450.0
  - [paB0D8CSSQ7J] COACH Tabby Shoulder Bag 20, LH/Dark Ruby | prob: 0.010188 | price: $395.0

### 3. Quantitative Measurements
- **Top-1 Item ID Changed:** True
- **Original Top-1:** paB0CN9GKZ2W
- **Mutated Top-1:** paB0D8CJB151
- **Jaccard Similarity @ 5:** 0.00%
- **Jaccard Similarity @ 10:** 17.65%
- **Mean Rank Displacement:** 7.00
- **Shared Probability TVD:** 0.048909

### 4. Engineering Assessment & Sanity Checks
1. **Request Integrity Check:** Both requests shared identical catalog ID, market (`US`), and timestamps.
2. **Invariant Verification:** Exactly one action was altered; non-mutated event IDs, timestamps, and order were preserved.
3. **Inference Consistency:** Mutating the action at position 0 (early) resulted in a Top-1 change, indicating the model output is sensitive to this position.
4. **Scientific Scope:** Observations describe input-output sensitivity. No claims are made about model internals, causal mechanisms, or attention patterns.