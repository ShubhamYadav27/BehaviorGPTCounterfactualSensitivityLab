# Experiment Methodology

## 1. Research Question

> When exactly one action in a behavioral history is changed, how does
> BehaviorGPT's subsequent prediction change, if at all?

This is a **sensitivity analysis** question. It does NOT assume that:
- Predictions must change
- A changed prediction is necessarily better
- The model is defective when its output remains unchanged

The purpose is to **measure and understand** actual model behavior.

## 2. Experimental Design

### 2.1 Independent Variable
A single action in the behavioral sequence, modified in a controlled way.

### 2.2 Mutation Types

| Mutation | What changes | What stays constant |
|---|---|---|
| **Cross-category swap** | Product ID replaced with a product from a different category | Action type, position, sequence length |
| **Within-category swap** | Product ID replaced with a product from the same category | Action type, position, category, sequence length |
| **Action-type swap** | Action type changed (e.g., `viewItem` ↔ `addToCart`) | Product ID, position, sequence length |
| **Brand swap** | Product replaced with same-category different-brand product | Action type, position, category, sequence length |

### 2.3 Mutation Positions

| Position | Indices | Hypothesis |
|---|---|---|
| **Early** | First 33% of sequence | Early events may have less influence (recency bias expected) |
| **Middle** | Middle 33% | Intermediate influence |
| **Late** | Last 33% | Recent events may dominate predictions |

### 2.4 Control
The **original history** serves as the control. Each mutated history differs
from its control by exactly one action, ensuring any observed change in
predictions is attributable to that single mutation.

## 3. Data Generation

### 3.1 Synthetic Histories
We generate synthetic behavioral histories using `SyntheticHistoryGenerator`
with a fixed random seed for reproducibility.

**5 archetypes** model distinct user personas:
1. Tech Enthusiast — electronics-heavy browsing
2. Fashion Forward — apparel and accessories
3. Home & Kitchen — home goods and appliances
4. Sports & Outdoors — athletic gear and equipment
5. Mixed Browser — cross-category exploration

Each history contains 6–12 actions drawn from verified catalog items
in the `retail_catalog_500k` catalog.

### 3.2 Catalog Items
All product IDs used in synthetic histories are drawn from items known to
exist in BehaviorGPT's default retail catalog. This ensures the model can
meaningfully process the events rather than encountering unknown items.

### 3.3 Deterministic Seeding
All randomness is seeded (`seed=42` by default) to ensure:
- Identical histories across runs
- Identical mutations across runs
- Reproducible experiment results (subject to model server state)

## 4. Inference Protocol

### 4.1 Paired Execution
For each experiment case:

```
Original history → Inference → Original recommendations (top-K)
Mutated history  → Inference → Mutated recommendations  (top-K)
```

Both histories are submitted with identical parameters:
- Same `limit` (default 10)
- Same catalog
- Same market
- Sequential execution (no parallel requests that could hit different server replicas)

### 4.2 Inference Backends

| Mode | Backend | Auth | Use case |
|---|---|---|---|
| `live` | Showcase endpoint (`behaviorgpt.unboxai.com`) | None | Default, no setup needed |
| `sdk` | Official SDK (`behaviorgpt` package) | `UNBOXAI_API_KEY` | Production integration |
| `mock` | Deterministic mock | None | Unit tests, offline development |

### 4.3 Provenance Recording
Each `InferenceOutput` records:
- `provider`: `"showcase"` or `"behaviorgpt_sdk"`
- `model_name`: as reported or labeled
- `catalog_id`: catalog used
- `latency_ms`: wall-clock time
- `status_code`: HTTP status
- `is_mock`: boolean flag

## 5. Metrics

### 5.1 Primary Metrics

**Top-1 Flip Rate:**
The fraction of cases where the top-1 recommended item changed.
$$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{top1}_{\text{orig}}^{(i)} \neq \text{top1}_{\text{mut}}^{(i)}]$$

**Jaccard@K:**
Set overlap between original and mutated top-K recommendation sets.
$$J@K = \frac{|S_{\text{orig}}^K \cap S_{\text{mut}}^K|}{|S_{\text{orig}}^K \cup S_{\text{mut}}^K|}$$

Measured at K ∈ {3, 5, 10}.

### 5.2 Secondary Metrics

**Mean Rank Displacement:**
Average absolute rank change for items appearing in both top-K lists.
$$\text{MRD} = \frac{1}{|I_{\text{shared}}|} \sum_{j \in I_{\text{shared}}} |r_{\text{orig}}(j) - r_{\text{mut}}(j)|$$

> **Note:** This metric is computed only over shared items. Items that
> enter or leave the top-K are not counted, which understates actual
> disruption when the mutation causes significant list turnover.

**Spearman Rank Correlation:**
Rank-order correlation between original and mutated positions of shared items.
Requires ≥ 3 shared items.

**Probability TVD:**
Total variation distance on model scores for shared items.
$$\text{TVD} = \frac{1}{2} \sum_{j \in I_{\text{shared}}} |p_{\text{orig}}(j) - p_{\text{mut}}(j)|$$

### 5.3 Exploratory Metrics

- **Price Mean Delta:** Shift in average recommended price
- **Category Shifted:** Whether top-1 item's category changed

## 6. Analysis Dimensions

Results are broken down by:
1. **Mutation type** — which kind of change was made
2. **Mutation position** — where in the sequence the change occurred
3. **Archetype** — which user persona was mutated

## 7. Limitations

1. **Synthetic data only.** Histories are generated, not observed from real users.
   The model's behavior on synthetic sequences may differ from production traffic.

2. **No control over server-side state.** The model may use global state, A/B
   testing, or caching that introduces variability unrelated to our mutations.

3. **Sequential execution.** We cannot guarantee identical server conditions
   between the original and mutated inference calls.

4. **Catalog coverage.** We use a subset of known catalog items. The model's
   behavior on unknown or rare items may differ.

5. **Single mutation only.** We test one change at a time. Interaction effects
   between multiple mutations are not measured.

6. **Model version opacity.** The showcase endpoint does not return a model
   version identifier. The label "BehaviorGPT-v4-12.5B" is from marketing
   materials and is not verified server-side.

## 8. Reproducibility

To reproduce the experiment:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e repo_temp  # Install official SDK

# Run with showcase (no API key needed)
python scripts/run_experiment.py --mode live --count 40 --seed 42

# Run with official SDK (requires API key)
export UNBOXAI_API_KEY="your-key-here"
python scripts/run_experiment.py --mode sdk --count 40 --seed 42

# Run offline with mock (deterministic, no network)
python scripts/run_experiment.py --mode mock --count 40 --seed 42
```

Results are saved to `data/` and can be explored via the dashboard:
```bash
uvicorn behavior_lab.api:app --port 8000
# Open http://localhost:8000
```
