# Architecture

## System Overview

The BehaviorGPT Counterfactual Sensitivity Lab is a research prototype that
investigates how single-action mutations in behavioral histories affect the
BehaviorGPT model's recommendations.

```
┌────────────────────────────────────────────────────────────────┐
│                        Entry Points                            │
│                                                                │
│  scripts/run_experiment.py    (CLI batch experiment runner)     │
│  src/behavior_lab/api.py      (FastAPI REST API + dashboard)   │
│  scripts/verify_behaviorgpt_sdk.py  (SDK health check)         │
└────────┬───────────────────────────────────┬───────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────────┐        ┌──────────────────────────┐
│  ExperimentRunner    │        │  FastAPI Application     │
│  (runner.py)         │        │  (api.py)                │
│                      │        │                          │
│  Orchestrates:       │        │  GET  /api/catalog/items │
│  1. Dataset gen      │        │  POST /api/journey/simulate │
│  2. Mutation gen     │        │  POST /api/counterfactual/compare │
│  3. Paired inference │        │  GET  /api/items/{id}/similar │
│  4. Metric compute   │        │  POST /api/items/similar │
│  5. Result persist   │        │  GET  /api/summary       │
│                      │        │  GET  /api/cases         │
│                      │        │  GET  /api/cases/{id}    │
│                      │        │  GET  /api/investigation │
│                      │        │  POST /api/run-interactive│
│                      │        │  GET  / (dashboard)      │
└────────┬────────────┘        └──────────┬───────────────┘
         │                                 │
         ▼                                 ▼
┌──────────────────────────────────────────────────────────┐
│              InferenceProvider Protocol                    │
│                                                          │
│  predict(history, limit, variant_label, case_id)         │
│  → InferenceOutput                                       │
│                                                          │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ BehaviorGPTClient│  │ SDK Adapter  │  │ Mock Mode  │ │
│  │ (mode="live")    │  │ (mode="sdk") │  │(mode="mock")│ │
│  │                  │  │              │  │            │ │
│  │ Showcase API     │  │ Official SDK │  │ Offline    │ │
│  │ No auth needed   │  │ Needs API key│  │ Determ.    │ │
│  └────────┬─────────┘  └──────┬───────┘  └────────────┘ │
└───────────┼────────────────────┼─────────────────────────┘
            │                    │
            ▼                    ▼
   behaviorgpt.unboxai.com   behaviorgpt-northeurope
   /api/recommendations      .api.unboxai.com/v1
                              /vector/recommendations
```

## Component Responsibilities

### Data Models (`models.py`)

Pydantic v2 models defining the experiment's type system:

| Model | Purpose |
|---|---|
| `ActionType` | Enum: `viewItem`, `addToCart`, `search`, `order` |
| `BehavioralAction` | Single event in a behavioral sequence |
| `BehavioralHistory` | Complete user history with metadata |
| `MutationSpec` | Describes the exact single-action change |
| `PairedExperimentCase` | Original + mutated history pair |
| `PredictionItem` | Single recommendation with score, model_score, prob, and metadata |
| `InferenceOutput` | Full inference response with provenance |
| `SensitivityMetrics` | Computed divergence metrics between two outputs (shared & penalized) |
| `PairedResult` | Complete experiment case with both outputs + metrics |
| `ExperimentSummary` | Aggregate statistics across all cases |
| `JourneyEventInput` | Wire schema for interactive multi-action sequence events |
| `JourneySimulateRequest` | Request schema for Journey Simulator endpoint |
| `CounterfactualCompareRequest` | Request schema for Counterfactual Playground endpoint |
| `SimilarItemRequest` | Request schema for Similar Item Explorer endpoint |
| `ItemTransitionsSummary` | Full set migration breakdown (shared, evicted, new items) |

### Inference Layer

#### Showcase Client (`client.py`)
- Direct HTTP client using `httpx`
- Calls `https://behaviorgpt.unboxai.com/api/recommendations`
- No API key required for default catalog
- Constructs raw wire-format payloads

#### SDK Adapter (`sdk_adapter.py`)
- Synchronous (`BehaviorGPTSDKAdapter`) and asynchronous (`AsyncBehaviorGPTSDKAdapter`) adapters
- Wraps official `behaviorgpt.UnboxAIClient` and `AsyncUnboxAIClient`
- Converts `BehavioralAction` → SDK event types (`View`, `AddToCart`, `Search`, `Order`)
- Injects deterministic relative timestamps to guarantee temporal request equivalence
- Exposes `similar_items()` for recommendation vs. item similarity analysis
- Normalizes `UnboxAIResponse.items` → `PredictionItem` list
- Exposes raw model scores in `model_score`, reserving `prob` strictly for calibrated probabilities
- Maps all SDK exceptions to structured `InferenceOutput` error responses
- Requires `UNBOXAI_API_KEY` environment variable (targets `retail_catalog_500k`)

#### Mock Mode (`client.py`)
- Deterministic offline inference for testing
- Uses history content to seed predictable recommendations
- `is_mock=True` flag on all outputs

### Experiment Pipeline

```
SyntheticHistoryGenerator  →  ControlledMutator  →  ExperimentRunner
        │                           │                      │
   Generates N               Creates paired          Runs inference
   behavioral               single-action             on both
   histories               mutations                  histories
        │                           │                      │
        ▼                           ▼                      ▼
  synthetic_histories.jsonl  paired_experiment_cases.jsonl  │
                                                           │
                          SensitivityMetricsCalculator  ◄──┘
                                    │
                                    ▼
                          experiment_results.jsonl
                          experiment_summary.json
```

### Metrics (`metrics.py`)

| Metric | Formula | Interpretation |
|---|---|---|
| Top-1 Flip Rate | `I(orig_top1 ≠ mut_top1)` | Binary: did the top recommendation change? |
| Top-1 Score Delta | `\|score_orig - score_mut\|` | Absolute change in raw model score for top item |
| Jaccard@K | `\|A∩B\| / \|A∪B\|` for top-K sets | Set overlap (1.0 = identical, 0.0 = disjoint) |
| Shared Rank Displacement | `mean(\|rank_orig - rank_mut\|)` over shared items | Average position change on shared items (None if disjoint) |
| Penalized Rank Displacement | `mean(\|rank_orig - rank_mut\|)` with penalty rank $K+1$ | Displacement across union including evicted & new items |
| Spearman Correlation | `ρ` on shared item ranks | Rank-order preservation (None if shared < 3) |
| Prob TVD | `½ Σ\|p_orig - p_mut\|` on shared items | Probability distribution shift (gated to probs in [0, 1]) |
| Score L1 Shared | `mean(\|score_orig - score_mut\|)` on shared items | Absolute score shift on uncalibrated model scores |
| Price Mean Delta | `mean_price_mut - mean_price_orig` | Price drift from mutation |
| Category Shifted | `top1_cat_orig ≠ top1_cat_mut` | Boolean: category change |

### Data Persistence

Experiment datasets are strictly partitioned by execution mode:
- `data/` — Historical benchmark records generated against the public showcase endpoint:
  - `synthetic_histories.jsonl` — 40 reproducible synthetic behavioral histories
  - `paired_experiment_cases.jsonl` — 40 single-action paired counterfactual cases
  - `experiment_results.jsonl` — inference outputs + metrics per case
  - `experiment_summary.json` — aggregate showcase statistics
  - `experiment_results.csv` — tabular export
- `data/sdk_results/` — Live authenticated SDK experiment records (git-ignored to prevent overwriting historical records)

### Frontend (`web/index.html`)

Single-page dark-themed dashboard with 4 tabs:
1. **Inspector** — browse individual experiment cases
2. **Breakdowns** — metrics by mutation type and position
3. **Deep-Dive** — detailed case investigation
4. **Sandbox** — live interactive counterfactual comparison

## Security

- API keys loaded from environment variables only (`UNBOXAI_API_KEY`)
- Never hardcoded or logged
- `.env.example` provided with placeholder
- `.gitignore` excludes `.env`
