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
│  Orchestrates:       │        │  GET  /api/summary       │
│  1. Dataset gen      │        │  GET  /api/cases         │
│  2. Mutation gen     │        │  GET  /api/cases/{id}    │
│  3. Paired inference │        │  GET  /api/investigation │
│  4. Metric compute   │        │  POST /api/run-interactive│
│  5. Result persist   │        │  GET  / (dashboard)      │
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
| `ActionType` | Enum: `viewItem`, `addToCart`, `search` |
| `BehavioralAction` | Single event in a behavioral sequence |
| `BehavioralHistory` | Complete user history with metadata |
| `MutationSpec` | Describes the exact single-action change |
| `PairedExperimentCase` | Original + mutated history pair |
| `PredictionItem` | Single recommendation with score and metadata |
| `InferenceOutput` | Full inference response with provenance |
| `SensitivityMetrics` | Computed divergence metrics between two outputs |
| `PairedResult` | Complete experiment case with both outputs + metrics |
| `ExperimentSummary` | Aggregate statistics across all cases |

### Inference Layer

#### Showcase Client (`client.py`)
- Direct HTTP client using `httpx`
- Calls `https://behaviorgpt.unboxai.com/api/recommendations`
- No API key required for default catalog
- Constructs raw wire-format payloads

#### SDK Adapter (`sdk_adapter.py`)
- Wraps official `behaviorgpt.UnboxAIClient`
- Converts `BehavioralAction` → SDK event types (`View`, `AddToCart`, `Search`)
- Normalizes `UnboxAIResponse.items` → `PredictionItem` list
- Maps all SDK exceptions to structured `InferenceOutput` error responses
- Requires `UNBOXAI_API_KEY` environment variable

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
| Jaccard@K | `|A∩B| / |A∪B|` for top-K sets | Set overlap (1.0 = identical, 0.0 = disjoint) |
| Mean Rank Displacement | `mean(|rank_orig - rank_mut|)` over shared items | Average position change |
| Spearman Correlation | `ρ` on shared item ranks | Rank-order preservation |
| Prob TVD | `½ Σ|p_orig - p_mut|` on shared items | Probability distribution shift |
| Price Mean Delta | `mean_price_mut - mean_price_orig` | Price drift from mutation |
| Category Shifted | `top1_cat_orig ≠ top1_cat_mut` | Boolean: category change |

### Data Persistence

All experiment data is stored in `data/` as JSONL and JSON:
- `synthetic_histories.jsonl` — generated behavioral histories
- `paired_experiment_cases.jsonl` — original + mutated pairs
- `experiment_results.jsonl` — inference outputs + metrics per case
- `experiment_summary.json` — aggregate statistics
- `experiment_results.csv` — tabular export

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
