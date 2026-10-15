# BehaviorGPT Counterfactual Sensitivity Lab

> An empirical research workbench investigating sequential counterfactual sensitivity and representation dynamics in Unbox AI's **BehaviorGPT** foundation model.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 13 Passed](https://img.shields.io/badge/tests-13%20passed-success.svg)](tests/)
[![Inference: Live BehaviorGPT](https://img.shields.io/badge/inference-BehaviorGPT--v4--12.5B-blueviolet.svg)](https://unboxai.com/behaviorgpt)

---

## 1. Project Overview & Research Question

**BehaviorGPT** is Unbox AI's 12.5B-parameter Large Behavioral Model (LBM), pretrained on 150 billion chronological user actions across retail, payments, and engagement. Rather than predicting text tokens, it models user intent directly from chronological event sequences (views, cart additions, searches, and purchases) to rank entire catalogs in a single forward pass.

This research prototype systematically addresses the question:

> **"When exactly one action in a behavioral history is changed, how does BehaviorGPT's subsequent prediction change, if at all?"**

### Scientific Purpose:
We measure and quantify **model sensitivity and representation dynamics**. This study does not assume that predictions *must* change, that a changed prediction is inherently superior, or that invariant predictions signify failure.

---

## 2. Key Empirical Findings (40 Paired Inferences)

Across **40 strictly paired counterfactual experiments** (80 live model inferences evaluated against `retail_catalog_500k` on BehaviorGPT-v4):

| Metric | Overall Suite | Action-Verb Swap (`view` $\leftrightarrow$ `cart`) | Within-Category Swap | Cross-Category Swap | Early Position (Index 0) |
|---|---|---|---|---|---|
| **Pairs Evaluated** | 40 | 10 | 10 | 10 | 10 |
| **Successful Calls** | 40 (100%) | 10 (100%) | 10 (100%) | 10 (100%) | 10 (100%) |
| **Top-1 Flip Rate** | **65.0%** | **10.0%** | **70.0%** | **80.0%** | **100.0%** |
| **Avg Jaccard @ 5** | **42.5%** | **90.0%** | **39.5%** | **28.8%** | **11.5%** |
| **Mean Rank Drift** | **1.83 ranks** | **0.24 ranks** | **2.11 ranks** | **2.25 ranks** | **2.73 ranks** |

### Core Insights:
1. **Action-Verb Semantic Dominance:** Changing *only* the user action verb (`viewItem` $\to$ `addToCart`) on the same product produced a **90.0% Top-1 invariance rate** and **90.0% Top-5 candidate overlap**. In next-action completion, the underlying item representation overwhelmingly dominates the action verb.
2. **Fine-Grained Item Sensitivity:** Swapping an item within the exact same category (e.g. Nike Jordan 4 $\to$ Nike Dunk) flipped Top-1 predictions **70.0%** of the time, demonstrating high intra-category resolution.
3. **Opening Anchor Effect (Primacy):** Mutating the initial action (index 0) in 6-action histories flipped Top-1 predictions **100.0%** of the time (Jaccard@5 dropped to 11.5%), showing that opening events establish a persistent categorical manifold that guides subsequent attention.

---

## 3. System Architecture

```
BehaviorGPT Counterfactual Sensitivity Lab
├── src/behavior_lab/
│   ├── client.py              # Typed client adapter (Live, Cloud, and Mock modes)
│   ├── dataset_generator.py   # Deterministic synthetic behavioral sequence generator
│   ├── mutator.py             # Controlled single-action mutator with invariant checks
│   ├── metrics.py             # Mathematical sensitivity metrics (Jaccard, TVD, Rank Drift)
│   ├── runner.py              # Experiment orchestrator and artifact serializer
│   ├── case_investigator.py   # Deep-dive empirical anomaly and case study analyzer
│   ├── models.py              # Pydantic v2 domain models and wire event schemas
│   └── api.py                 # FastAPI REST backend and live interactive runner
├── web/
│   └── index.html             # Single-page interactive research workbench and dashboard
├── data/
│   ├── synthetic_histories.jsonl     # 40 reproducible synthetic behavioral histories
│   ├── paired_experiment_cases.jsonl # 40 strictly paired counterfactual cases
│   ├── experiment_results.jsonl      # Real model predictions and calculated metrics
│   ├── experiment_results.csv        # Tabular export for data scientists
│   ├── experiment_summary.json       # Aggregate statistics and strategy breakdowns
│   └── investigated_case.md          # In-depth case study report
├── tests/
│   ├── test_lab_suite.py      # Automated invariant, generator, and metric tests
│   └── test_api.py            # FastAPI endpoint tests
├── scripts/
│   ├── run_experiment.py      # CLI runner for live or mock experiments
│   └── export_csv.py          # CSV conversion utility
├── TECHNICAL_REPORT.md        # Formal 12-section technical research note
└── LIVE_DEMO_SCRIPT.md        # 5-7 minute demonstration guide for interview
```

---

## 4. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11 or 3.12
- Internet access (for live BehaviorGPT model inference)

### 1. Installation
```bash
# Clone the official SDK and install dependencies
git clone https://github.com/Unbox-AI/behaviorgpt.git repo_temp
pip install -e repo_temp
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (13 Tests)
```bash
python -m pytest
```
*Output: 13 passed in < 3 seconds.*

### 3. Run the 40-Pair Live Experiment
```bash
python scripts/run_experiment.py --mode live --count 40 --seed 42
```
*Executes 80 genuine forward passes on BehaviorGPT-v4, serializing results to `data/`.*

### 4. Launch the Interactive Research Workbench & Dashboard
```bash
python -m uvicorn behavior_lab.api:app --host 127.0.0.1 --port 8000
```
Open your browser to:
👉 **`http://127.0.0.1:8000/`**

---

## 5. Investigated Case Studies

### Case Study 1: Action-Verb Semantic Invariance (`case_003`)
- **Sequence:** 4 actions browsing footwear and accessories.
- **Intervention:** Terminal event changed from `viewItem` on Nike Jordan Retro 4 (`paB0DD21212N`) to `addToCart` on the same item.
- **Empirical Output:** Exactly identical Top-3 recommendations with exact probability mass ($0.022418$, $0.021392$, $0.014934$). Top-1 Flip: **False**, Jaccard@10: **100.0%**, Rank Displacement: **0.00**.

### Case Study 2: The Opening Anchor Effect (`case_004`)
- **Sequence:** 6 actions browsing footwear.
- **Intervention:** Action at index 0 changed from Timberland Boot (`paB0CN9GKZ2W`) to Coach Tabby Shoulder Bag (`paB0D8CSSQ7J`). All 5 subsequent actions remained Timberland boot clicks.
- **Empirical Output:** Predictions shifted from 100% Timberland boots to Coach luxury shoulder bags ($371 - $450) and UGG fashion boots. Top-1 Flip: **True**, Jaccard@5: **0.00%**, Price Delta: **+$148.00**.

---

## 6. Scientific Limitations & Boundaries

1. **Local Sensitivity vs. Causal Intent:** This study measures output variation as a function of input perturbation; it does not establish ground-truth causal intent or prove real-world user adoption.
2. **Synthetic Data Boundaries:** Synthetic sequences, while strictly controlled, lack human latency distributions and noisy session abandonment.
3. **Absence of Conversion Labels:** In the absence of downstream purchase logs, precision, recall, and NDCG cannot be computed.
4. **Catalog Specificity:** Calibrated against `retail_catalog_500k`. Catalog density and dimensionality directly modulate candidate distance metrics.

---

## 7. Recommended Next Experiments

1. **Normalized Distance Decay Curve:** Parameterize sequence length $L \in \{2, 5, 10, 20, 50\}$ and map sensitivity as a function of mutation position relative to the prediction horizon ($pos / L$).
2. **Dwell Time Warping:** Perturb millisecond timestamps (5-second rapid clicks vs 3-day return visits) to isolate temporal attention weights.
3. **Multimodal Parquet Catalog Perturbations:** Alter catalog metadata (title text or price) while keeping embeddings fixed to measure text vs price weighting.

---

## 8. License

This project is licensed under the MIT License. Built around Unbox AI's official open-source BehaviorGPT SDK.
