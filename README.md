# BehaviorGPT Counterfactual Sensitivity Lab

> An empirical research workbench investigating sequential counterfactual sensitivity and representation dynamics in Unbox AI's **BehaviorGPT** foundation model.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 51 Passed](https://img.shields.io/badge/tests-51%20passed-success.svg)](tests/)
[![Inference: BehaviorGPT SDK](https://img.shields.io/badge/inference-BehaviorGPT%20SDK-blueviolet.svg)](https://github.com/Unbox-AI/behaviorgpt)

---

## 1. Project Overview & Research Question

**BehaviorGPT** is Unbox AI's Large Behavioral Model (LBM). According to Unbox AI's marketing materials, it was pretrained on 150 billion chronological user actions across retail, payments, and engagement. Rather than predicting text tokens, it models user intent directly from chronological event sequences (views, cart additions, searches, and purchases) to rank entire catalogs in a single forward pass. The model version ("v4-12.5B") and parameter count cited in this project are from public marketing materials and have not been independently verified.

This research prototype systematically addresses the question:

> **"When exactly one action in a behavioral history is changed, how does BehaviorGPT's subsequent prediction change, if at all?"**

### Scientific Purpose:
We measure and quantify **model sensitivity and representation dynamics**. This study does not assume that predictions *must* change, that a changed prediction is inherently superior, or that invariant predictions signify failure.

---

## 2. Key Empirical Findings (40 Paired Inferences)

Across **40 strictly paired counterfactual experiments** (80 inferences via the public BehaviorGPT showcase endpoint):

| Metric | Overall Suite | Action-Verb Swap | Within-Category Swap | Cross-Category Swap | Early Position (Index 0) |
|---|---|---|---|---|---|
| **Pairs Evaluated** | 40 | 10 | 10 | 10 | 10 |
| **Successful Calls** | 40 (100%) | 10 (100%) | 10 (100%) | 10 (100%) | 10 (100%) |
| **Top-1 Flip Rate** | **65.0%** | **10.0%** | **70.0%** | **80.0%** | **100.0%** |
| **Avg Jaccard @ 5** | **42.5%** | **90.0%** | **39.5%** | **28.8%** | **11.5%** |
| **Mean Rank Drift** | **1.83 ranks** | **0.24 ranks** | **2.11 ranks** | **2.25 ranks** | **2.73 ranks** |

### Core Insights:
1. **Action-Verb Stability:** Changing *only* the user action verb (`viewItem` → `addToCart`) on the same product produced a **90.0% Top-1 invariance rate** and **90.0% Top-5 candidate overlap**.
2. **Fine-Grained Item Sensitivity:** Swapping an item within the exact same category flipped Top-1 predictions **70.0%** of the time, demonstrating high intra-category resolution.
3. **Opening Position Sensitivity:** Mutating the initial action (index 0) in 6-action histories flipped Top-1 predictions **100.0%** of the time (Jaccard@5 dropped to 11.5%).

---

## 3. System Architecture

```
BehaviorGPT Counterfactual Sensitivity Lab
├── src/behavior_lab/
│   ├── client.py              # Showcase endpoint client (live and mock modes)
│   ├── sdk_adapter.py         # Official BehaviorGPT SDK adapter (SDK mode)
│   ├── dataset_generator.py   # Deterministic synthetic behavioral sequence generator
│   ├── mutator.py             # Controlled single-action mutator with invariant checks
│   ├── metrics.py             # Mathematical sensitivity metrics (Jaccard, TVD, Rank Drift)
│   ├── runner.py              # Experiment orchestrator with InferenceProvider protocol
│   ├── case_investigator.py   # Deep-dive empirical anomaly and case study analyzer
│   ├── models.py              # Pydantic v2 domain models and wire event schemas
│   └── api.py                 # FastAPI REST backend and live interactive runner
├── web/
│   └── index.html             # Single-page interactive research workbench and dashboard
├── data/                      # Experiment artifacts (JSONL, JSON, CSV, markdown)
├── tests/
│   ├── test_lab_suite.py      # Core invariant, generator, and metric tests (7 tests)
│   ├── test_api.py            # FastAPI endpoint tests (13 tests)
│   └── test_sdk_adapter.py    # SDK adapter unit tests (11 tests)
├── scripts/
│   ├── run_experiment.py      # CLI runner: --mode live|mock|sdk
│   ├── verify_behaviorgpt_sdk.py  # SDK health check and verification
│   └── export_csv.py          # CSV conversion utility
├── docs/
│   ├── SDK_INTEGRATION_AUDIT.md       # SDK vs showcase technical comparison
│   ├── IMPLEMENTATION_PLAN.md         # Integration implementation plan
│   ├── ARCHITECTURE.md                # System architecture documentation
│   └── EXPERIMENT_METHODOLOGY.md      # Formal experiment methodology
├── TECHNICAL_REPORT.md        # Formal 12-section technical research note
└── LIVE_DEMO_SCRIPT.md        # 5-7 minute demonstration guide for interview
```

### Inference Backends

The system supports three inference modes through a unified `InferenceProvider` protocol:

| Mode | Backend | Auth Required | Use Case |
|---|---|---|---|
| `live` | Showcase endpoint (`behaviorgpt.unboxai.com`) | None | Default, no setup needed |
| `sdk` | Official `behaviorgpt` Python SDK | `UNBOXAI_API_KEY` | Production integration |
| `mock` | Deterministic offline | None | Unit tests, offline development |

---

## 4. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11 or 3.12
- Internet access (for live inference)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/ShubhamYadav27/BehaviorGPTCounterfactualSensitivityLab.git
cd BehaviorGPTCounterfactualSensitivityLab

# Install dependencies and official BehaviorGPT SDK
pip install -r requirements.txt
pip install behaviorgpt
```

### 2. Run Automated Regression Suite (51 Tests)
```bash
python -m pytest tests/ -v
```

### 3. Run Experiments

```bash
# Showcase mode (no API key needed)
python scripts/run_experiment.py --mode live --count 40 --seed 42

# Official SDK mode (requires API key)
export UNBOXAI_API_KEY="your-key-here"  # or on Windows: set UNBOXAI_API_KEY=your-key-here
python scripts/run_experiment.py --mode sdk --count 40 --seed 42

# Mock mode (offline, deterministic)
python scripts/run_experiment.py --mode mock --count 40 --seed 42
```

### 4. Verify SDK Integration
```bash
python scripts/verify_behaviorgpt_sdk.py
```
Reports SDK installation, API key status, and test inference result with structured exit codes.

### 5. Launch the Interactive Dashboard
```powershell
$env:PYTHONPATH = "src"; python -m uvicorn behavior_lab.api:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000/`** in your browser.

---

## 5. Official SDK Integration

The project integrates the official [BehaviorGPT Python SDK](https://github.com/Unbox-AI/behaviorgpt) through a dedicated adapter layer (`sdk_adapter.py`):

```python
from behavior_lab.sdk_adapter import BehaviorGPTSDKAdapter

# Requires UNBOXAI_API_KEY environment variable
with BehaviorGPTSDKAdapter(market="us", catalog_id="retail_catalog") as adapter:
    output = adapter.predict(history, limit=10, variant_label="original", case_id="case_001")
    print(output.items[0].name, output.items[0].score)
```

**Key integration details:**
- Converts `BehavioralAction` objects → SDK event types (`View`, `AddToCart`, `Search`)
- Normalizes `UnboxAIResponse` → internal `InferenceOutput` / `PredictionItem` schema
- Maps all SDK exceptions (`AuthenticationError`, `RateLimitError`, `BadRequestError`) to structured error responses
- Records execution provenance: provider, latency, catalog ID, model identifier
- Results clearly labeled as `provider="behaviorgpt_sdk"` vs `provider="showcase"`

See [docs/SDK_INTEGRATION_AUDIT.md](docs/SDK_INTEGRATION_AUDIT.md) for the complete technical comparison.

---

## 6. Investigated Case Studies

### Case Study 1: Action-Verb Semantic Invariance (`case_003`)
- **Intervention:** Terminal event changed from `viewItem` to `addToCart` on the same Nike Jordan Retro 4.
- **Result:** Exactly identical Top-3 recommendations. Top-1 Flip: **False**, Jaccard@10: **100.0%**.

### Case Study 2: The Opening Anchor Effect (`case_004`)
- **Intervention:** Action at index 0 changed from Timberland Boot to Coach Tabby Shoulder Bag. All 5 subsequent actions remained Timberland boot interactions.
- **Result:** Predictions shifted from Timberland boots to Coach luxury bags. Top-1 Flip: **True**, Jaccard@5: **0.00%**, Price Delta: **+\$148.00**.

---

## 7. Scientific Limitations & Boundaries

1. **Local Sensitivity vs. Causal Intent:** Measures output variation as a function of input perturbation; does not establish ground-truth causal intent.
2. **Synthetic Data Boundaries:** Sequences are controlled but lack real-world noise.
3. **Absence of Conversion Labels:** Without downstream purchase logs, precision/recall/NDCG cannot be computed.
4. **Catalog Specificity:** Calibrated against `retail_catalog_500k`.

---

## 8. Documentation

| Document | Description |
|---|---|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Formal 12-section research note |
| [LIVE_DEMO_SCRIPT.md](LIVE_DEMO_SCRIPT.md) | 5-7 minute demonstration guide |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/EXPERIMENT_METHODOLOGY.md](docs/EXPERIMENT_METHODOLOGY.md) | Experiment design and metrics |
| [docs/SDK_INTEGRATION_AUDIT.md](docs/SDK_INTEGRATION_AUDIT.md) | SDK vs showcase comparison |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Integration implementation plan |

---

## 9. License

This project is licensed under the MIT License. Built as an independent research experiment using Unbox AI's public BehaviorGPT SDK and showcase endpoint.
