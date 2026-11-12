# SDK Integration Implementation Plan

**Date:** 2026-11-12
**Status:** IN PROGRESS

---

## 1. Objective

Replace the showcase-based inference path with the official BehaviorGPT Python SDK
while preserving the existing research pipeline, metrics, dashboard, and test suite.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  ExperimentRunner                        │
│                                                         │
│  mode="live"  → BehaviorGPTClient  (showcase endpoint)  │
│  mode="mock"  → BehaviorGPTClient  (deterministic mock) │
│  mode="sdk"   → BehaviorGPTSDKAdapter (official SDK)    │
│                                                         │
│  All backends implement the InferenceProvider protocol:  │
│    predict(history, limit, variant_label, case_id)       │
│    → InferenceOutput                                     │
└─────────────────────────────────────────────────────────┘
```

The `InferenceProvider` protocol ensures the runner is agnostic to the backend.
A factory function `_create_inference_provider(mode)` returns the correct backend.

---

## 3. Implementation Steps

### Step 1: Model Updates ✓
**File:** `src/behavior_lab/models.py`

- Add `provider: str = "showcase"` to `InferenceOutput`
- Add `raw_payload: Optional[Dict[str, Any]] = None` to `InferenceOutput`
- Add `provider: str = "showcase"` to `ExperimentSummary`
- Update `execution_mode` comment to include `"sdk"`

### Step 2: SDK Adapter ✓
**File:** `src/behavior_lab/sdk_adapter.py` (NEW)

Components:
- `is_sdk_available()` — checks if `behaviorgpt` is importable
- `_action_to_sdk_event(action)` — converts `BehavioralAction` → SDK event
- `_history_to_sdk_events(history)` — converts full history
- `_sdk_item_to_prediction(item, rank)` — converts SDK `Item` → `PredictionItem`
- `BehaviorGPTSDKAdapter` class:
  - `__init__(market, catalog_id, api_key, timeout)`
  - `predict(history, limit, variant_label, case_id)` → `InferenceOutput`
  - `close()`, context manager support
  - Structured exception handling for all SDK error types
  - Structured logging (no credential exposure)

### Step 3: Runner Update ✓
**File:** `src/behavior_lab/runner.py`

- Add `InferenceProvider` protocol
- Add `_create_inference_provider(mode)` factory
- Accept `mode="sdk"` in addition to `"live"` and `"mock"`
- Pass `provider_label` to `_compute_summary`
- Record provider in `ExperimentSummary`
- Extract `model_name` from actual inference results

### Step 4: Package Exports ✓
**File:** `src/behavior_lab/__init__.py`

- Add `BehaviorGPTSDKAdapter` and `is_sdk_available` to exports

### Step 5: SDK Verification Script ✓
**File:** `scripts/verify_behaviorgpt_sdk.py` (NEW)

- Checks SDK installation
- Checks API key configuration
- Initializes UnboxAIClient
- Executes minimal test inference
- Returns structured exit codes (0–5)
- Reports blockers honestly

### Step 6: CLI Update ✓
**File:** `scripts/run_experiment.py`

- Add `sdk` to `--mode` choices

### Step 7: Unit Tests ✓
**File:** `tests/test_sdk_adapter.py` (NEW)

Tests:
- Event conversion (View, AddToCart, Search)
- Full history conversion
- SDK Item → PredictionItem conversion (basic, missing fields, string price, invalid price)
- SDK availability check
- InferenceOutput provider field (default and explicit)

### Step 8: API Layer Update
**File:** `src/behavior_lab/api.py`

- Add `provider` query parameter to `/api/summary` response
- Update `/api/run-interactive` to optionally accept `mode` parameter
- Clearly label SDK vs showcase results in API responses

### Step 9: Documentation
- `docs/SDK_INTEGRATION_AUDIT.md` ✓
- `docs/IMPLEMENTATION_PLAN.md` ✓ (this file)
- `docs/ARCHITECTURE.md` (pending)
- `docs/EXPERIMENT_METHODOLOGY.md` (pending)
- Update `README.md` with SDK instructions

---

## 4. File Inventory

| File | Status | Change Type |
|---|---|---|
| `src/behavior_lab/sdk_adapter.py` | ✓ | NEW |
| `src/behavior_lab/models.py` | ✓ | MODIFIED |
| `src/behavior_lab/runner.py` | ✓ | MODIFIED |
| `src/behavior_lab/client.py` | — | UNCHANGED (showcase preserved) |
| `src/behavior_lab/__init__.py` | ✓ | MODIFIED |
| `src/behavior_lab/api.py` | PENDING | MODIFY |
| `scripts/verify_behaviorgpt_sdk.py` | ✓ | NEW |
| `scripts/run_experiment.py` | ✓ | MODIFIED |
| `tests/test_sdk_adapter.py` | ✓ | NEW |
| `tests/test_lab_suite.py` | — | UNCHANGED |
| `tests/test_api.py` | — | UNCHANGED |
| `docs/SDK_INTEGRATION_AUDIT.md` | ✓ | NEW |
| `docs/IMPLEMENTATION_PLAN.md` | ✓ | NEW |

---

## 5. Testing Strategy

| Test Layer | What | Mock SDK? |
|---|---|---|
| Unit: event conversion | `_action_to_sdk_event` correctness | No SDK call |
| Unit: response normalization | `_sdk_item_to_prediction` correctness | Mocked Item objects |
| Unit: model fields | `InferenceOutput.provider` default | No SDK call |
| Integration: adapter | `BehaviorGPTSDKAdapter.predict` with mock client | Mocked `UnboxAIClient` |
| Regression: existing tests | All 13 existing tests must pass | N/A |
| Live verification | `scripts/verify_behaviorgpt_sdk.py` | Real SDK (requires API key) |

---

## 6. Known Blockers

| Blocker | Impact | Resolution |
|---|---|---|
| **No `UNBOXAI_API_KEY`** | Cannot verify live SDK inference | Implementation complete; provide key to activate |
| Showcase endpoint may be rate-limited | Existing live experiments may fail intermittently | Handled with error responses |

---

## 7. Rollback Plan

The showcase client (`client.py`) is **not modified**. If SDK integration fails:
1. Use `--mode live` to fall back to showcase
2. Remove `sdk_adapter.py` and revert `runner.py` and `models.py`
3. All existing tests and functionality remain intact
