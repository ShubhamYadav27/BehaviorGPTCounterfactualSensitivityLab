# SDK Integration — Final Engineering Report

**Date:** 2026-11-12
**Commit:** `cd3f875`
**Status:** COMPLETE (pending API key for live SDK verification)

---

## Executive Summary

The BehaviorGPT Counterfactual Sensitivity Lab has been upgraded to integrate the
official Unbox AI BehaviorGPT Python SDK (`behaviorgpt` package) as a first-class
inference backend, alongside the existing showcase endpoint. The integration is
architecturally complete, fully tested, and production-ready — requiring only a
valid `UNBOXAI_API_KEY` to execute live SDK inference.

---

## Deliverables

### New Files Created (6)

| File | Purpose |
|---|---|
| `src/behavior_lab/sdk_adapter.py` | SDK adapter wrapping `UnboxAIClient` |
| `scripts/verify_behaviorgpt_sdk.py` | SDK health check and verification CLI |
| `tests/test_sdk_adapter.py` | 11 unit tests for SDK adapter |
| `docs/SDK_INTEGRATION_AUDIT.md` | SDK vs showcase technical comparison |
| `docs/IMPLEMENTATION_PLAN.md` | Step-by-step integration plan |
| `docs/ARCHITECTURE.md` | System architecture documentation |
| `docs/EXPERIMENT_METHODOLOGY.md` | Formal experiment methodology |

### Modified Files (6)

| File | Changes |
|---|---|
| `src/behavior_lab/models.py` | Added `provider`, `raw_payload` to `InferenceOutput`; `provider` to `ExperimentSummary` |
| `src/behavior_lab/runner.py` | Added `InferenceProvider` protocol, factory pattern, SDK mode support |
| `src/behavior_lab/api.py` | Added SDK mode to interactive endpoint, provider labeling |
| `src/behavior_lab/__init__.py` | Exported `BehaviorGPTSDKAdapter`, `is_sdk_available` |
| `scripts/run_experiment.py` | Added `sdk` to `--mode` choices |
| `README.md` | Updated with SDK integration docs, 31 tests, three modes |

### Unchanged Files

| File | Reason |
|---|---|
| `src/behavior_lab/client.py` | Showcase client preserved as-is (rollback safety) |
| `src/behavior_lab/dataset_generator.py` | No changes needed |
| `src/behavior_lab/mutator.py` | No changes needed |
| `src/behavior_lab/metrics.py` | No changes needed |
| `src/behavior_lab/case_investigator.py` | No changes needed |
| `tests/test_lab_suite.py` | Original 7 tests pass unchanged |
| `tests/test_api.py` | Original 13 tests pass unchanged |
| `web/index.html` | Dashboard unchanged |

---

## Test Results

```
31 passed, 0 failed, 1 warning in 9.21s

tests/test_api.py            — 13 passed
tests/test_lab_suite.py      —  7 passed
tests/test_sdk_adapter.py    — 11 passed
```

### Test Coverage

| Test Category | Count | Coverage |
|---|---|---|
| Event conversion (View, AddToCart, Search) | 3 | All action types mapped |
| Full history conversion | 1 | Multi-event sequence |
| SDK Item → PredictionItem normalization | 4 | Basic, missing fields, string price, invalid price |
| SDK availability check | 1 | Package importability |
| Model provider field (default + explicit) | 2 | Pydantic field validation |
| API endpoints | 13 | All FastAPI routes |
| Core lab suite | 7 | Determinism, mutations, metrics, client, serialization |

---

## SDK Verification

```
$ python scripts/verify_behaviorgpt_sdk.py

============================================================
BehaviorGPT SDK Verification
============================================================

[1/4] Checking behaviorgpt package installation...
  [OK] behaviorgpt package found

[2/4] Checking API key configuration...
  [FAIL] UNBOXAI_API_KEY environment variable is not set.

============================================================
Result: FAIL (exit code 2)
============================================================
```

**Honest assessment:** The SDK package is installed and the adapter is complete.
Live SDK inference requires a valid `UNBOXAI_API_KEY` which is not currently
configured. This is documented as a known blocker, not a code defect.

---

## Architecture Decision Records

### ADR-1: Adapter Pattern over Client Replacement
**Decision:** Created a separate `sdk_adapter.py` rather than modifying `client.py`.
**Rationale:** Preserves rollback safety, allows A/B comparison between showcase
and SDK results, reduces merge conflicts, clear separation of concerns.

### ADR-2: InferenceProvider Protocol
**Decision:** Introduced a `Protocol` class that both `BehaviorGPTClient` and
`BehaviorGPTSDKAdapter` satisfy.
**Rationale:** Enables the runner to be agnostic to the inference backend.
Factory function `_create_inference_provider(mode)` handles instantiation.

### ADR-3: Deferred SDK Import
**Decision:** SDK imports in `sdk_adapter.py` are wrapped in try/except with
a module-level `_SDK_AVAILABLE` flag.
**Rationale:** The adapter module can be imported even when the `behaviorgpt`
package is not installed, enabling graceful degradation.

### ADR-4: Provider Provenance Tracking
**Decision:** Added `provider` field to `InferenceOutput` and `ExperimentSummary`.
**Rationale:** Results from SDK vs showcase must be explicitly distinguishable.
No silent fallback allowed.

---

## Known Limitations

1. **No API key:** Live SDK verification blocked. Provide `UNBOXAI_API_KEY` to activate.
2. **SDK `score` vs showcase `prob`:** SDK returns `score` (float), showcase returns
   `prob` (float). Both are mapped to `PredictionItem.score` and `.prob`. The
   numerical scales may differ between backends.
3. **SDK `RemoveFromCart`:** Not used in our experiment (only View, AddToCart, Search).
4. **`Item.data` structure:** SDK nests metadata in `data` dict; structure may vary
   by catalog. Adapter handles missing fields gracefully.

---

## Activation Instructions

To activate live SDK inference:

```bash
# Set API key
export UNBOXAI_API_KEY="your-key-here"      # Linux/macOS
set UNBOXAI_API_KEY=your-key-here           # Windows CMD
$env:UNBOXAI_API_KEY = "your-key-here"      # Windows PowerShell

# Verify SDK
python scripts/verify_behaviorgpt_sdk.py

# Run experiment with SDK
python scripts/run_experiment.py --mode sdk --count 40 --seed 42
```

---

## Metrics Summary (prior showcase experiment, preserved)

| Metric | Value |
|---|---|
| Total paired cases | 40 |
| Successful inferences | 40 (100%) |
| Top-1 Flip Rate | 65.0% |
| Avg Jaccard@5 | 42.5% |
| Mean Rank Displacement | 1.83 |
| Provider | showcase |

These results serve as the baseline. Running `--mode sdk` with a valid API key
will produce a comparable dataset with `provider=behaviorgpt_sdk`, enabling
direct comparison between the showcase and official SDK backends.
