# Pre-Demo Technical Audit & Architecture Verification
**Project**: BehaviorGPT Counterfactual Sensitivity Lab & Behavioral Decision Explorer  
**Subject**: Independent Source-Code Audit for Technical Review with Jens Palmborg (Unbox AI)  
**Date**: November 15, 2026  
**Auditor**: Principal Backend Engineer & Senior ML Systems Architect  
**Regression Test Status**: **66 / 66 Passing (100%)** (`pytest tests/ -v`)  

---

## 1. Executive Summary & Alignment Assessment

### Objective
This audit was conducted to independently verify that the project satisfies the original technical objective set by **Jens Palmborg** (Unbox AI):
> *"I want you to build something on the SDK and show it for me, and we can follow up with what you think technically we can develop further on this model."*

### Core Finding: Fully Aligned and Verified
The project **genuinely and correctly integrates the official Unbox AI BehaviorGPT Python SDK (`behaviorgpt v0.11.0`)**. It is not a trivial API wrapper or a superficial script. Instead, it implements a scientifically rigorous **Behavioral Decision Explorer** that investigates sequential attention dynamics, counterfactual sensitivity, and latent embedding geometry:
1. **Behavioral Journey Simulator**: Multi-event behavioral trajectory builder executing through `AsyncUnboxAIClient.complete()`.
2. **Counterfactual Playground**: Pairwise differential engine computing top-K candidate divergence, shared and penalized rank displacement, and item transition flows (shared, evicted, new).
3. **Similar Item Explorer**: Direct latent representation explorer querying catalog embedding neighborhoods via `AsyncUnboxAIClient.similar_items()`.
4. **Single-Origin Architecture**: Unified FastAPI backend serving the interactive dashboard (`http://127.0.0.1:8000`), proxying all API requests, and isolating API keys server-side.
5. **Honest Demarcation**: Explicit architectural separation between the official enterprise SDK integration and the legacy public showcase endpoint.

---

## 2. Official SDK Integration Verification

### 2.1 Package Verification & Installed Version
- **Package Name**: `behaviorgpt`
- **Installed Version**: `0.11.0` (Verified via `pip show behaviorgpt`)
- **Author**: Unbox AI (`https://unboxai.com/behaviorgpt`)
- **Dependencies**: `httpx`, `pyarrow`, `pydantic`
- **Installation Location**: `.venv/Lib/site-packages/behaviorgpt`
- **Verified Package Exports**:
  ```python
  from behaviorgpt import (
      AsyncUnboxAIClient,
      UnboxAIClient,
      View,
      AddToCart,
      Order,
      Search,
      RemoveFromCart,
      UnboxAIError,
      AuthenticationError,
      BadRequestError,
      RateLimitError,
  )
  ```

### 2.2 Client Architecture & Lifecycle Management
Source: [`src/behavior_lab/sdk_adapter.py`](file:///c:/BehaviorGPT%20Counterfactual%20Sensitivity%20Lab/src/behavior_lab/sdk_adapter.py)
The project provides two adapter layers wrapping the official SDK:
- **`AsyncBehaviorGPTSDKAdapter`**: Native asynchronous adapter wrapping `AsyncUnboxAIClient` for non-blocking execution in FastAPI routes and concurrent pipelines.
- **`BehaviorGPTSDKAdapter`**: Synchronous adapter wrapping `UnboxAIClient` for batch experiment runners and CLI scripts.

#### Application-Level Lifecycle Integration
Source: [`src/behavior_lab/api.py`](file:///c:/BehaviorGPT%20Counterfactual%20Sensitivity%20Lab/src/behavior_lab/api.py#L48-L76)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sdk_adapter = None
    try:
        from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter, is_sdk_available
        if is_sdk_available() and os.environ.get("UNBOXAI_API_KEY"):
            app.state.sdk_adapter = AsyncBehaviorGPTSDKAdapter(market="us")
            logger.info("Application-level AsyncBehaviorGPTSDKAdapter pre-initialized.")
    except Exception as ex:
        logger.warning("Could not pre-initialize AsyncBehaviorGPTSDKAdapter: %s", ex)
    yield
    if getattr(app.state, "sdk_adapter", None) is not None:
        await app.state.sdk_adapter.aclose()
        logger.info("Application-level AsyncBehaviorGPTSDKAdapter closed cleanly.")
```
- Pre-initializes a shared `AsyncBehaviorGPTSDKAdapter` during FastAPI startup.
- Automatically reuses the pooled connection across requests with matching market/catalog parameters.
- Implements `aclose()` during shutdown to prevent socket leaks.
- Verified by unit test `test_api_lifespan_lifecycle`.

### 2.3 Authentication & Credential Isolation
- **Authentication Method**: Bearer token passed to `AsyncUnboxAIClient(api_key=resolved_key)`.
- **Server-Side Enforcement**: `UNBOXAI_API_KEY` is loaded exclusively on the backend via `python-dotenv` from the root `.env` file.
- **Client Security**: No API key is ever transmitted to, stored in, or readable by the browser JavaScript.
- **Graceful Error Handling**: Missing API keys or invalid credentials raise `SDKAuthError` (mapped to `HTTP 401 Unauthorized` or clear diagnostic output), rather than crashing.

### 2.4 Action-to-SDK-Event Mapping
Source: [`src/behavior_lab/sdk_adapter.py`](file:///c:/BehaviorGPT%20Counterfactual%20Sensitivity%20Lab/src/behavior_lab/sdk_adapter.py#L92-L145)
The adapter accurately translates the application's domain events into official SDK `UserEvent` objects:

| Application Action | Official SDK Event | SDK Semantics & Justification |
| :--- | :--- | :--- |
| `ActionType.VIEW_ITEM` | `View(item=pid, timestamp=t)` | Product page view. |
| `ActionType.SEARCH` | `Search(query=q, timestamp=t)` | Text search query intervention. |
| `ActionType.ORDER` | `Order(items=[pid], timestamp=t)` | Purchased conversion item. |
| `ActionType.ADD_TO_CART` | `View(item=pid, timestamp=t)` | **Normalized to `View` by default**. Per BehaviorGPT v0.11.0 docstring: *"AddToCart is deprecated: the model reads it the same as View. Send a View, and list what was bought in Order([...])"*. Normalizing prevents SDK deprecation warnings while preserving the original `addToCart` verb in the application's experiment history provenance. |

### 2.5 Deterministic Timestamp Strategy
Source: [`src/behavior_lab/sdk_adapter.py`](file:///c:/BehaviorGPT%20Counterfactual%20Sensitivity%20Lab/src/behavior_lab/sdk_adapter.py#L60-L90)
To eliminate temporal bias when comparing original vs. counterfactual histories:
- `resolve_base_timestamp(history, base_time)` establishes a shared anchor.
- Priority:
  1. Explicit shared `base_time` (ensures identical session start for paired evaluations).
  2. `history.created_at` ISO-8601 string if present.
  3. Dynamic recent deterministic anchor snapped to current UTC hour (`datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)`).
- Event timestamps are computed deterministically as `base_time + timedelta(milliseconds=action.offset_ms)`.
- Request inference timestamp is set to `base_time + max_offset + 1000ms`, ensuring all events precede the inference point.

### 2.6 Model Score vs. Probability Semantics
Source: [`src/behavior_lab/sdk_adapter.py`](file:///c:/BehaviorGPT%20Counterfactual%20Sensitivity%20Lab/src/behavior_lab/sdk_adapter.py#L147-L177)
- **Technical Reality**: `behaviorgpt` SDK item scores represent **uncalibrated inner-product similarity scores** (dot products between session embedding and item representations in latent space), not calibrated probabilities.
- **Implementation Correctness**:
  - `_sdk_item_to_prediction()` assigns `score = item.score` and `model_score = item.score`.
  - Sets `prob = None`.
  - Disables probability-bound metrics (e.g. Total Variation Distance) on raw dot-product scores.
  - UI labels SDK predictions explicitly as `Score Semantics: Uncalibrated Model Score`.

---

## 3. Feature-by-Feature Audit: The Original Application

The project satisfies Jens's request by building an analytical research platform that explores the behavioral properties of the foundation model.

### 3.1 Tool 1: Behavioral Journey Simulator
- **Route**: `POST /api/journey/simulate`
- **Adapter**: `AsyncBehaviorGPTSDKAdapter.predict()` -> `AsyncUnboxAIClient.complete()`
- **UI Tab**: Tab 1 (`#tab-journey`)
- **Key Capabilities**:
  - Interactive multi-action timeline builder (`viewItem`, `search`, `order`, `addToCart`).
  - Preloaded behavioral persona presets:
    - *Athletic Runner*: Footwear views + query "running shoes" + Dunk Low order.
    - *Tech Enthusiast*: iPad + Galaxy Tab + PS5 DualSense controller views.
    - *Luxury Shopper*: Coach Tabby + Coach Soft Tabby luxury leather goods.
    - *Home Decorator*: Loloi rug view + Yankee Candle view + Keurig coffee order.
  - Configurable parameters: catalog (`retail_catalog_500k`, `hm_fashion`, `amazon_sample`), market (`us`), Top-K (`5, 10, 15`).
  - Measures live inference latency and displays raw uncalibrated model scores.

### 3.2 Tool 2: Counterfactual Playground
- **Route**: `POST /api/counterfactual/compare`
- **Adapter**: `AsyncBehaviorGPTSDKAdapter.predict()` -> `SensitivityMetricsCalculator.compute()`
- **UI Tab**: Tab 2 (`#tab-playground`)
- **Core Research Hypothesis**:
  > *"When exactly one action in a behavioral history is mutated, how does BehaviorGPT's subsequent prediction change?"*
- **Mathematical Rigor in Sensitivity Metrics**:
  - **Top-1 Flip**: Boolean indicator and identifier tracking.
  - **Jaccard Similarity @ K**: Overlap ratio of candidate sets $\frac{|A \cap B|}{|A \cup B|}$ for $K \in \{3, 5, 10\}$.
  - **Shared Rank Displacement**: Average rank movement strictly over items present in both Top-K lists:
    $$\text{MRD}_{\text{shared}} = \frac{1}{|S|} \sum_{i \in S} |\text{rank}_A(i) - \text{rank}_B(i)|$$
  - **Penalized Rank Displacement**: Rigorous metric assigning penalty rank $(K + 1)$ to evicted and newly appearing items:
    $$\text{MRD}_{\text{penalized}} = \frac{1}{|A \cup B|} \left( \sum_{i \in S} |\text{rank}_A(i) - \text{rank}_B(i)| + \sum_{e \in \text{evicted}} |(K+1) - \text{rank}_A(e)| + \sum_{n \in \text{new}} |(K+1) - \text{rank}_B(n)| \right)$$
    *Guarantees completely disjoint lists cannot produce a misleading 0.0 value.*
  - **Item Transition Analysis**: Decomposes the recommendation shift into three explicit categories:
    1. `shared`: Items retained across both lists, displaying rank delta and score delta.
    2. `evicted`: Items displaced from Top-K as a consequence of the mutation.
    3. `new`: Items recruited into Top-K due to the counterfactual intervention.

### 3.3 Tool 3: Similar Item Explorer
- **Routes**: `GET /api/items/{item_id}/similar` and `POST /api/items/similar`
- **Adapter**: `AsyncBehaviorGPTSDKAdapter.similar_items()` -> `AsyncUnboxAIClient.similar_items()`
- **UI Tab**: Tab 3 (`#tab-similar`)
- **Technical Value**:
  - Explores BehaviorGPT's underlying item embedding manifold.
  - Shows catalog nearest neighbors with inner product similarity scores.
  - Directly demonstrates the SDK's catalog-grounded embedding capabilities.

### 3.4 Baseline Benchmark & Investigated Case Studies
- **UI Tabs**: Tab 4 (Paired Case Inspector), Tab 5 (Experimental Breakdowns), Tab 6 (Investigated Cases).
- **Benchmark Corpus**: 40 paired experiment cases across 4 mutation archetypes:
  - *Cross-Category Swap*: 80.0% Top-1 flip rate, 28.8% average Jaccard.
  - *Within-Category Swap*: 70.0% Top-1 flip rate, 39.5% average Jaccard.
  - *Action-Verb Swap (view vs. cart)*: 10.0% Top-1 flip rate, 90.0% average Jaccard (demonstrates item identity dominance over verb).
  - *Early Position Mutation (Index 0)*: 100.0% Top-1 flip rate (contextual anchor effect).
- **Isolation**: Historical benchmark data is preserved in `data/experiment_results.jsonl` and clearly labeled with `Historical Record — October 15, 2026`.

---

## 4. Audit of the Showcase Sandbox

### 4.1 Technical Profile
- **Location**: Tab 7 (`#tab-sandbox`)
- **Route**: `POST /api/run-interactive`
- **Upstream Target**: `https://behaviorgpt.unboxai.com/api/recommendations` (Express.js public showcase endpoint)
- **Client**: `BehaviorGPTClient` in `src/behavior_lab/client.py` (via `httpx.Client`)
- **Input Format**: Simple newline-delimited product IDs.
- **Output**: Calibrated item probabilities (`prob`) from the public demo server.

### 4.2 Comparison with SDK-Backed Tools
| Dimension | Showcase Sandbox (Tab 7) | Counterfactual Playground (Tab 2) |
| :--- | :--- | :--- |
| **Gateway** | Public web endpoint (`unboxai.com/api/recommendations`) | Official Python SDK (`behaviorgpt` / enterprise gateway) |
| **Authentication** | Unauthenticated public showcase | Bearer Token (`UNBOXAI_API_KEY`) |
| **Event Expressiveness** | Product ID lists only | Multi-action sequences (`viewItem`, `search`, `order`, `cart`) |
| **Score Semantics** | Calibrated probability (`prob`) | Raw dot-product score (`model_score`) |
| **Catalog Grounding** | Fixed showcase index | Multi-catalog support (`retail_catalog_500k`, etc.) |
| **Transitions** | None | Full item transition analysis (shared, evicted, new) |

### 4.3 Why It Must Remain Separate
1. **Technical Integrity**: The public showcase endpoint and the official SDK gateway are completely different backends with different authentication, protocols, and score semantics. Merging them would obscure this reality.
2. **Historical Benchmark Continuity**: The October 15, 2026 research benchmark was evaluated against the showcase endpoint. Retaining the sandbox allows continuous verification of that baseline.
3. **Educational Value for Jens**: During the discussion with Jens, having both demonstrates that you understand the architectural distinction between Unbox AI's consumer-facing marketing demo and their developer SDK.

---

## 5. Security Audit

| Checkpoint | Audit Result | Evidence |
| :--- | :--- | :--- |
| **API Key Storage** | **SECURE** | Loaded server-side via `os.environ.get("UNBOXAI_API_KEY")`. |
| **Git Exclusions** | **SECURE** | `.env`, `.venv/`, `*.log`, and `data/sdk_results/` are ignored in `.gitignore`. `git status --ignored` confirms `.env` is untracked. |
| **Hardcoded Secrets** | **CLEAN** | Repository-wide regex audit (`git grep -i "ubx_"`) returns zero hardcoded credentials in tracked files. |
| **Frontend Isolation** | **SECURE** | `web/index.html` contains no tokens, headers, or client-side secrets. All calls route to local FastAPI endpoints. |
| **Offline Resilience** | **VERIFIED** | Both the SDK tools and the showcase sandbox provide full `Offline Mock Mode` for air-gapped demonstration. |

---

## 6. Architecture & Single-Origin Verification

### The Static-Server (Port 5500) Issue and Final Resolution
- **Symptom**: Refreshing at `http://127.0.0.1:5500` produced `HTTP 501: Unsupported method ('POST')`.
- **Root Cause**: The frontend was being served by Python's static `http.server` on port 5500, which has no `do_POST` implementation.
- **Architectural Fix**:
  1. **FastAPI Single-Origin Delivery**: The backend at `http://127.0.0.1:8000` serves `web/index.html` at `GET /` and `GET /index.html`.
  2. **Automatic Origin Resolver (`getApiBase()`)**: If a developer accidentally opens `http://127.0.0.1:5500`, JavaScript automatically redirects API calls to `http://127.0.0.1:8000` via CORS, and displays an on-screen warning banner directing them to port 8000.
  3. **Showcase Direct Wire Format**: Corrected `events` wire format to `{ type: 'viewItem', product: pid }`, matching upstream discriminator requirements.

### Live Host Probing Verification
Probed live host endpoints on ports 8000 and 5500:
```text
GET  http://127.0.0.1:8000/                    -> Status 200 OK (Dashboard HTML)
GET  http://127.0.0.1:8000/health              -> Status 200 OK ({"status": "ok"})
POST http://127.0.0.1:8000/api/run-interactive -> Status 200 OK (Live Inference: 5 items, metrics computed)
POST http://127.0.0.1:8000/api/journey/simulate -> Status 200 OK (Simulator verified)
POST http://127.0.0.1:8000/api/counterfactual/compare -> Status 200 OK (Playground verified)
GET  http://127.0.0.1:8000/api/items/paB0D93VKPPK/similar -> Status 200 OK (Similar Items verified)
GET  http://127.0.0.1:5500/index.html           -> Status 200 OK (Serves upgraded script with 8000 routing)
```

---

## 7. Full Regression Test Suite Results

Command: `.\.venv\Scripts\python.exe -m pytest tests/ -v`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-8.2.2, pluggy-1.6.0
rootdir: C:\BehaviorGPT Counterfactual Sensitivity Lab
configfile: pyproject.toml
plugins: anyio-4.16.0
collected 66 items

tests/test_api.py::test_api_summary PASSED                               [  1%]
tests/test_api.py::test_api_cases PASSED                                 [  3%]
tests/test_api.py::test_api_case_detail PASSED                           [  4%]
tests/test_api.py::test_api_case_detail_not_found PASSED                 [  6%]
tests/test_api.py::test_api_investigation PASSED                         [  7%]
tests/test_api.py::test_dashboard_ui PASSED                              [  9%]
tests/test_api.py::test_dashboard_ui_stage_b2_sections PASSED            [ 10%]
tests/test_api.py::test_run_interactive_success PASSED                   [ 12%]
tests/test_api.py::test_run_interactive_first_sequence_failure PASSED    [ 13%]
tests/test_api.py::test_run_interactive_second_sequence_failure PASSED   [ 15%]
tests/test_api.py::test_ui_handles_unsuccessful_http_response PASSED     [ 16%]
tests/test_api.py::test_ui_handles_upstream_inference_error PASSED       [ 18%]
tests/test_api.py::test_ui_distinct_labels_for_historical_and_live PASSED [ 19%]
tests/test_api.py::test_api_summary_unverified_metadata PASSED           [ 21%]
tests/test_api.py::test_run_interactive_sdk_mode_uses_async_adapter PASSED [ 22%]
tests/test_api.py::test_api_lifespan_lifecycle PASSED                    [ 24%]
tests/test_api.py::test_api_catalog_items PASSED                         [ 25%]
tests/test_api.py::test_journey_simulate_validation PASSED               [ 27%]
tests/test_api.py::test_journey_simulate_mock_mode PASSED                [ 28%]
tests/test_api.py::test_journey_simulate_sdk_mode PASSED                 [ 30%]
tests/test_api.py::test_journey_simulate_upstream_error PASSED           [ 31%]
tests/test_api.py::test_counterfactual_compare_mock_mode PASSED          [ 33%]
tests/test_api.py::test_counterfactual_compare_sdk_mode_with_transitions PASSED [ 34%]
tests/test_api.py::test_similar_items_mock_and_post PASSED               [ 36%]
tests/test_api.py::test_similar_items_sdk_mode PASSED                    [ 37%]
tests/test_api.py::test_similar_items_sdk_not_found_404 PASSED           [ 39%]
tests/test_api.py::test_showcase_correct_http_method_and_payload PASSED  [ 40%]
tests/test_api.py::test_showcase_successful_response_parsing PASSED      [ 42%]
tests/test_api.py::test_showcase_http_4xx_and_5xx_responses[400] PASSED  [ 43%]
tests/test_api.py::test_showcase_http_4xx_and_5xx_responses[401] PASSED  [ 45%]
tests/test_api.py::test_showcase_http_4xx_and_5xx_responses[404] PASSED  [ 46%]
tests/test_api.py::test_showcase_http_4xx_and_5xx_responses[500] PASSED  [ 48%]
tests/test_api.py::test_showcase_http_4xx_and_5xx_responses[502] PASSED  [ 50%]
tests/test_api.py::test_showcase_http_4xx_and_5xx_responses[503] PASSED  [ 51%]
tests/test_api.py::test_showcase_unsupported_method_response_501 PASSED  [ 53%]
tests/test_api.py::test_showcase_network_timeout_and_connection_failure PASSED [ 54%]
tests/test_api.py::test_showcase_invalid_or_empty_response PASSED        [ 56%]
tests/test_api.py::test_showcase_interactive_mock_mode PASSED            [ 57%]
tests/test_api.py::test_showcase_frontend_diagnostics_and_error_states PASSED [ 59%]
tests/test_api.py::test_regression_protection_sdk_backed_tools PASSED    [ 60%]
tests/test_api.py::test_single_origin_architecture PASSED                [ 62%]
tests/test_lab_suite.py::test_synthetic_dataset_determinism PASSED       [ 63%]
tests/test_lab_suite.py::test_different_seeds_produce_different_histories PASSED [ 65%]
tests/test_lab_suite.py::test_mutation_exactly_one_action_changed PASSED [ 66%]
tests/test_lab_suite.py::test_metric_calculations_known_analytical_cases PASSED [ 68%]
tests/test_lab_suite.py::test_mock_client_inference PASSED               [ 69%]
tests/test_lab_suite.py::test_error_handling_invalid_endpoint PASSED     [ 71%]
tests/test_lab_suite.py::test_jsonl_serialization_round_trip PASSED      [ 72%]
tests/test_sdk_adapter.py::test_view_action_converts_to_sdk_view PASSED  [ 74%]
tests/test_sdk_adapter.py::test_add_to_cart_normalized_to_view_by_default PASSED [ 75%]
tests/test_sdk_adapter.py::test_add_to_cart_legacy_mapping_isolated PASSED [ 77%]
tests/test_sdk_adapter.py::test_resolve_base_timestamp_precedence PASSED [ 78%]
tests/test_sdk_adapter.py::test_order_action_converts_to_sdk_order PASSED [ 80%]
tests/test_sdk_adapter.py::test_search_converts_to_sdk_search PASSED     [ 81%]
tests/test_sdk_adapter.py::test_deterministic_timestamps_in_history_conversion PASSED [ 83%]
tests/test_sdk_adapter.py::test_sdk_item_to_prediction_uncalibrated_score PASSED [ 84%]
tests/test_sdk_adapter.py::test_sdk_item_to_prediction_missing_fields PASSED [ 86%]
tests/test_sdk_adapter.py::test_sdk_item_none_keywords_and_top_terms PASSED [ 87%]
tests/test_sdk_adapter.py::test_disjoint_lists_rank_displacement_not_misleading_zero PASSED [ 89%]
tests/test_sdk_adapter.py::test_partially_overlapping_lists_displacement_and_score_l1 PASSED [ 90%]
tests/test_sdk_adapter.py::test_within_category_mutation_strict_category_matching PASSED [ 92%]
tests/test_sdk_adapter.py::test_within_category_mutation_rejects_missing_metadata PASSED [ 93%]
tests/test_sdk_adapter.py::test_sdk_adapter_default_catalog PASSED       [ 95%]
tests/test_sdk_adapter.py::test_sdk_adapter_similar_items PASSED         [ 96%]
tests/test_sdk_adapter.py::test_async_sdk_adapter_lifecycle PASSED       [ 98%]
tests/test_sdk_adapter.py::test_runner_data_dir_isolation PASSED         [100%]

============================= 66 passed in 6.95s ==============================
```

---

## 8. Technical Discussion Points for Jens Palmborg

Jens explicitly asked: *"and we can follow up with what you think technically we can develop further on this model."*

Here are five concrete, high-impact technical proposals grounded directly in the SDK integration experience:

### 1. Token-Level Attention & Event Attribution in SDK
- **Problem**: When a recommendation list shifts, developers cannot programmatically determine *which* past event in a 10-action session drove the change.
- **Proposal**: Add an optional `include_attributions=True` parameter to `complete()`, returning an attention attribution vector across input event tokens. This would provide explainability for behavioral foundation models.

### 2. Built-in Counterfactual Perturbation / Sensitivity API
- **Problem**: Measuring sensitivity currently requires building custom client-side orchestration, paired request dispatch, and transition calculations.
- **Proposal**: Expose a first-class SDK method:
  ```python
  client.compare(
      history_a=session_a,
      history_b=session_b,
      metrics=["jaccard", "rank_displacement", "item_transitions"]
  )
  ```
  Executing this server-side would allow Unbox AI to reuse cached key-value states for shared prefixes, dramatically cutting inference latency for counterfactual pairs.

### 3. Sampling Diversity & Temperature Controls in `complete()`
- **Problem**: Current SDK inference is strictly deterministic top-K retrieval by inner product.
- **Proposal**: Introduce `temperature: float = 1.0` and `top_p: float = 0.9` parameters to allow controlled stochastic exploration for cold-start or exploratory recommendation modes.

### 4. Optional Score Calibration Head
- **Problem**: Raw inner product scores cannot be directly compared across different catalog categories or converted to conversion probabilities without custom client calibration.
- **Proposal**: Provide a calibrated probability option `return_probabilities=True` (e.g. via Platt scaling or temperature-calibrated softmax over candidate subsets) alongside raw dot-product scores.

### 5. Dynamic Embedding Projection in `similar_items()`
- **Problem**: `similar_items()` requires items to exist in the pre-indexed catalog.
- **Proposal**: Allow ad-hoc item feature projection (`client.similar_items(item_features={...})`), enabling instant embedding retrieval for newly listed products before full catalog re-indexing.

---

## 9. Final Pre-Demo Readiness Checklist

- [x] **SDK Installed & Functional**: `behaviorgpt 0.11.0` verified.
- [x] **Async Adapter Operational**: `AsyncBehaviorGPTSDKAdapter` tested with real live inference.
- [x] **Three Core SDK Tools Verified**:
  - Journey Simulator (`POST /api/journey/simulate`)
  - Counterfactual Playground (`POST /api/counterfactual/compare`)
  - Similar Item Explorer (`GET /api/items/{id}/similar`)
- [x] **Single-Origin Application Verified**: Hosted at `http://127.0.0.1:8000`.
- [x] **Static Server (Port 5500) Protected**: Automatic redirection to port 8000 prevents HTTP 501.
- [x] **Showcase Sandbox Demarcated**: Clearly identified as public REST showcase baseline, isolated in Tab 7.
- [x] **Credentials Isolated**: No API keys in git, frontend, or public logs.
- [x] **Regression Suite Clean**: 66/66 passing tests.
- [x] **Git Cleanliness**: No uncommitted migrations, no deletions of historical benchmark data.

**Verdict: The project is technically robust, scientifically sound, and fully prepared for demonstration to Jens Palmborg.**
