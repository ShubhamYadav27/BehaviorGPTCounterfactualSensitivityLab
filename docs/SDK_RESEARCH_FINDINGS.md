# BehaviorGPT Official SDK Research Findings & Empirical Audit

**Date:** 2026-11-15  
**Auditor:** Principal Backend Engineer & Senior ML Research Scientist  
**Installed SDK Source:** `behaviorgpt` v0.11.0 (`repo_temp/src/behaviorgpt`)  
**Status:** COMPLETE (Stage A Audit)  

---

## 1. Official SDK Event Semantics & Architectural Equivalence

### 1.1 The `AddToCart` Deprecation & Empirical Invariance Validation
A central finding in our historical showcase research was that changing *only* the user action verb from `viewItem` to `addToCart` on the same item (`case_003`, `case_007`, etc.) produced a **90.0% Top-1 invariance rate** and a **90.0% Top-5 candidate overlap**.

Inspection of the official BehaviorGPT SDK source code (`src/behaviorgpt/types/events.py`, Lines 160–172) reveals the exact architectural reason:

```python
class AddToCart(_ItemEvent, _Timed):
    type: Literal["addToCart"] = "addToCart"

    def __init__(self, item: Optional[ItemRef] = None, **data):
        warnings.warn(
            "`AddToCart` is deprecated and will be removed in 0.7: the model reads "
            "it the same as `View`. Send a `View`, and list what was bought in "
            "`Order([...])`",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(item, **data)
```

**Key Scientific Takeaway for Jens & Unbox AI**:
* The model's embedding encoder does not assign distinct behavioral weights to cart additions versus item views. Both map to the same internal state representation.
* Our empirical experiment independently discovered and verified this model property through external counterfactual perturbation, prior to inspecting the deprecation warning in the SDK v0.11.0 source code.

### 1.2 Modern `Order` Event Semantics
In SDK v0.11.0, the primary conversion event is `Order`:
```python
class Order(_Timed):
    type: Literal["order"] = "order"
    items: Optional[List[CartItem]] = None

    def __init__(self, items: Optional[Union[str, Sequence[OrderItem]]] = None, ...):
        ...
```
* `Order` accepts a sequence of purchased items: `Order(["paB0DD21212N", "paB0D93VKPPK"])`.
* Quantities and cart IDs are ignored by the model; the model reads *which* items were converted.
* In our Stage B roadmap, the Behavioral Journey Simulator must support `Order([items])` as the terminal conversion event.

### 1.3 `ExternalItem` Capabilities
SDK v0.11.0 introduces `ExternalItem`:
```python
class ExternalItem(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    categories: Optional[str] = None
    image: Optional[str] = None  # https URL or data URI
```
* Allows querying BehaviorGPT with cold-start products that do not exist in the pre-embedded catalog.
* Enables multimodal conditioning (text title or product image bytes).

---

## 2. Request Equivalence & Temporal Anchoring

In counterfactual sensitivity research, input equivalence must be strictly preserved across all non-mutated parameters:

$$\Delta = f(x_{\text{mut}}, \theta) - f(x_{\text{orig}}, \theta)$$

### 2.1 The Implicit Timestamp Drift Issue
In `behaviorgpt._client.py` (`__from_sequence`, Lines 500–550):
* When events lack explicit `timestamp` fields, the SDK auto-stamps events to end immediately before the request execution time:
  $$\text{event\_time} = \text{now}() - \Delta t$$
* When `orig_out = provider.predict(orig)` executes, its events are stamped relative to $T_0$.
* When `mut_out = provider.predict(mut)` executes ~1,000ms later, its events are stamped relative to $T_0 + 1000\text{ms}$.
* Although a 1-second drift has minimal impact on short sequences, from a strict scientific perspective, non-mutated actions must carry **identical absolute or relative timestamps**.

### 2.2 Recommendation for Stage B:
Pass explicit `timestamp=base_time + timedelta(milliseconds=action.offset_ms)` when constructing SDK `View`, `Search`, and `Order` instances to ensure 100% temporal invariance between paired runs.

---

## 3. Catalog Configuration & Category Consistency Audit

### 3.1 Catalog Identifier
* The official SDK documentation uses `"retail_catalog"` as a generic example.
* In production with `UNBOXAI_API_KEY`, attempting to query `retail_catalog` fails with `HTTP 400: unknown catalog: retail_catalog`.
* The authenticated production catalog for our account is **`retail_catalog_500k`**.
* All adapters, verification scripts, and experiments must target `retail_catalog_500k`.

### 3.2 Category Mutation Leakage in Historical Experiments
An audit of `src/behavior_lab/mutator.py` revealed a subtle methodological bug in historical case generation:
* Lines 35–40 defined `within_category_pool` with only 4 mappings:
  ```python
  self.within_category_pool = {
      NIKE_PANTS: NIKE_PRO_SHORTS,
      NIKE_JORDAN4_BRED: NIKE_DUNK_LOW,
      APPLE_IPAD_109: SAMSUNG_GALAXY_TAB,
      COACH_TABBY_BAG: NIKE_AIR_JORDAN_RETRO,  # BUG: Luxury Bag -> Footwear
  }
  ```
* In Lines 124–134, if an item was missing from `within_category_pool`, the mutator fell back to:
  ```python
  candidates = [NIKE_DUNK_LOW, NIKE_JORDAN4_BRED, NIKE_PANTS, UNDER_ARMOUR_CHARGED, APPLE_IPAD_109]
  replacement_id = next(c for c in candidates if c != target_orig.product_id)
  ```
* **Consequence in Historical Data**:
  * In `case_014`, a **Keurig Coffee Maker** (Home & Kitchen) was mutated to a **Nike Dunk Low** (Footwear) under the label `within_category_swap`.
  * In `case_034`, a **Loloi Living Room Rug** (Home & Kitchen) was mutated to a **Nike Dunk Low** (Footwear) under the label `within_category_swap`.
  * In `case_022`, a **PS5 Gaming Controller** was mutated to a **Nike Dunk Low** under the label `within_category_swap`.
* **Scientific Impact**:
  * This leakage explains why historical "within-category" swaps showed an unexpectedly high **70.0% Top-1 flip rate**. Half of the cases were actually cross-category swaps!
  * In Stage B, category-consistent mutations must be strictly enforced by querying the item's verified category metadata.

---

## 4. Score Calibration: Model Score vs. Probability

| Dimension | Showcase API (Historical) | Official SDK v0.11.0 (Live) |
|---|---|---|
| **JSON Field** | `"prob": 0.0062...` | `"score": 1.2469...` |
| **Domain** | $[0, 1]$, approximately summing to $\le 1.0$ | Unbounded real values ($\ge 1.0$ observed) |
| **Nature** | Softmax calibrated probability estimate | Uncalibrated inner-product / logit score |
| **Statistical Validity** | Total Variation Distance valid | TVD invalid on uncalibrated scores |

* In `sdk_adapter.py`, assigning `prob = item.score` created invalid probabilities $> 1.0$.
* All SDK interfaces must expose `model_score: float`.
* Probability-based metrics (e.g. `prob_tvd_shared`, `top1_prob_delta`) must be omitted or clearly flagged as score deltas (`top1_score_delta`).

---

## 5. Summary of Stage B1 Implementation Status

1. **Rank Displacement Metric**: Fixed in `src/behavior_lab/metrics.py`. Returns `None` for disjoint lists (`rank_displacement_shared`), and calculates `rank_displacement_penalized` with penalty rank $K+1$ for total list turnover.
2. **Strict Category Validation**: Enforced in `src/behavior_lab/mutator.py`. Added verified category siblings for all categories (`XBOX_WIRELESS_CONTROLLER` for Gaming, `COACH_SOFT_TABBY` for Luxury Bags). Strict lookup ensures siblings are from the exact same category and raises `ValueError` if unavailable.
3. **Explicit Timestamp Anchoring**: Implemented in `src/behavior_lab/sdk_adapter.py`. Uses `DEFAULT_REFERENCE_TIMESTAMP` (2026-01-01T12:00:00Z) plus action `offset_ms` to guarantee 100% temporal invariance between paired requests.
4. **Honest Model Score Labeling**: `_sdk_item_to_prediction` stores `item.score` in `model_score` and `score`, leaving `prob = None`. Metrics compute `score_l1_shared` and `top1_score_delta`, bypassing TVD when scores are uncalibrated.
5. **Separation of Experiment Datasets**: `ExperimentRunner(mode="sdk")` defaults to `data/sdk_results/`, safeguarding historical showcase benchmarks in `data/`.
6. **Native Async Adapter**: Implemented `AsyncBehaviorGPTSDKAdapter` wrapping `AsyncUnboxAIClient` with clean `aclose()` and `async with` context manager support.
