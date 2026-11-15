# Technical Audit & Gap Analysis: Behavioral Decision Explorer

**Date:** 2026-11-12  
**Role:** Principal Backend Engineer, Senior ML Engineer, AI Research Scientist, Python SDK Integration Architect  
**Target:** Upgrade BehaviorGPT Counterfactual Sensitivity Lab $\to$ Behavioral Decision Explorer  

---

## 1. Executive Summary & Verification

| Dimension | Previous State | Audited Current State | Gap / Target |
|---|---|---|---|
| **SDK Version** | v0.1.0 (local commit `8916c99`) | **v0.11.0** (updated from `origin/main`) | Unlocks `AsyncUnboxAIClient`, `ExternalItem`, `similar_items()`, and modern `Order` |
| **Client Types** | Synchronous `UnboxAIClient` only | Both `UnboxAIClient` & `AsyncUnboxAIClient` present in `behaviorgpt` | Use `AsyncUnboxAIClient` in FastAPI to prevent blocking event loop |
| **Catalog Target** | Defaulted to demo `retail_catalog` | **`retail_catalog_500k`** | Required by account credentials (`UNBOXAI_API_KEY`) |
| **Event Semantics** | `AddToCart` used directly | `AddToCart` deprecated by Unbox AI (model reads as `View`) | Support `View`, `Search`, `Order([item, ...])`; document `AddToCart` equivalence |
| **Similar Items** | Not exposed in UI / API | Supported via `client.similar_items(item_id, limit, catalog_id)` | Build Similar Item Explorer tab & API endpoint |
| **Interactive UI** | 2-history text comparison (Showcase API) | Needs interactive builder for `Search`, `View`, `Order` with live SDK | Build Behavioral Journey Simulator & Counterfactual Playground |
| **Regression Suite** | 31 unit tests passing | **33 unit tests passing** | Extend to cover Async adapter, Journey Simulator, Similar Items |

---

## 2. Official SDK v0.11.0 Deep Dive & Method Signatures

### 2.1 Client Classes: `UnboxAIClient` vs. `AsyncUnboxAIClient`

Both clients inherit from `_BaseClient` in `behaviorgpt._client`:

```python
class UnboxAIClient(_BaseClient):
    def __init__(
        self,
        market: str,
        *,
        timezone: str = "UTC",
        api_key: Optional[str] = None,
        base_url: str = "https://behaviorgpt-northeurope.api.unboxai.com/v1",
        default_catalog_id: str = "retail_catalog",
        http_timeout: float | httpx.Timeout = DEFAULT_HTTP_TIMEOUT,
    ): ...

    def complete(
        self,
        history: UserHistoryInput = (),
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        register_event: bool = False,
    ) -> UnboxAIResponse: ...

    def similar_items(
        self,
        item_id: ItemRef,  # Union[str, Item, ExternalItem]
        limit: int = 10,
        offset: int = 0,
        *,
        catalog_id: Optional[str] = None,
    ) -> UnboxAIResponse: ...
```

`AsyncUnboxAIClient` provides the exact asynchronous equivalent for all methods:
```python
class AsyncUnboxAIClient(_BaseClient):
    async def complete(...) -> UnboxAIResponse: ...
    async def similar_items(...) -> UnboxAIResponse: ...
    async def aclose(): ...
```
> **FastAPI Recommendation:** Replace blocking synchronous `client.predict()` in FastAPI route handlers with an async adapter backed by `AsyncUnboxAIClient` or connection-pooled client.

---

### 2.2 Event Types & Semantics

1. **`Search(query: str)`**:
   * Generates a text search intent.
   * If `Search` is the terminal event in a history, the SDK routes the query to `/vector/search`.
2. **`View(item: str | ExternalItem)`**:
   * Logs a product interaction.
   * If the terminal event is `View`, the SDK routes to `/vector/recommendations`.
3. **`Order(items: Sequence[OrderItem], cart_id: Optional[str])`**:
   * Signifies a completed conversion.
   * In SDK v0.11.0, `Order` accepts a sequence of items: `Order(["item_a", "item_b"])`.
4. **`AddToCart(item: str)`**:
   * Deprecated in the SDK source code:
     > *"AddToCart is deprecated and will be removed in 0.7: the model reads it the same as View. Send a View, and list what was bought in Order([...])"*
   * **Crucial Scientific Alignment:** Our Phase 0 counterfactual experiments empirically demonstrated a **90.0% Top-1 invariance** when swapping `viewItem` $\leftrightarrow$ `addToCart`. The official SDK code confirms this exact architectural decision.
5. **`ExternalItem`**:
   * Allows submitting an ad-hoc product not in the catalog via text or image URL:
     `ExternalItem(name="Linen shirt", brand="Acme", categories="Clothing, Shirts")`

---

## 3. Product Development Plan: Behavioral Decision Explorer

### Feature 1: Behavioral Journey Simulator
* **Interactive timeline builder**: Add `Search(query)`, `View(item_id)`, and `Order([item_ids])` step-by-step.
* **Live Catalog Autocomplete / Validation**: Check item IDs against known catalog items (`retail_catalog_500k`).
* **Inference Execution**: Submit sequence to BehaviorGPT via official SDK.
* **Rich Output**: Display predicted Top-K items with names, brands, categories, prices, and model confidence scores.

### Feature 2: Counterfactual Playground
* **Strict Invariant Comparison**:
  $$\text{Original Sequence} \quad \text{vs.} \quad \text{Mutated Sequence (exactly 1 change)}$$
* **Perturbation Modes**: Query change, viewed item change, event position swap, sequence truncation.
* **Differential Metrics**: Real-time calculation of Top-1 Flip, Jaccard@5, Jaccard@10, Mean Rank Displacement.

### Feature 3: Similar Item Explorer
* Invoke official SDK `client.similar_items(item_id, limit=5, catalog_id="retail_catalog_500k")`.
* Display nearest neighbors in the catalog embedding space.
* Compare behavioral recommendations (context-dependent) vs. item-to-item similarity (context-independent).

### Feature 4: Behavioral Stability Research Engine
* Batch experiment runner supporting systematic sensitivity sweeps across:
  * Sequence length $L \in \{2, 4, 8, 12\}$
  * Mutation position (Early, Middle, Late)
  * Event semantic type (Query vs. Item vs. Purchase)
* Outputs isolated to `data/sdk_results/` without mutating historical showcase data.

### Feature 5: Research Dashboard Upgrade
* Upgrade existing `web/index.html`:
  * Add **Journey Simulator** tab.
  * Add **Similar Items** comparison panel.
  * Clearly label provenance tags (`provider: behaviorgpt_sdk` vs. `showcase` vs. `mock`).
  * Display real-time SDK connection status.
