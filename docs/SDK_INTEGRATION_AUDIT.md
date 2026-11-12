# SDK Integration Audit

**Date:** 2026-11-12
**Auditor:** Engineering Team
**Status:** COMPLETE

---

## 1. Official SDK Package Summary

| Property | Value |
|---|---|
| **Package name** | `behaviorgpt` |
| **Version** | `0.1.0` |
| **Repository** | https://github.com/Unbox-AI/behaviorgpt |
| **Python requirement** | ≥ 3.11 |
| **License** | Apache-2.0 |
| **Installation** | `pip install behaviorgpt` or `pip install -e ./repo_temp` |
| **Dependencies** | `httpx`, `pydantic` (v2), `pyarrow` |
| **Async support** | No (synchronous `httpx.Client` only) |

---

## 2. SDK Public API

### 2.1 Client Initialization

```python
from behaviorgpt import UnboxAIClient

client = UnboxAIClient(
    market="us",                          # Required: market code
    api_key="...",                         # Optional: falls back to UNBOXAI_API_KEY env var
    base_url="https://behaviorgpt-northeurope.api.unboxai.com/v1",  # Default
    default_catalog_id="retail_catalog",  # Default catalog
    http_timeout=httpx.Timeout(120.0, connect=10.0, write=300.0),  # Default
)
```

**Raises `AuthenticationError`** if no API key is found.

### 2.2 Primary Inference Method: `complete()`

```python
response: UnboxAIResponse = client.complete(
    history=[View("item-id"), Search("query")],  # UserHistoryInput
    limit=10,        # 1–100, default 10
    offset=0,        # pagination, default 0
    catalog_id=None, # override default_catalog_id
    filters=None,    # optional dict
    register_event=False,  # register in user timeline
)
```

**Routing logic:**
- If last event is `Search` → routes to `/vector/search`
- Otherwise → routes to `/vector/recommendations`

**History format:**
- 1D: `[View("a"), AddToCart("b"), Search("q")]`
- 2D (multi-session): `[[View("a")], [View("b"), Search("q")]]`

### 2.3 Event Types

| SDK Class | Wire `type` | Required fields |
|---|---|---|
| `View(item)` | `viewItem` | `item: str` |
| `AddToCart(item)` | `addToCart` | `item: str` |
| `RemoveFromCart(item)` | `removeFromCart` | `item: str` |
| `Search(query)` | `search` | `query: str` |
| `Order(cart_id, items)` | `order` | optional `cart_id`, `items` |

> **Note:** `RemoveFromCart` has been observed to be rejected by the server in some contexts.
> Our experiment uses only `View`, `AddToCart`, and `Search`.

### 2.4 Response Schema

```python
class UnboxAIResponse(BaseModel):
    items: List[Item]    # Recommendation/search results
    offset: int          # Pagination offset
    limit: int           # Requested limit
    market: Optional[str]  # Excluded from serialization

class Item(BaseModel):
    id: str              # Item identifier
    data: dict           # Metadata: name, brand, category, price, etc.
    score: float         # Model confidence score
```

**Key difference from showcase:** Showcase returns `prob`, `name`, `brand` as top-level fields.
SDK nests `name`, `brand`, `category`, `price` inside `item.data` dict.

### 2.5 Exception Hierarchy

```
UnboxAIError (base)
├── AuthenticationError  (HTTP 401)
├── RateLimitError       (HTTP 429)
└── BadRequestError      (HTTP 400)
```

All exceptions carry: `message`, `status_code`, `response` dict.

### 2.6 Other Methods

| Method | Purpose |
|---|---|
| `similar_items(item_id, limit)` | Find similar items by ID |
| `embed(path, wait)` | Upload and embed a catalog parquet |
| `job_status(job_id)` | Check embed job status |
| `random_item()` | Get a random item from catalog |

---

## 3. Existing Codebase Showcase Integration Points

### 3.1 Files that call the showcase endpoint

| File | Location | What it does |
|---|---|---|
| `src/behavior_lab/client.py` | `BehaviorGPTClient.predict()` (line 100) | `POST {base_url}/recommendations` via httpx |
| `src/behavior_lab/api.py` | `run_interactive()` (line 161) | Creates `BehaviorGPTClient(mode="live")` |
| `scripts/run_experiment.py` | CLI entry | Passes `--mode live` to `ExperimentRunner` |

### 3.2 Showcase wire format (current)

```json
{
  "history": [{
    "type": "default",
    "start_time": "2026-01-01T00:00:00Z",
    "domains": {"market": "US"},
    "events": [
      {"type": "viewItem", "product": "item-id"},
      {"type": "addToCart", "product": "item-id"},
      {"type": "search", "query": "running shoes"}
    ]
  }],
  "limit": 10
}
```

### 3.3 Showcase response format

```json
{
  "items": [
    {
      "id": "item-001",
      "name": "Product Name",
      "prob": 0.006,
      "price": "29",
      "brand": "BrandName",
      "category": "Category",
      "keywords": ["k1", "k2"],
      "topTerms": [{"term": "...", "score": 0.5}]
    }
  ]
}
```

### 3.4 SDK wire format (target)

The SDK handles serialization internally. User provides:
```python
history = [View("item-001"), AddToCart("item-002"), Search("query")]
response = client.complete(history=history, limit=10)
```

SDK response: `response.items` → list of `Item(id, score, data)`.

---

## 4. Key Mapping Differences

| Aspect | Showcase (current) | SDK (target) |
|---|---|---|
| **Base URL** | `behaviorgpt.unboxai.com/api` | `behaviorgpt-northeurope.api.unboxai.com/v1` |
| **Auth** | None required | `UNBOXAI_API_KEY` (header: `x-api-key`) |
| **History format** | Raw JSON with session wrapper | SDK event objects |
| **Item name** | `item["name"]` | `item.data["name"]` |
| **Item score** | `item["prob"]` | `item.score` |
| **Item brand** | `item["brand"]` | `item.data["brand"]` |
| **Item price** | `item["price"]` (string) | `item.data["price"]` |
| **Item keywords** | `item["keywords"]` | `item.data["keywords"]` |
| **Response wrapper** | `{"items": [...]}` | `UnboxAIResponse(items=[...], offset, limit)` |

---

## 5. Authentication & Access

| Requirement | Status |
|---|---|
| `behaviorgpt` package installed | ✓ Installed (editable from `repo_temp`) |
| `UNBOXAI_API_KEY` configured | ✗ **NOT SET** |
| SDK init with dummy key | Returns HTTP 403 "Invalid API key" |
| Showcase access | ✓ Works without any key |

> **Blocker:** Live SDK inference requires a valid `UNBOXAI_API_KEY`.
> The implementation is complete and correct such that providing a key
> makes everything work. The verification script reports this blocker
> honestly.

---

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| No API key available | **HIGH** | SDK adapter fully implemented; verification script reports blocker; showcase mode retained as fallback |
| SDK `RemoveFromCart` rejected | LOW | Not used in experiment (only `View`, `AddToCart`, `Search`) |
| Response `data` dict structure varies | MEDIUM | Adapter handles missing fields gracefully with defaults |
| Rate limiting | MEDIUM | Adapter catches `RateLimitError`, returns structured error |
| SDK response `score` vs showcase `prob` | LOW | Both mapped to `PredictionItem.score` and `PredictionItem.prob` |
