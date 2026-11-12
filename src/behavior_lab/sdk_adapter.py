"""SDK adapter for official BehaviorGPT Python package.

Provides a clean adapter layer between the official `behaviorgpt` SDK
(https://github.com/Unbox-AI/behaviorgpt) and the Counterfactual
Sensitivity Lab's internal data models.

The adapter:
- Converts BehavioralAction objects into SDK event types (View, AddToCart, Search)
- Calls `UnboxAIClient.complete()` for inference
- Normalizes `UnboxAIResponse` into the lab's `InferenceOutput` / `PredictionItem` schema
- Maps SDK exceptions into structured error responses
- Records execution provenance (latency, provider, model identifier)
"""

import logging
import os
import time
from typing import List, Optional

from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    InferenceOutput,
    PredictionItem,
)

logger = logging.getLogger(__name__)

# Deferred SDK imports — fail gracefully if the package is missing
_SDK_AVAILABLE = False
try:
    from behaviorgpt import (
        AddToCart as SDKAddToCart,
        AuthenticationError as SDKAuthError,
        BadRequestError as SDKBadRequest,
        RateLimitError as SDKRateLimit,
        Search as SDKSearch,
        UnboxAIClient,
        UnboxAIError as SDKError,
        View as SDKView,
    )
    from behaviorgpt.types import Item as SDKItem, UnboxAIResponse as SDKResponse

    _SDK_AVAILABLE = True
except ImportError:
    pass


def is_sdk_available() -> bool:
    """Return True if the behaviorgpt package is importable."""
    return _SDK_AVAILABLE


def _action_to_sdk_event(action: BehavioralAction):
    """Convert a lab BehavioralAction into an SDK UserEvent.

    Mapping:
        ActionType.VIEW_ITEM  -> View(item_id)
        ActionType.ADD_TO_CART -> AddToCart(item_id)
        ActionType.SEARCH      -> Search(query)

    Raises ValueError if the SDK package is not installed.
    """
    if not _SDK_AVAILABLE:
        raise RuntimeError("behaviorgpt package is not installed")

    if action.action_type == ActionType.SEARCH:
        return SDKSearch(query=action.query or "")
    elif action.action_type == ActionType.ADD_TO_CART:
        return SDKAddToCart(item=action.product_id or "")
    else:  # VIEW_ITEM
        return SDKView(item=action.product_id or "")


def _history_to_sdk_events(history: BehavioralHistory) -> list:
    """Convert an entire BehavioralHistory into a list of SDK UserEvent objects."""
    return [_action_to_sdk_event(a) for a in history.actions]


def _sdk_item_to_prediction(item, rank: int) -> PredictionItem:
    """Convert an SDK Item into the lab's PredictionItem schema.

    SDK Item fields:
        item.id    -> str
        item.score -> float  (model confidence, analogous to showcase 'prob')
        item.data  -> dict   (contains 'name', 'brand', 'category', 'price', etc.)
    """
    data = item.data or {}
    # Price may be nested under markets -> {market} -> price, or directly in data
    raw_price = data.get("price")
    price_val = None
    if raw_price is not None and raw_price != "":
        try:
            price_val = float(raw_price)
        except (ValueError, TypeError):
            pass

    return PredictionItem(
        id=item.id,
        name=data.get("name", "Unknown Product"),
        score=item.score,
        prob=item.score,  # SDK 'score' is the probability analogue
        rank=rank,
        brand=data.get("brand"),
        category=data.get("category"),
        price=price_val,
        currency=data.get("currency", "USD"),
        keywords=data.get("keywords", []),
        top_terms=data.get("topTerms", []),
    )


class BehaviorGPTSDKAdapter:
    """Adapter wrapping the official UnboxAIClient for experiment inference.

    Usage:
        adapter = BehaviorGPTSDKAdapter(market="us")
        output = adapter.predict(history, limit=10, variant_label="original", case_id="case_001")
    """

    PROVIDER = "behaviorgpt_sdk"

    def __init__(
        self,
        market: str = "us",
        catalog_id: str = "retail_catalog",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                "The behaviorgpt package is not installed. "
                "Install it with: pip install behaviorgpt  "
                "(or pip install -e path/to/behaviorgpt-repo)"
            )

        resolved_key = api_key or os.environ.get("UNBOXAI_API_KEY")
        if not resolved_key:
            raise SDKAuthError(
                "UNBOXAI_API_KEY is not set. "
                "Set it as an environment variable or pass api_key= explicitly."
            )

        self._market = market
        self._catalog_id = catalog_id

        import httpx
        self._client = UnboxAIClient(
            market=market,
            api_key=resolved_key,
            default_catalog_id=catalog_id,
            http_timeout=httpx.Timeout(timeout, connect=10.0),
        )
        logger.info(
            "SDK adapter initialized: market=%s, catalog=%s, base_url=%s",
            market,
            catalog_id,
            self._client.base_url,
        )

    def predict(
        self,
        history: BehavioralHistory,
        limit: int = 10,
        variant_label: str = "original",
        case_id: str = "case_000",
    ) -> InferenceOutput:
        """Submit a behavioral history through the official SDK and return normalized predictions."""
        start = time.perf_counter()

        try:
            sdk_events = _history_to_sdk_events(history)
            logger.debug(
                "SDK predict: case=%s variant=%s events=%d limit=%d",
                case_id, variant_label, len(sdk_events), limit,
            )

            response = self._client.complete(
                history=sdk_events,
                limit=limit,
                catalog_id=self._catalog_id,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            items = [
                _sdk_item_to_prediction(item, rank)
                for rank, item in enumerate(response.items, start=1)
            ]

            logger.info(
                "SDK predict OK: case=%s variant=%s items=%d latency=%.1fms",
                case_id, variant_label, len(items), elapsed_ms,
            )

            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=items,
                latency_ms=elapsed_ms,
                model_name=f"BehaviorGPT-SDK-{self._market}",
                catalog_id=self._catalog_id,
                provider=self.PROVIDER,
                is_mock=False,
                status_code=200,
                error_message=None,
                raw_payload={"item_count": len(items), "offset": response.offset},
            )

        except SDKAuthError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error("SDK AuthenticationError: %s", e)
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=elapsed_ms,
                model_name=f"BehaviorGPT-SDK-{self._market}",
                catalog_id=self._catalog_id,
                provider=self.PROVIDER,
                status_code=401,
                error_message=f"AuthenticationError: {e}",
            )

        except SDKRateLimit as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("SDK RateLimitError: %s", e)
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=elapsed_ms,
                model_name=f"BehaviorGPT-SDK-{self._market}",
                catalog_id=self._catalog_id,
                provider=self.PROVIDER,
                status_code=429,
                error_message=f"RateLimitError: {e}",
            )

        except SDKBadRequest as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error("SDK BadRequestError: %s", e)
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=elapsed_ms,
                model_name=f"BehaviorGPT-SDK-{self._market}",
                catalog_id=self._catalog_id,
                provider=self.PROVIDER,
                status_code=400,
                error_message=f"BadRequestError: {e}",
            )

        except SDKError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error("SDK UnboxAIError: status=%s msg=%s", e.status_code, e)
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=elapsed_ms,
                model_name=f"BehaviorGPT-SDK-{self._market}",
                catalog_id=self._catalog_id,
                provider=self.PROVIDER,
                status_code=e.status_code or 500,
                error_message=f"UnboxAIError: {e}",
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("SDK unexpected error: %s", e)
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=elapsed_ms,
                model_name=f"BehaviorGPT-SDK-{self._market}",
                catalog_id=self._catalog_id,
                provider=self.PROVIDER,
                status_code=500,
                error_message=f"{type(e).__name__}: {e}",
            )

    def close(self):
        """Close the underlying HTTP client."""
        if hasattr(self._client, '_http_client'):
            self._client._http_client.close()
        logger.info("SDK adapter closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
