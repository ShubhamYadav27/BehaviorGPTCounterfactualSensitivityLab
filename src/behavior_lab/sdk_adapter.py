"""SDK adapter for official BehaviorGPT Python package.

Provides a clean adapter layer between the official `behaviorgpt` SDK
(https://github.com/Unbox-AI/behaviorgpt) and the Counterfactual
Sensitivity Lab's internal data models.

The adapter:
- Converts BehavioralAction objects into official SDK event types (View, AddToCart, Search, Order)
- Supports deterministic relative timestamps to guarantee temporal request equivalence
- Distinguishes uncalibrated model scores (inner product similarity) from calibrated probabilities
- Calls `UnboxAIClient.complete()` or `AsyncUnboxAIClient.complete()` for inference
- Exposes `similar_items()` via both synchronous and asynchronous adapters
- Normalizes SDK responses into the lab's `InferenceOutput` / `PredictionItem` schema
- Maps SDK exceptions into structured error responses
- Records execution provenance (latency, provider, model identifier)
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

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
        AsyncUnboxAIClient,
        AuthenticationError as SDKAuthError,
        BadRequestError as SDKBadRequest,
        Order as SDKOrder,
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


def resolve_base_timestamp(
    history: Optional[BehavioralHistory] = None,
    base_time: Optional[datetime] = None,
) -> datetime:
    """Resolve a timezone-aware deterministic base timestamp for SDK event sequences.

    Precedence:
    1. Explicit `base_time` if provided (e.g. shared pair session anchor).
    2. `history.created_at` parsed as ISO-8601 UTC if present and valid.
    3. Recent deterministic anchor snapped to current UTC hour:
       datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).

    This eliminates stale-history recency bias while guaranteeing exact temporal equivalence
    between paired original and counterfactual sequences.
    """
    if base_time is not None:
        if base_time.tzinfo is None:
            return base_time.replace(tzinfo=timezone.utc)
        return base_time

    if history is not None and history.created_at:
        try:
            parsed = datetime.fromisoformat(history.created_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, TypeError):
            pass

    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _action_to_sdk_event(
    action: BehavioralAction,
    timestamp: Optional[datetime] = None,
    legacy_cart_mapping: bool = False,
):
    """Convert a lab BehavioralAction into an official SDK UserEvent.

    Mapping:
        ActionType.VIEW_ITEM   -> View(item, timestamp)
        ActionType.ADD_TO_CART -> View(item, timestamp) [Default: Normalized per SDK guidance]
                                  AddToCart(item, timestamp) [Isolated legacy mapping if legacy_cart_mapping=True]
            (Note: In BehaviorGPT v0.11.0, AddToCart is explicitly deprecated by Unbox AI:
             "the model reads it the same as View. Send a View, and list what was bought in Order([...])".
             Normalizing to View avoids deprecation warnings while preserving the original
             action verb in the application's experiment history provenance).
        ActionType.ORDER       -> Order(items=[item], timestamp)
        ActionType.SEARCH      -> Search(query, timestamp)

    Raises RuntimeError if the behaviorgpt package is not installed.
    """
    if not _SDK_AVAILABLE:
        raise RuntimeError("behaviorgpt package is not installed")

    if action.action_type == ActionType.SEARCH:
        return SDKSearch(query=action.query or "", timestamp=timestamp)
    elif action.action_type == ActionType.ADD_TO_CART:
        if legacy_cart_mapping:
            return SDKAddToCart(item=action.product_id or "", timestamp=timestamp)
        # Default: normalize to SDK View per official Unbox AI directive
        return SDKView(item=action.product_id or "", timestamp=timestamp)
    elif action.action_type == ActionType.ORDER:
        items = [action.product_id] if action.product_id else []
        return SDKOrder(items=items, timestamp=timestamp)
    else:  # VIEW_ITEM
        return SDKView(item=action.product_id or "", timestamp=timestamp)


def _history_to_sdk_events(
    history: BehavioralHistory,
    base_time: Optional[datetime] = None,
    legacy_cart_mapping: bool = False,
) -> list:
    """Convert an entire BehavioralHistory into a list of SDK UserEvent objects.

    Uses resolve_base_timestamp() to establish a deterministic temporal anchor,
    ensuring identical temporal coordinates between paired original and counterfactual sequences.
    """
    ref_time = resolve_base_timestamp(history=history, base_time=base_time)
    events = []
    for action in history.actions:
        event_time = ref_time + timedelta(milliseconds=action.offset_ms)
        events.append(_action_to_sdk_event(action, timestamp=event_time, legacy_cart_mapping=legacy_cart_mapping))
    return events


def _sdk_item_to_prediction(item, rank: int) -> PredictionItem:
    """Convert an SDK Item into the lab's PredictionItem schema.

    Important: BehaviorGPT SDK item.score represents an uncalibrated inner-product similarity score,
    not a calibrated probability. We record it in `model_score` and `score`, while setting
    `prob=None` to prevent invalid probability-based calculations (e.g. Total Variation Distance).
    """
    data = item.data or {}
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
        model_score=item.score,
        prob=None,  # Uncalibrated model score, not a probability
        rank=rank,
        brand=data.get("brand"),
        category=data.get("category"),
        price=price_val,
        currency=data.get("currency", "USD"),
        keywords=data.get("keywords") or [],
        top_terms=data.get("topTerms") or [],
    )


class BehaviorGPTSDKAdapter:
    """Synchronous adapter wrapping the official UnboxAIClient for experiment inference.

    Usage:
        adapter = BehaviorGPTSDKAdapter(market="us")
        output = adapter.predict(history, limit=10, variant_label="original", case_id="case_001")
        similar = adapter.similar_items(item_id="paB0D93VKPPK", limit=5)
    """

    PROVIDER = "behaviorgpt_sdk"

    def __init__(
        self,
        market: str = "us",
        catalog_id: str = "retail_catalog_500k",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        legacy_cart_mapping: bool = False,
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
        self._legacy_cart_mapping = legacy_cart_mapping

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
        base_time: Optional[datetime] = None,
    ) -> InferenceOutput:
        """Submit a behavioral history through the official SDK and return normalized predictions."""
        start = time.perf_counter()

        try:
            sdk_events = _history_to_sdk_events(
                history, base_time=base_time, legacy_cart_mapping=self._legacy_cart_mapping
            )
            logger.debug(
                "SDK predict: case=%s variant=%s events=%d limit=%d",
                case_id,
                variant_label,
                len(sdk_events),
                limit,
            )

            ref_time = resolve_base_timestamp(history=history, base_time=base_time)
            max_offset = max([a.offset_ms for a in history.actions], default=0)
            req_timestamp = ref_time + timedelta(milliseconds=max_offset + 1000)

            response = self._client.complete(
                history=sdk_events,
                limit=limit,
                catalog_id=self._catalog_id,
                timestamp=req_timestamp,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            items = [
                _sdk_item_to_prediction(item, rank)
                for rank, item in enumerate(response.items, start=1)
            ]

            logger.info(
                "SDK predict OK: case=%s variant=%s items=%d latency=%.1fms",
                case_id,
                variant_label,
                len(items),
                elapsed_ms,
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

    def similar_items(
        self,
        item_id: str,
        limit: int = 10,
        catalog_id: Optional[str] = None,
    ) -> List[PredictionItem]:
        """Fetch similar items using official SDK similar_items()."""
        cat_id = catalog_id or self._catalog_id
        response = self._client.similar_items(
            item_id=item_id,
            limit=limit,
            catalog_id=cat_id,
        )
        return [
            _sdk_item_to_prediction(item, rank)
            for rank, item in enumerate(response.items, start=1)
        ]

    def close(self):
        """Close the underlying HTTP client."""
        if hasattr(self._client, "_http_client"):
            self._client._http_client.close()
        logger.info("SDK adapter closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncBehaviorGPTSDKAdapter:
    """Native asynchronous adapter wrapping AsyncUnboxAIClient.

    Designed for non-blocking execution in async FastAPI endpoints and concurrent pipelines.
    Properly implements async lifecycle: aclose() and async context manager.
    """

    PROVIDER = "behaviorgpt_sdk_async"

    def __init__(
        self,
        market: str = "us",
        catalog_id: str = "retail_catalog_500k",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        legacy_cart_mapping: bool = False,
    ):
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                "The behaviorgpt package is not installed. "
                "Install it with: pip install behaviorgpt"
            )

        resolved_key = api_key or os.environ.get("UNBOXAI_API_KEY")
        if not resolved_key:
            raise SDKAuthError(
                "UNBOXAI_API_KEY is not set. "
                "Set it as an environment variable or pass api_key= explicitly."
            )

        self._market = market
        self._catalog_id = catalog_id
        self._legacy_cart_mapping = legacy_cart_mapping

        import httpx

        self._client = AsyncUnboxAIClient(
            market=market,
            api_key=resolved_key,
            default_catalog_id=catalog_id,
            http_timeout=httpx.Timeout(timeout, connect=10.0),
        )
        logger.info(
            "Async SDK adapter initialized: market=%s, catalog=%s",
            market,
            catalog_id,
        )

    async def predict(
        self,
        history: BehavioralHistory,
        limit: int = 10,
        variant_label: str = "original",
        case_id: str = "case_000",
        base_time: Optional[datetime] = None,
    ) -> InferenceOutput:
        """Submit a behavioral history asynchronously through the official SDK."""
        start = time.perf_counter()
        try:
            sdk_events = _history_to_sdk_events(
                history, base_time=base_time, legacy_cart_mapping=self._legacy_cart_mapping
            )
            ref_time = resolve_base_timestamp(history=history, base_time=base_time)
            max_offset = max([a.offset_ms for a in history.actions], default=0)
            req_timestamp = ref_time + timedelta(milliseconds=max_offset + 1000)

            response = await self._client.complete(
                history=sdk_events,
                limit=limit,
                catalog_id=self._catalog_id,
                timestamp=req_timestamp,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            items = [
                _sdk_item_to_prediction(item, rank)
                for rank, item in enumerate(response.items, start=1)
            ]

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
            logger.error("Async SDK AuthenticationError: %s", e)
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
            logger.warning("Async SDK RateLimitError: %s", e)
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
            logger.error("Async SDK BadRequestError: %s", e)
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
            logger.error("Async SDK UnboxAIError: status=%s msg=%s", e.status_code, e)
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
            logger.exception("Async SDK unexpected error: %s", e)
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

    async def similar_items(
        self,
        item_id: str,
        limit: int = 10,
        catalog_id: Optional[str] = None,
    ) -> List[PredictionItem]:
        """Fetch similar items asynchronously."""
        cat_id = catalog_id or self._catalog_id
        response = await self._client.similar_items(
            item_id=item_id,
            limit=limit,
            catalog_id=cat_id,
        )
        return [
            _sdk_item_to_prediction(item, rank)
            for rank, item in enumerate(response.items, start=1)
        ]

    async def aclose(self):
        """Close the underlying async HTTP client session."""
        if hasattr(self._client, "aclose"):
            await self._client.aclose()
        elif hasattr(self._client, "_http_client"):
            await self._client._http_client.aclose()
        logger.info("Async SDK adapter closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
