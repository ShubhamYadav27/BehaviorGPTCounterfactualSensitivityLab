"""Client adapter for BehaviorGPT inference.

Supports:
1. Live inference via Unbox AI's public showcase endpoint (https://behaviorgpt.unboxai.com/api).
   This is NOT the official Python SDK gateway. No API key is required for the showcase.
   If UNBOXAI_API_KEY is set, it is sent as a header but the same showcase endpoint is used.
2. Deterministic mock mode for offline unit testing and regression verification.

Note: The official BehaviorGPT Python SDK (behaviorgpt package) uses a different
base URL and endpoint path. This client does not use the official SDK.
"""

import os
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
import httpx

from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    InferenceOutput,
    PredictionItem,
)


class BehaviorGPTClient:
    """Robust client adapter interfacing with Unbox AI's BehaviorGPT model."""

    def __init__(
        self,
        mode: str = "live",
        base_url: str = "https://behaviorgpt.unboxai.com/api",
        catalog_id: str = "retail_catalog_500k",
        model_name: str = "BehaviorGPT-v4-12.5B",
        market: str = "US",
        timeout_seconds: float = 20.0,
        api_key: Optional[str] = None,
    ):
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.catalog_id = catalog_id
        self.model_name = model_name
        self.market = market
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key or os.environ.get("UNBOXAI_API_KEY")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-byo-key"] = self.api_key
            headers["x-api-key"] = self.api_key

        self._http_client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
        )

    def close(self):
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def predict(
        self,
        history: BehavioralHistory,
        limit: int = 10,
        variant_label: str = "original",
        case_id: str = "case_000",
    ) -> InferenceOutput:
        """Submit a behavioral history and return normalized predictions."""
        if self.mode == "mock":
            return self._mock_predict(history, limit, variant_label, case_id)

        start_time = time.perf_counter()
        session_start_iso = datetime.now(tz=UTC).isoformat()

        # Build wire events from behavioral history
        wire_events = []
        for i, act in enumerate(history.actions):
            wire_events.append(act.to_wire_dict())

        payload = {
            "history": [
                {
                    "type": "default",
                    "start_time": session_start_iso,
                    "domains": {"market": self.market},
                    "events": wire_events,
                }
            ],
            "limit": limit,
        }

        try:
            resp = self._http_client.post(
                f"{self.base_url}/recommendations",
                json=payload,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                return InferenceOutput(
                    case_id=case_id,
                    variant=variant_label,
                    items=[],
                    latency_ms=elapsed_ms,
                    model_name=self.model_name,
                    catalog_id=self.catalog_id,
                    status_code=resp.status_code,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    raw_payload=None,
                )

            data = resp.json()
            raw_items = data.get("items", [])
            parsed_items = []
            for rank_idx, item in enumerate(raw_items, start=1):
                prob_val = item.get("prob")
                score_val = float(prob_val) if prob_val is not None else 1.0 / rank_idx
                parsed_items.append(
                    PredictionItem(
                        id=item.get("id", f"item_{rank_idx}"),
                        name=item.get("name", "Unknown Product"),
                        score=score_val,
                        prob=prob_val,
                        rank=rank_idx,
                        brand=item.get("brand"),
                        category=item.get("category"),
                        price=float(item["price"]) if item.get("price") is not None and item.get("price") != "" else None,
                        currency=item.get("currency", "USD"),
                        keywords=item.get("keywords", []),
                        top_terms=item.get("topTerms", []),
                    )
                )

            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=parsed_items,
                latency_ms=elapsed_ms,
                model_name=self.model_name,
                catalog_id=self.catalog_id,
                status_code=200,
                error_message=None,
                raw_payload={"item_count": len(parsed_items)},
            )

        except Exception as ex:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=elapsed_ms,
                model_name=self.model_name,
                catalog_id=self.catalog_id,
                status_code=500,
                error_message=f"Exception: {type(ex).__name__}: {str(ex)}",
                raw_payload=None,
            )

    def _mock_predict(
        self,
        history: BehavioralHistory,
        limit: int = 10,
        variant_label: str = "original",
        case_id: str = "case_000",
    ) -> InferenceOutput:
        """Deterministic mock inference for unit tests."""
        # Use history items to deterministically seed mock recommendations
        last_item = history.actions[-1].product_id if history.actions else "cold"
        is_shoes = any("shoe" in (a.product_id or "").lower() or "sneaker" in (a.product_id or "").lower() for a in history.actions)
        
        items = []
        for i in range(1, limit + 1):
            if is_shoes:
                pid = f"mock_shoe_{i}"
                pname = f"Mock Athletic Running Shoe Model {i}"
                brand = "Nike"
                cat = "Footwear"
                price = 100.0 + i * 15.0
            else:
                pid = f"mock_item_{last_item[:4]}_{i}"
                pname = f"Mock Relevant Item {i} for {last_item}"
                brand = "GenericBrand"
                cat = "General"
                price = 50.0 + i * 10.0

            prob = 0.01 / (i + 0.5)
            items.append(
                PredictionItem(
                    id=pid,
                    name=pname,
                    score=prob,
                    prob=prob,
                    rank=i,
                    brand=brand,
                    category=cat,
                    price=price,
                    currency="USD",
                    keywords=[cat.lower(), brand.lower()],
                    top_terms=[],
                )
            )

        return InferenceOutput(
            case_id=case_id,
            variant=variant_label,
            items=items,
            latency_ms=1.2,
            model_name="BehaviorGPT-v4-Mock",
            catalog_id=self.catalog_id,
            is_mock=True,
            status_code=200,
        )
