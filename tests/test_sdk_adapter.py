"""Unit tests for BehaviorGPT SDK adapter.

Tests use mocked SDK responses — they never make real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
)
from behavior_lab.sdk_adapter import (
    _action_to_sdk_event,
    _history_to_sdk_events,
    _sdk_item_to_prediction,
    is_sdk_available,
)


# ---------------------------------------------------------------------------
# Event conversion tests
# ---------------------------------------------------------------------------

def test_view_action_converts_to_sdk_view():
    action = BehavioralAction(
        action_id="a1",
        action_type=ActionType.VIEW_ITEM,
        product_id="product-123",
    )
    event = _action_to_sdk_event(action)
    assert event.type == "viewItem"
    assert event.item == "product-123"


def test_add_to_cart_converts_to_sdk_addtocart():
    action = BehavioralAction(
        action_id="a2",
        action_type=ActionType.ADD_TO_CART,
        product_id="product-456",
    )
    event = _action_to_sdk_event(action)
    assert event.type == "addToCart"
    assert event.item == "product-456"


def test_search_converts_to_sdk_search():
    action = BehavioralAction(
        action_id="a3",
        action_type=ActionType.SEARCH,
        query="running shoes",
    )
    event = _action_to_sdk_event(action)
    assert event.type == "search"
    assert event.query == "running shoes"


def test_full_history_conversion():
    history = BehavioralHistory(
        history_id="h1",
        archetype="TEST",
        description="Test history",
        actions=[
            BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="p1"),
            BehavioralAction(action_id="a2", action_type=ActionType.ADD_TO_CART, product_id="p2"),
            BehavioralAction(action_id="a3", action_type=ActionType.SEARCH, query="test"),
        ],
        created_at="2026-01-01T00:00:00Z",
    )
    events = _history_to_sdk_events(history)
    assert len(events) == 3
    assert events[0].type == "viewItem"
    assert events[1].type == "addToCart"
    assert events[2].type == "search"


# ---------------------------------------------------------------------------
# SDK Item -> PredictionItem conversion tests
# ---------------------------------------------------------------------------

def _make_mock_sdk_item(item_id: str, score: float, data: dict):
    """Create a mock SDK Item object."""
    mock = MagicMock()
    mock.id = item_id
    mock.score = score
    mock.data = data
    return mock


def test_sdk_item_to_prediction_basic():
    item = _make_mock_sdk_item(
        "item-001",
        0.0123,
        {"name": "Test Product", "brand": "TestBrand", "category": "Electronics", "price": 29.99},
    )
    pred = _sdk_item_to_prediction(item, rank=1)
    assert pred.id == "item-001"
    assert pred.name == "Test Product"
    assert pred.score == 0.0123
    assert pred.prob == 0.0123
    assert pred.rank == 1
    assert pred.brand == "TestBrand"
    assert pred.category == "Electronics"
    assert pred.price == 29.99


def test_sdk_item_to_prediction_missing_fields():
    item = _make_mock_sdk_item("item-002", 0.005, {})
    pred = _sdk_item_to_prediction(item, rank=3)
    assert pred.id == "item-002"
    assert pred.name == "Unknown Product"
    assert pred.brand is None
    assert pred.price is None
    assert pred.rank == 3


def test_sdk_item_price_string_conversion():
    item = _make_mock_sdk_item("item-003", 0.01, {"name": "Widget", "price": "19.99"})
    pred = _sdk_item_to_prediction(item, rank=2)
    assert pred.price == 19.99


def test_sdk_item_price_invalid():
    item = _make_mock_sdk_item("item-004", 0.01, {"name": "Thing", "price": "N/A"})
    pred = _sdk_item_to_prediction(item, rank=1)
    assert pred.price is None


# ---------------------------------------------------------------------------
# SDK availability check
# ---------------------------------------------------------------------------

def test_sdk_available():
    # The SDK is installed in this environment
    assert is_sdk_available() is True


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------

def test_inference_output_provider_field():
    from behavior_lab.models import InferenceOutput
    out = InferenceOutput(
        case_id="test",
        variant="original",
        items=[],
        latency_ms=10.0,
        provider="behaviorgpt_sdk",
    )
    assert out.provider == "behaviorgpt_sdk"


def test_inference_output_default_provider():
    from behavior_lab.models import InferenceOutput
    out = InferenceOutput(
        case_id="test",
        variant="original",
        items=[],
        latency_ms=10.0,
    )
    assert out.provider == "showcase"
