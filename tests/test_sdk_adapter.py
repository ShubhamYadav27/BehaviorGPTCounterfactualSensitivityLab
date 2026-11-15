"""Unit and regression tests for BehaviorGPT SDK adapter and Stage B1 integrity.

Tests verify:
1. Event conversion fidelity (View, AddToCart, Order, Search) with deterministic timestamps
2. Score vs probability integrity (uncalibrated scores stored in model_score, prob is None)
3. Disjoint list anomaly resolution (case_001 fix: shared displacement is None, penalized displacement > 0)
4. Strict category matching in within-category mutations (no category leakage)
5. Metric calculations on uncalibrated vs calibrated outputs
6. Synchronous and Asynchronous SDK adapter lifecycle & resource cleanup
7. Experiment runner data directory isolation
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from behavior_lab.dataset_generator import VERIFIED_ITEMS
from behavior_lab.metrics import SensitivityMetricsCalculator
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    InferenceOutput,
    MutationPosition,
    MutationType,
    PredictionItem,
)
from behavior_lab.mutator import ControlledMutator, ITEM_TO_META, CATEGORY_TO_ITEMS
from behavior_lab.runner import ExperimentRunner
from behavior_lab.sdk_adapter import (
    AsyncBehaviorGPTSDKAdapter,
    BehaviorGPTSDKAdapter,
    _action_to_sdk_event,
    _history_to_sdk_events,
    _sdk_item_to_prediction,
    is_sdk_available,
    resolve_base_timestamp,
)


# ---------------------------------------------------------------------------
# Event conversion tests
# ---------------------------------------------------------------------------

def test_view_action_converts_to_sdk_view():
    ref_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    action = BehavioralAction(
        action_id="a1",
        action_type=ActionType.VIEW_ITEM,
        product_id="product-123",
        offset_ms=500,
    )
    event = _action_to_sdk_event(action, timestamp=ref_ts)
    assert event.type == "viewItem"
    assert event.item == "product-123"
    assert event.timestamp == ref_ts


def test_add_to_cart_normalized_to_view_by_default():
    """AddToCart normalizes to SDK View by default per Unbox AI documentation, with no deprecation warning."""
    import warnings

    ref_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    action = BehavioralAction(
        action_id="a2",
        action_type=ActionType.ADD_TO_CART,
        product_id="product-456",
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        event = _action_to_sdk_event(action, timestamp=ref_ts, legacy_cart_mapping=False)

    assert event.type == "viewItem"
    assert event.item == "product-456"
    assert event.timestamp == ref_ts
    # Assert zero deprecation warnings emitted
    dep_warnings = [w for w in recorded_warnings if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 0


def test_add_to_cart_legacy_mapping_isolated():
    """Explicit legacy_cart_mapping=True constructs SDK AddToCart and raises DeprecationWarning."""
    ref_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    action = BehavioralAction(
        action_id="a2_leg",
        action_type=ActionType.ADD_TO_CART,
        product_id="product-456",
    )
    with pytest.warns(DeprecationWarning, match="`AddToCart` is deprecated"):
        event = _action_to_sdk_event(action, timestamp=ref_ts, legacy_cart_mapping=True)

    assert event.type == "addToCart"
    assert event.item == "product-456"
    assert event.timestamp == ref_ts


def test_resolve_base_timestamp_precedence():
    """Test resolution order: explicit base_time > history.created_at > current UTC hour."""
    # 1. Explicit base_time overrides everything
    explicit_ts = datetime(2026, 5, 20, 8, 30, 0, tzinfo=timezone.utc)
    hist = BehavioralHistory(
        history_id="h_ts",
        archetype="TEST",
        description="TS Test",
        actions=[],
        created_at="2026-10-15T12:00:00Z",
    )
    resolved_1 = resolve_base_timestamp(history=hist, base_time=explicit_ts)
    assert resolved_1 == explicit_ts

    # 2. Uses history.created_at when base_time is None
    resolved_2 = resolve_base_timestamp(history=hist, base_time=None)
    assert resolved_2 == datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

    # 3. Falls back to recent UTC hour anchor when created_at is missing
    hist_no_created = BehavioralHistory(
        history_id="h_none",
        archetype="TEST",
        description="No Created",
        actions=[],
        created_at="",
    )
    resolved_3 = resolve_base_timestamp(history=hist_no_created, base_time=None)
    assert resolved_3.tzinfo is not None
    now_utc = datetime.now(timezone.utc)
    assert resolved_3.year == now_utc.year
    assert resolved_3.minute == 0
    assert resolved_3.second == 0


def test_order_action_converts_to_sdk_order():
    ref_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    action = BehavioralAction(
        action_id="a_order",
        action_type=ActionType.ORDER,
        product_id="product-789",
    )
    event = _action_to_sdk_event(action, timestamp=ref_ts)
    assert event.type == "order"
    assert len(event.items) == 1
    assert event.items[0].item == "product-789"
    assert event.timestamp == ref_ts


def test_search_converts_to_sdk_search():
    ref_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    action = BehavioralAction(
        action_id="a3",
        action_type=ActionType.SEARCH,
        query="running shoes",
    )
    event = _action_to_sdk_event(action, timestamp=ref_ts)
    assert event.type == "search"
    assert event.query == "running shoes"
    assert event.timestamp == ref_ts


def test_deterministic_timestamps_in_history_conversion():
    base_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    history = BehavioralHistory(
        history_id="h1",
        archetype="TEST",
        description="Test history",
        actions=[
            BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="p1", offset_ms=0),
            BehavioralAction(action_id="a2", action_type=ActionType.SEARCH, query="test", offset_ms=1500),
            BehavioralAction(action_id="a3", action_type=ActionType.ORDER, product_id="p3", offset_ms=3000),
        ],
        created_at="2026-01-01T00:00:00Z",
    )
    events = _history_to_sdk_events(history, base_time=base_time)
    assert len(events) == 3
    assert events[0].timestamp == base_time
    assert events[1].timestamp == base_time + timedelta(milliseconds=1500)
    assert events[2].timestamp == base_time + timedelta(milliseconds=3000)


# ---------------------------------------------------------------------------
# SDK Item -> PredictionItem conversion & score integrity tests
# ---------------------------------------------------------------------------

def _make_mock_sdk_item(item_id: str, score: float, data: dict):
    mock = MagicMock()
    mock.id = item_id
    mock.score = score
    mock.data = data
    return mock


def test_sdk_item_to_prediction_uncalibrated_score():
    """Uncalibrated SDK scores must be stored in model_score, not prob."""
    item = _make_mock_sdk_item(
        "item-001",
        1.5093,  # Typical BehaviorGPT raw similarity score (> 1.0)
        {"name": "Test Product", "brand": "TestBrand", "category": "Electronics", "price": 29.99},
    )
    pred = _sdk_item_to_prediction(item, rank=1)
    assert pred.id == "item-001"
    assert pred.name == "Test Product"
    assert pred.score == 1.5093
    assert pred.model_score == 1.5093
    assert pred.prob is None  # Must NOT be presented as a probability
    assert pred.rank == 1
    assert pred.brand == "TestBrand"
    assert pred.category == "Electronics"
    assert pred.price == 29.99


def test_sdk_item_to_prediction_missing_fields():
    item = _make_mock_sdk_item("item-002", 0.85, {})
    pred = _sdk_item_to_prediction(item, rank=3)
    assert pred.id == "item-002"
    assert pred.name == "Unknown Product"
    assert pred.brand is None
    assert pred.price is None
    assert pred.rank == 3
    assert pred.prob is None
    assert pred.model_score == 0.85


def test_sdk_item_none_keywords_and_top_terms():
    item = _make_mock_sdk_item(
        "item-005",
        0.05,
        {"name": "Null Term Item", "keywords": None, "topTerms": None},
    )
    pred = _sdk_item_to_prediction(item, rank=1)
    assert pred.keywords == []
    assert pred.top_terms == []


# ---------------------------------------------------------------------------
# case_001 Disjoint Lists Anomaly Regression Test
# ---------------------------------------------------------------------------

def test_disjoint_lists_rank_displacement_not_misleading_zero():
    """Verify case_001 fix: completely disjoint lists yield None for shared displacement and > 0 for penalized."""
    # List A and List B have zero overlap
    items_a = [
        PredictionItem(id=f"item_a_{i}", name=f"Product A{i}", score=2.0 - i * 0.1, rank=i)
        for i in range(1, 6)
    ]
    items_b = [
        PredictionItem(id=f"item_b_{i}", name=f"Product B{i}", score=1.8 - i * 0.1, rank=i)
        for i in range(1, 6)
    ]

    out_a = InferenceOutput(case_id="case_001", variant="original", items=items_a, latency_ms=10.0)
    out_b = InferenceOutput(case_id="case_001", variant="mutated", items=items_b, latency_ms=10.0)

    metrics = SensitivityMetricsCalculator.compute(out_a, out_b)

    assert metrics.top1_changed is True
    assert metrics.jaccard_top5 == 0.0
    assert metrics.shared_item_count == 0
    # Must NOT report 0.0 for rank displacement when no items are shared!
    assert metrics.rank_displacement_shared is None
    assert metrics.rank_displacement_avg is None
    # Penalized displacement accounts for all 10 evicted/new items with penalty rank K+1 = 6
    # For each item, |rank - 6|: ranks 1..5 displaced by 5, 4, 3, 2, 1 = 15; sum for 10 items = 30; 30/10 = 3.0
    assert metrics.rank_displacement_penalized == 3.0
    # TVD on uncalibrated raw scores must be None
    assert metrics.prob_tvd_shared is None
    # Score L1 on shared items must be None because 0 items shared
    assert metrics.score_l1_shared is None


def test_partially_overlapping_lists_displacement_and_score_l1():
    """Lists with 2 shared items compute displacement only over shared items and penalized across union."""
    items_a = [
        PredictionItem(id="p1", name="P1", score=10.0, rank=1),
        PredictionItem(id="p2", name="P2", score=8.0, rank=2),
        PredictionItem(id="p3", name="P3", score=6.0, rank=3),
    ]
    items_b = [
        PredictionItem(id="p2", name="P2", score=9.0, rank=1),  # rank 2 -> 1, delta score = 1.0
        PredictionItem(id="p4", name="P4", score=7.0, rank=2),
        PredictionItem(id="p1", name="P1", score=5.0, rank=3),  # rank 1 -> 3, delta score = 5.0
    ]

    out_a = InferenceOutput(case_id="c_overlap", variant="original", items=items_a, latency_ms=10.0)
    out_b = InferenceOutput(case_id="c_overlap", variant="mutated", items=items_b, latency_ms=10.0)

    metrics = SensitivityMetricsCalculator.compute(out_a, out_b)
    assert metrics.shared_item_count == 2
    # Shared displacements: p1 |1-3|=2, p2 |2-1|=1 -> avg (2+1)/2 = 1.5
    assert metrics.rank_displacement_shared == 1.5
    assert metrics.rank_displacement_avg == 1.5
    # Score L1 on shared: (|10-5| + |8-9|) / 2 = (5 + 1)/2 = 3.0
    assert metrics.score_l1_shared == 3.0
    # Top-1 score delta: |10.0 - 9.0| = 1.0
    assert metrics.top1_score_delta == 1.0


# ---------------------------------------------------------------------------
# Strict Category Matching & Category Leakage Regression Tests
# ---------------------------------------------------------------------------

def test_within_category_mutation_strict_category_matching():
    """Ensure within_category substitutions always produce a verified sibling from the same category."""
    mutator = ControlledMutator()

    for item_key, item_info in VERIFIED_ITEMS.items():
        orig_id = item_info["id"]
        orig_cat = item_info["cat"]

        history = BehavioralHistory(
            history_id="test_cat",
            archetype="TEST",
            description="Category test",
            actions=[
                BehavioralAction(
                    action_id="act_0",
                    action_type=ActionType.VIEW_ITEM,
                    product_id=orig_id,
                    offset_ms=0,
                )
            ],
            created_at="2026-01-01T00:00:00Z",
        )

        mutated, spec = mutator._apply_single_mutation(
            orig_hist=history,
            case_id="case_cat_test",
            mutation_type=MutationType.ITEM_REPLACEMENT_WITHIN_CATEGORY,
            pos_cat=MutationPosition.LATE,
            pos_idx=0,
        )

        mutated_id = mutated.actions[0].product_id
        assert mutated_id != orig_id, f"Item {orig_id} was replaced with itself!"
        mutated_meta = ITEM_TO_META.get(mutated_id)
        assert mutated_meta is not None, f"Replacement {mutated_id} is not in verified catalog"
        assert mutated_meta["cat"] == orig_cat, (
            f"Category leakage detected! Item {orig_id} ({orig_cat}) was mutated to {mutated_id} ({mutated_meta['cat']})"
        )


def test_within_category_mutation_rejects_missing_metadata():
    """Unverified items without verified category metadata must raise ValueError, not silently leak."""
    mutator = ControlledMutator()
    history = BehavioralHistory(
        history_id="test_unverified",
        archetype="TEST",
        description="Unverified item test",
        actions=[
            BehavioralAction(
                action_id="act_0",
                action_type=ActionType.VIEW_ITEM,
                product_id="unverified_foreign_item_999",
                offset_ms=0,
            )
        ],
        created_at="2026-01-01T00:00:00Z",
    )

    with pytest.raises(ValueError, match="Cannot perform within-category replacement"):
        mutator._apply_single_mutation(
            orig_hist=history,
            case_id="case_err",
            mutation_type=MutationType.ITEM_REPLACEMENT_WITHIN_CATEGORY,
            pos_cat=MutationPosition.LATE,
            pos_idx=0,
        )


# ---------------------------------------------------------------------------
# SDK Adapter Synchronous and Asynchronous Lifecycle Tests
# ---------------------------------------------------------------------------

def test_sdk_adapter_default_catalog():
    with patch.dict("os.environ", {"UNBOXAI_API_KEY": "test_key"}):
        with patch("behavior_lab.sdk_adapter.UnboxAIClient") as mock_client:
            adapter = BehaviorGPTSDKAdapter()
            assert adapter._catalog_id == "retail_catalog_500k"
            mock_client.assert_called_once()
            _, kwargs = mock_client.call_args
            assert kwargs.get("default_catalog_id") == "retail_catalog_500k"


def test_sdk_adapter_similar_items():
    with patch.dict("os.environ", {"UNBOXAI_API_KEY": "test_key"}):
        with patch("behavior_lab.sdk_adapter.UnboxAIClient") as mock_client_cls:
            mock_instance = MagicMock()
            mock_client_cls.return_value = mock_instance

            mock_sim_response = MagicMock()
            mock_sim_response.items = [
                _make_mock_sdk_item("sim_1", 0.95, {"name": "Similar Product 1"}),
                _make_mock_sdk_item("sim_2", 0.88, {"name": "Similar Product 2"}),
            ]
            mock_instance.similar_items.return_value = mock_sim_response

            adapter = BehaviorGPTSDKAdapter()
            items = adapter.similar_items(item_id="paB0D93VKPPK", limit=2)
            assert len(items) == 2
            assert items[0].id == "sim_1"
            assert items[0].rank == 1
            assert items[0].model_score == 0.95
            assert items[0].prob is None


def test_async_sdk_adapter_lifecycle():
    """Verify AsyncBehaviorGPTSDKAdapter initialization, predict, similar_items, and aclose."""

    def run_coroutine(coro):
        while True:
            try:
                coro.send(None)
            except StopIteration as e:
                return e.value

    async def scenario():
        with patch.dict("os.environ", {"UNBOXAI_API_KEY": "test_key"}):
            with patch("behavior_lab.sdk_adapter.AsyncUnboxAIClient") as mock_async_client_cls:
                mock_instance = MagicMock()
                mock_instance.aclose = AsyncMock()
                mock_response = MagicMock()
                mock_response.items = [_make_mock_sdk_item("item_1", 1.25, {"name": "Item 1"})]
                mock_response.offset = 0
                mock_instance.complete = AsyncMock(return_value=mock_response)
                mock_instance.similar_items = AsyncMock(return_value=mock_response)
                mock_async_client_cls.return_value = mock_instance

                async with AsyncBehaviorGPTSDKAdapter() as adapter:
                    assert adapter.PROVIDER == "behaviorgpt_sdk_async"
                    history = BehavioralHistory(
                        history_id="async_test",
                        archetype="TEST",
                        description="Async test",
                        actions=[BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="p1")],
                        created_at="2026-01-01T00:00:00Z",
                    )
                    output = await adapter.predict(history, limit=1)
                    assert output.status_code == 200
                    assert len(output.items) == 1
                    assert output.items[0].model_score == 1.25

                    sim_items = await adapter.similar_items("p1", limit=1)
                    assert len(sim_items) == 1

                mock_instance.aclose.assert_awaited_once()

    run_coroutine(scenario())


# ---------------------------------------------------------------------------
# Runner Data Directory Isolation Test
# ---------------------------------------------------------------------------

def test_runner_data_dir_isolation():
    """Verify that mode='sdk' defaults to 'data/sdk_results' to protect historical showcase data in 'data/'."""
    sdk_runner = ExperimentRunner(mode="sdk")
    assert str(sdk_runner.data_dir).replace("\\", "/") == "data/sdk_results"

    live_runner = ExperimentRunner(mode="live")
    assert str(live_runner.data_dir).replace("\\", "/") == "data"
