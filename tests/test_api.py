"""Automated tests for FastAPI endpoints in BehaviorGPT Counterfactual Sensitivity Lab."""

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException
from behavior_lab.api import (
    app,
    get_summary,
    get_cases,
    get_case_detail,
    get_investigation,
    get_dashboard,
    health_check,
    get_catalog_items,
    simulate_journey,
    compare_counterfactual,
    get_similar_items_endpoint,
    post_similar_items_endpoint,
    run_interactive,
    InteractiveHistoryRequest,
    lifespan,
)
from behavior_lab.models import (
    CounterfactualCompareRequest,
    JourneySimulateRequest,
    SimilarItemRequest,
)


class MockResponse:
    def __init__(self, status_code: int, data: dict = None, text: str = ""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text or (json.dumps(data) if data is not None else "")

    def json(self):
        return self._data


def _drive_coro(res):
    if inspect.iscoroutine(res):
        while True:
            try:
                res.send(None)
            except StopIteration as e:
                return e.value
    return res


class DirectClient:
    """Fast, deterministic test client directly invoking FastAPI endpoints.

    Avoids AnyIO BlockingPortal loopback socket synchronization hangs on Windows.
    """

    def get(self, path: str):
        if path == "/api/summary":
            return MockResponse(200, get_summary())
        elif path == "/api/catalog/items":
            return MockResponse(200, get_catalog_items())
        elif path.startswith("/api/items/") and path.endswith("/similar"):
            parts = path.split("/")
            item_id = parts[3]
            try:
                val = _drive_coro(get_similar_items_endpoint(item_id=item_id, mode="mock"))
                return MockResponse(200, val)
            except HTTPException as e:
                return MockResponse(e.status_code, {"detail": e.detail})
        elif path.startswith("/api/cases?"):
            limit = 50
            if "limit=" in path:
                limit = int(path.split("limit=")[1].split("&")[0])
            return MockResponse(200, get_cases(limit=limit))
        elif path == "/api/cases":
            return MockResponse(200, get_cases())
        elif path.startswith("/api/cases/"):
            case_id = path.replace("/api/cases/", "")
            try:
                return MockResponse(200, get_case_detail(case_id))
            except HTTPException as e:
                return MockResponse(e.status_code, {"detail": e.detail})
        elif path == "/api/investigation":
            return MockResponse(200, get_investigation())
        elif path in ("/", "/index.html"):
            resp = get_dashboard()
            return MockResponse(resp.status_code, text=resp.body.decode("utf-8"))
        elif path == "/health":
            return MockResponse(200, health_check())
        return MockResponse(404, {"detail": "Not found"})

    def post(self, path: str, json: dict = None):
        if path == "/api/run-interactive":
            try:
                req = InteractiveHistoryRequest(**(json or {}))
                val = _drive_coro(run_interactive(req))
                return MockResponse(200, val)
            except HTTPException as e:
                return MockResponse(e.status_code, {"detail": e.detail})
        elif path == "/api/journey/simulate":
            try:
                req = JourneySimulateRequest(**(json or {}))
                val = _drive_coro(simulate_journey(req))
                return MockResponse(200, val)
            except HTTPException as e:
                return MockResponse(e.status_code, {"detail": e.detail})
        elif path == "/api/counterfactual/compare":
            try:
                req = CounterfactualCompareRequest(**(json or {}))
                val = _drive_coro(compare_counterfactual(req))
                return MockResponse(200, val)
            except HTTPException as e:
                return MockResponse(e.status_code, {"detail": e.detail})
        elif path == "/api/items/similar":
            try:
                req = SimilarItemRequest(**(json or {}))
                val = _drive_coro(post_similar_items_endpoint(req))
                return MockResponse(200, val)
            except HTTPException as e:
                return MockResponse(e.status_code, {"detail": e.detail})
        return MockResponse(404, {"detail": "Not found"})


@pytest.fixture
def client():
    return DirectClient()


def test_api_summary(client):
    """GET /api/summary returns valid experiment summary."""
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_cases" in data
    assert data["total_cases"] == 40
    assert data["successful_cases"] == 40
    assert data["failed_cases"] == 0
    assert data["top1_change_rate"] == 0.65


def test_api_cases(client):
    """GET /api/cases returns list of evaluated paired cases."""
    response = client.get("/api/cases?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "cases" in data
    assert len(data["cases"]) <= 10
    case = data["cases"][0]
    assert "case_id" in case
    assert "archetype" in case
    assert "top1_changed" in case
    assert "jaccard_top5" in case


def test_api_case_detail(client):
    """GET /api/cases/{case_id} returns detailed pair data."""
    response = client.get("/api/cases/case_001")
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"]["case_id"] == "case_001"
    assert "original_output" in data["result"]
    assert "mutated_output" in data["result"]


def test_api_case_detail_not_found(client):
    """GET /api/cases/{nonexistent} returns 404."""
    response = client.get("/api/cases/case_9999")
    assert response.status_code == 404


def test_api_investigation(client):
    """GET /api/investigation returns deep dive reports."""
    response = client.get("/api/investigation")
    assert response.status_code == 200
    data = response.json()
    assert "reports" in data
    assert "investigated_cases" in data


def test_dashboard_ui(client):
    """GET / returns HTML dashboard with key UI elements."""
    response = client.get("/")
    assert response.status_code == 200
    assert "BehaviorGPT Counterfactual Sensitivity Lab" in response.text
    assert "Paired Case Inspector" in response.text


def test_dashboard_ui_stage_b2_sections(client):
    """GET / contains all three flagship Stage B2 interactive sections."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # 1. Behavioral Journey Simulator
    assert "Behavioral Journey Simulator" in html
    assert "tab-journey" in html
    assert "builderActionType" in html
    assert "builderProductSelect" in html
    assert "runJourneySimulation()" in html

    # 2. Counterfactual Playground
    assert "Counterfactual Sensitivity Playground" in html
    assert "tab-playground" in html
    assert "cfTimelineA" in html
    assert "cfTimelineB" in html
    assert "runCounterfactualComparison()" in html
    assert "Item Transition Dynamics" in html
    assert "cfTransitionsTableBody" in html

    # 3. Similar Item Explorer
    assert "Catalog Similar Item Explorer" in html
    assert "tab-similar" in html
    assert "similarSeedSelect" in html
    assert "similarSeedId" in html
    assert "runSimilarItemsQuery()" in html
    assert "similar_items()" in html


def test_run_interactive_success(client, monkeypatch):
    """1. Successful pair of live responses produces valid sensitivity metrics."""
    from behavior_lab.models import InferenceOutput, PredictionItem
    from behavior_lab.client import BehaviorGPTClient

    def mock_predict(self, history, limit=5, variant_label="original", case_id="case_000"):
        items = [
            PredictionItem(id=f"item_{i}_{variant_label}", name=f"Product {i}", score=0.9 - i * 0.1, prob=0.5 / (i + 1), rank=i + 1)
            for i in range(limit)
        ]
        return InferenceOutput(
            case_id=case_id,
            variant=variant_label,
            items=items,
            latency_ms=10.0,
            status_code=200,
        )

    monkeypatch.setattr(BehaviorGPTClient, "predict", mock_predict)

    payload = {
        "history_a_items": ["item_1", "item_2"],
        "history_b_items": ["item_1", "item_3"],
        "market": "US",
        "limit": 5,
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "top1_changed" in data["metrics"]
    assert "jaccard_top5" in data["metrics"]
    assert "sequence_a_output" in data
    assert "sequence_b_output" in data
    assert len(data["sequence_a_output"]["items"]) == 5
    assert len(data["sequence_b_output"]["items"]) == 5


def test_run_interactive_first_sequence_failure(client, monkeypatch):
    """2. Failed first-sequence inference returns an appropriate HTTP 502 error."""
    from behavior_lab.models import InferenceOutput
    from behavior_lab.client import BehaviorGPTClient

    def mock_predict(self, history, limit=5, variant_label="original", case_id="case_000"):
        if "Sequence A" in variant_label:
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[],
                latency_ms=5.0,
                status_code=500,
                error_message="HTTP 500: Internal Server Error from showcase",
            )
        return InferenceOutput(
            case_id=case_id,
            variant=variant_label,
            items=[],
            latency_ms=5.0,
            status_code=200,
        )

    monkeypatch.setattr(BehaviorGPTClient, "predict", mock_predict)

    payload = {
        "history_a_items": ["item_1"],
        "history_b_items": ["item_2"],
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 502
    err_detail = resp.json().get("detail", "")
    assert "Sequence A" in err_detail
    assert "500" in err_detail or "Internal Server Error" in err_detail


def test_run_interactive_second_sequence_failure(client, monkeypatch):
    """3. Failed second-sequence inference returns an appropriate HTTP 502 error."""
    from behavior_lab.models import InferenceOutput, PredictionItem
    from behavior_lab.client import BehaviorGPTClient

    def mock_predict(self, history, limit=5, variant_label="original", case_id="case_000"):
        if "Sequence A" in variant_label:
            return InferenceOutput(
                case_id=case_id,
                variant=variant_label,
                items=[PredictionItem(id="item_ok", name="OK Product", score=1.0, prob=0.5, rank=1)],
                latency_ms=5.0,
                status_code=200,
            )
        return InferenceOutput(
            case_id=case_id,
            variant=variant_label,
            items=[],
            latency_ms=5.0,
            status_code=504,
            error_message="HTTP 504: Upstream Gateway Timeout",
        )

    monkeypatch.setattr(BehaviorGPTClient, "predict", mock_predict)

    payload = {
        "history_a_items": ["item_1"],
        "history_b_items": ["item_2"],
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 502
    err_detail = resp.json().get("detail", "")
    assert "Sequence B" in err_detail
    assert "504" in err_detail or "Timeout" in err_detail


def test_ui_handles_unsuccessful_http_response(client):
    """4. The UI script contains explicit handling for non-ok HTTP responses."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "if (!res.ok)" in html
    assert "Live Inference Failed" in html
    assert "The public showcase endpoint returned an error. No sensitivity metrics were calculated." in html


def test_ui_handles_upstream_inference_error(client):
    """5. The UI script handles upstream inference errors and suppresses false success metrics."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "outA.status_code !== 200 || outB.status_code !== 200" in html
    assert "Boolean(outA.error_message) || Boolean(outB.error_message)" in html
    assert "Live Inference Unsuccessful" in html
    assert "Upstream model failed. Misleading zero-valued metrics have been suppressed." in html


def test_ui_distinct_labels_for_historical_and_live(client):
    """6. Historical and live results have distinct labels and explanations."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Historical Case Inspector label
    assert "Historical Record — October 15, 2026" in html
    assert "Historical predictions are loaded from the saved experiment records" in html

    # Interactive Sandbox label
    assert "Live Query — Public Showcase Endpoint" in html
    assert "The interactive sandbox sends fresh requests to the public showcase endpoint" in html
    assert "does not use the official authenticated BehaviorGPT SDK gateway" in html


def test_api_summary_unverified_metadata(client):
    """Verify /api/summary explicitly labels metadata as unverified local descriptors."""
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "metadata_provenance" in data
    assert data["is_model_name_verified"] is False
    assert data["is_catalog_id_verified"] is False
    assert data["metadata_provenance"]["model_name_status"] == "unverified_local_descriptor"


def test_run_interactive_sdk_mode_uses_async_adapter(client, monkeypatch):
    """Verify that mode='sdk' in /api/run-interactive uses AsyncBehaviorGPTSDKAdapter."""
    from behavior_lab.models import InferenceOutput, PredictionItem
    from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter

    monkeypatch.setenv("UNBOXAI_API_KEY", "test_key")
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "__init__", lambda self, market="us", **kw: None)

    mock_predict = AsyncMock()
    mock_predict.return_value = InferenceOutput(
        case_id="interactive_sdk",
        variant="Sequence A",
        items=[PredictionItem(id="item_sdk", name="SDK Item", score=1.5, rank=1)],
        latency_ms=15.0,
        provider="behaviorgpt_sdk_async",
        status_code=200,
    )
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "predict", mock_predict)
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "aclose", AsyncMock())

    payload = {
        "history_a_items": ["item_1"],
        "history_b_items": ["item_2"],
        "mode": "sdk",
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "behaviorgpt_sdk"
    assert mock_predict.await_count == 2


def test_api_lifespan_lifecycle():
    """Verify that lifespan initializes and cleanly closes app.state.sdk_adapter."""
    mock_adapter = MagicMock()
    mock_adapter.aclose = AsyncMock()

    def run_coro(coro):
        while True:
            try:
                coro.send(None)
            except StopIteration as e:
                return e.value

    async def scenario():
        with patch.dict("os.environ", {"UNBOXAI_API_KEY": "test_key"}):
            with patch("behavior_lab.sdk_adapter.AsyncBehaviorGPTSDKAdapter", return_value=mock_adapter):
                async with lifespan(app):
                    assert app.state.sdk_adapter is mock_adapter
                mock_adapter.aclose.assert_awaited_once()

    run_coro(scenario())


def test_api_catalog_items(client):
    """GET /api/catalog/items returns verified items and categories."""
    resp = client.get("/api/catalog/items")
    assert resp.status_code == 200
    data = resp.json()
    assert data["catalog_id"] == "retail_catalog_500k"
    assert data["total_items"] == 16
    assert len(data["items"]) == 16
    assert "Footwear" in data["categories"]
    assert "Gaming" in data["categories"]
    assert "Electronics" in data["categories"]
    # Check that item metadata includes has_similar_items
    dunk = next(it for it in data["items"] if it["key"] == "NIKE_DUNK_LOW")
    assert dunk["has_similar_items"] is True
    assert dunk["id"] == "paB0D93VKPPK"


def test_journey_simulate_validation(client):
    """POST /api/journey/simulate validates empty and malformed events."""
    # Empty events
    resp = client.post("/api/journey/simulate", json={"events": []})
    assert resp.status_code == 400

    # Search event missing query
    resp = client.post(
        "/api/journey/simulate",
        json={"events": [{"action_type": "search", "query": ""}]},
    )
    assert resp.status_code == 400

    # View event missing product_id
    resp = client.post(
        "/api/journey/simulate",
        json={"events": [{"action_type": "view", "product_id": ""}]},
    )
    assert resp.status_code == 400


def test_journey_simulate_mock_mode(client):
    """POST /api/journey/simulate succeeds with mock provider."""
    payload = {
        "events": [
            {"action_type": "view", "product_id": "paB07R17XQD2"},
            {"action_type": "search", "query": "running shoes"},
            {"action_type": "order", "product_id": "paB0D93VKPPK"},
        ],
        "limit": 5,
        "mode": "mock",
    }
    resp = client.post("/api/journey/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert data["event_count"] == 3
    assert len(data["items"]) == 5
    assert data["items"][0]["rank"] == 1
    assert "latency_ms" in data
    assert data["provenance"]["mode"] == "mock"


def test_journey_simulate_sdk_mode(client, monkeypatch):
    """POST /api/journey/simulate uses AsyncBehaviorGPTSDKAdapter in sdk mode."""
    from behavior_lab.models import InferenceOutput, PredictionItem
    from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter

    monkeypatch.setenv("UNBOXAI_API_KEY", "test_key")
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "__init__", lambda self, market="us", **kw: None)

    mock_predict = AsyncMock()
    mock_predict.return_value = InferenceOutput(
        case_id="journey_sim",
        variant="Journey Simulation",
        items=[
            PredictionItem(id="p1", name="Product 1", score=2.5, model_score=2.5, rank=1),
            PredictionItem(id="p2", name="Product 2", score=1.8, model_score=1.8, rank=2),
        ],
        latency_ms=25.0,
        provider="behaviorgpt_sdk_async",
        status_code=200,
    )
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "predict", mock_predict)
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "aclose", AsyncMock())

    payload = {
        "events": [{"action_type": "view", "product_id": "paB07R17XQD2"}],
        "limit": 2,
        "mode": "sdk",
    }
    resp = client.post("/api/journey/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "behaviorgpt_sdk_async"
    assert len(data["items"]) == 2
    assert data["items"][0]["model_score"] == 2.5
    mock_predict.assert_awaited_once()


def test_journey_simulate_upstream_error(client, monkeypatch):
    """POST /api/journey/simulate returns 502 Bad Gateway on upstream error."""
    from behavior_lab.models import InferenceOutput
    from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter

    monkeypatch.setenv("UNBOXAI_API_KEY", "test_key")
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "__init__", lambda self, market="us", **kw: None)

    mock_predict = AsyncMock()
    mock_predict.return_value = InferenceOutput(
        case_id="journey_sim",
        variant="Journey Simulation",
        items=[],
        latency_ms=5.0,
        provider="behaviorgpt_sdk_async",
        status_code=500,
        error_message="Internal Server Error from upstream BehaviorGPT",
    )
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "predict", mock_predict)
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "aclose", AsyncMock())

    payload = {
        "events": [{"action_type": "view", "product_id": "paB07R17XQD2"}],
        "mode": "sdk",
    }
    resp = client.post("/api/journey/simulate", json=payload)
    assert resp.status_code == 502
    assert "Inference failed" in resp.json()["detail"]


def test_counterfactual_compare_mock_mode(client):
    """POST /api/counterfactual/compare succeeds with mock mode and computes transitions."""
    payload = {
        "original_events": [{"action_type": "view", "product_id": "paB07R17XQD2"}],
        "mutated_events": [{"action_type": "view", "product_id": "paB0DK3YY682"}],
        "limit": 5,
        "mode": "mock",
        "mutation_description": "Cross-Category Swap",
    }
    resp = client.post("/api/counterfactual/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert "metrics" in data
    assert "jaccard_top5" in data["metrics"]
    assert "item_transitions" in data
    assert "shared" in data["item_transitions"]
    assert "evicted" in data["item_transitions"]
    assert "new" in data["item_transitions"]


def test_counterfactual_compare_sdk_mode_with_transitions(client, monkeypatch):
    """POST /api/counterfactual/compare correctly computes shared/evicted/new transitions and metrics."""
    from behavior_lab.models import InferenceOutput, PredictionItem
    from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter

    monkeypatch.setenv("UNBOXAI_API_KEY", "test_key")
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "__init__", lambda self, market="us", **kw: None)

    # Sequence A produces [A, B, C]
    # Sequence B produces [B, C, D]
    out_a = InferenceOutput(
        case_id="case_cf",
        variant="Original",
        items=[
            PredictionItem(id="item_A", name="Item A", score=3.0, model_score=3.0, rank=1),
            PredictionItem(id="item_B", name="Item B", score=2.0, model_score=2.0, rank=2),
            PredictionItem(id="item_C", name="Item C", score=1.0, model_score=1.0, rank=3),
        ],
        latency_ms=10.0,
        provider="behaviorgpt_sdk_async",
        status_code=200,
    )
    out_b = InferenceOutput(
        case_id="case_cf",
        variant="Counterfactual",
        items=[
            PredictionItem(id="item_B", name="Item B", score=2.5, model_score=2.5, rank=1),  # rank 2 -> 1
            PredictionItem(id="item_C", name="Item C", score=1.8, model_score=1.8, rank=2),  # rank 3 -> 2
            PredictionItem(id="item_D", name="Item D", score=1.2, model_score=1.2, rank=3),  # new
        ],
        latency_ms=12.0,
        provider="behaviorgpt_sdk_async",
        status_code=200,
    )

    mock_predict = AsyncMock(side_effect=[out_a, out_b])
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "predict", mock_predict)
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "aclose", AsyncMock())

    payload = {
        "original_events": [{"action_type": "view", "product_id": "item_1"}],
        "mutated_events": [{"action_type": "view", "product_id": "item_2"}],
        "limit": 3,
        "mode": "sdk",
    }
    resp = client.post("/api/counterfactual/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "behaviorgpt_sdk_async"

    # Verify transitions
    transitions = data["item_transitions"]
    assert transitions["shared_count"] == 2  # item_B and item_C
    assert transitions["evicted_count"] == 1  # item_A
    assert transitions["new_count"] == 1  # item_D

    evicted_ids = [it["product_id"] for it in transitions["evicted"]]
    assert "item_A" in evicted_ids

    new_ids = [it["product_id"] for it in transitions["new"]]
    assert "item_D" in new_ids

    shared_b = next(it for it in transitions["shared"] if it["product_id"] == "item_B")
    assert shared_b["original_rank"] == 2
    assert shared_b["counterfactual_rank"] == 1
    assert shared_b["rank_delta"] == -1  # moved up by 1 rank

    # Verify metrics
    metrics = data["metrics"]
    assert metrics["shared_item_count"] == 2
    assert metrics["rank_displacement_shared"] is not None
    assert metrics["rank_displacement_penalized"] > 0


def test_similar_items_mock_and_post(client):
    """GET and POST /api/items/{item_id}/similar work in mock mode."""
    # GET
    resp = client.get("/api/items/paB0D93VKPPK/similar")
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert data["seed_item"]["id"] == "paB0D93VKPPK"
    assert len(data["similar_items"]) == 10

    # POST
    resp_post = client.post(
        "/api/items/similar",
        json={"item_id": "paB0D93VKPPK", "limit": 4, "mode": "mock"},
    )
    assert resp_post.status_code == 200
    data_post = resp_post.json()
    assert len(data_post["similar_items"]) == 4
    assert data_post["similar_items"][0]["rank"] == 1


def test_similar_items_sdk_mode(client, monkeypatch):
    """GET /api/items/{item_id}/similar calls AsyncBehaviorGPTSDKAdapter.similar_items."""
    from behavior_lab.models import PredictionItem
    from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter

    monkeypatch.setenv("UNBOXAI_API_KEY", "test_key")
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "__init__", lambda self, market="us", **kw: None)

    mock_similar = AsyncMock()
    mock_similar.return_value = [
        PredictionItem(id="sim_1", name="Air Jordan 1", score=2.1, model_score=2.1, rank=1),
        PredictionItem(id="sim_2", name="Air Jordan 3", score=1.9, model_score=1.9, rank=2),
    ]
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "similar_items", mock_similar)
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "aclose", AsyncMock())

    from behavior_lab.api import get_similar_items_endpoint

    res = _drive_coro(get_similar_items_endpoint(item_id="paB0D93VKPPK", mode="sdk"))
    assert res["provider"] == "behaviorgpt_sdk_async"
    assert len(res["similar_items"]) == 2
    assert res["seed_item"]["key"] == "NIKE_DUNK_LOW"
    mock_similar.assert_awaited_once_with("paB0D93VKPPK", limit=10, catalog_id="retail_catalog_500k")


def test_similar_items_sdk_not_found_404(client, monkeypatch):
    """GET /api/items/{item_id}/similar returns 404 when item not found in catalog."""
    from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter
    from behaviorgpt._exceptions import UnboxAIError

    monkeypatch.setenv("UNBOXAI_API_KEY", "test_key")
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "__init__", lambda self, market="us", **kw: None)

    mock_similar = AsyncMock(side_effect=UnboxAIError("API Error: Item 'unknown_id' not found in catalog 'retail_catalog_500k'."))
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "similar_items", mock_similar)
    monkeypatch.setattr(AsyncBehaviorGPTSDKAdapter, "aclose", AsyncMock())

    from behavior_lab.api import get_similar_items_endpoint

    with pytest.raises(HTTPException) as exc_info:
        _drive_coro(get_similar_items_endpoint(item_id="unknown_id", mode="sdk"))
    assert exc_info.value.status_code == 404
    assert "not found in catalog" in exc_info.value.detail


# ==============================================================================
# PHASE 4: SHOWCASE SANDBOX REGRESSION TESTS (Method, Contract, 501, Errors, UX)
# ==============================================================================

def test_showcase_correct_http_method_and_payload(monkeypatch):
    """Verify BehaviorGPTClient sends POST to https://behaviorgpt.unboxai.com/api/recommendations with valid wire payload."""
    from behavior_lab.client import BehaviorGPTClient
    from behavior_lab.models import BehavioralAction, BehavioralHistory, ActionType
    import httpx

    captured_request = {}

    def mock_post(url, json=None, **kwargs):
        captured_request["url"] = url
        captured_request["method"] = "POST"
        captured_request["payload"] = json
        return httpx.Response(200, json={"items": [{"id": "item_1", "name": "Item 1", "prob": 0.5}]}, request=httpx.Request("POST", url))

    client = BehaviorGPTClient(mode="live")
    monkeypatch.setattr(client._http_client, "post", mock_post)

    history = BehavioralHistory(
        history_id="test_showcase",
        archetype="CUSTOM",
        description="Showcase Test History",
        created_at="2026-10-15T00:00:00Z",
        actions=[
            BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="paB08NYK61PJ", offset_ms=0),
            BehavioralAction(action_id="a2", action_type=ActionType.VIEW_ITEM, product_id="paB0CBT2582L", offset_ms=1000),
        ],
    )

    out = client.predict(history, limit=5, variant_label="Test")
    assert out.status_code == 200
    assert captured_request["url"] == "https://behaviorgpt.unboxai.com/api/recommendations"
    assert captured_request["method"] == "POST"

    payload = captured_request["payload"]
    assert "history" in payload
    assert payload["limit"] == 5
    assert len(payload["history"]) == 1
    session = payload["history"][0]
    assert session["type"] == "default"
    assert session["domains"] == {"market": "US"}
    assert len(session["events"]) == 2
    assert session["events"][0]["product"] == "paB08NYK61PJ"
    assert session["events"][0]["type"] == "viewItem"
    assert session["events"][1]["product"] == "paB0CBT2582L"


def test_showcase_successful_response_parsing(monkeypatch):
    """Verify BehaviorGPTClient parses 200 OK responses with full item attributes."""
    from behavior_lab.client import BehaviorGPTClient
    from behavior_lab.models import BehavioralAction, BehavioralHistory, ActionType
    import httpx

    mock_json = {
        "items": [
            {
                "id": "paB08NYK61PJ",
                "name": "Air Jordan 4 Retro",
                "prob": 0.045,
                "brand": "Nike",
                "category": "Footwear",
                "price": "215.00",
                "currency": "USD",
                "keywords": ["jordan", "retro"],
                "topTerms": [{"term": "sneaker", "weight": 0.9}],
            }
        ]
    }

    client = BehaviorGPTClient(mode="live")
    monkeypatch.setattr(
        client._http_client,
        "post",
        lambda url, **kw: httpx.Response(200, json=mock_json, request=httpx.Request("POST", url)),
    )

    history = BehavioralHistory(
        history_id="test_parse",
        archetype="CUSTOM",
        description="Showcase Parse History",
        created_at="2026-10-15T00:00:00Z",
        actions=[BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="paB08NYK61PJ", offset_ms=0)],
    )

    out = client.predict(history, limit=5)
    assert out.status_code == 200
    assert out.error_message is None
    assert len(out.items) == 1
    item = out.items[0]
    assert item.id == "paB08NYK61PJ"
    assert item.name == "Air Jordan 4 Retro"
    assert item.prob == 0.045
    assert item.score == 0.045
    assert item.rank == 1
    assert item.brand == "Nike"
    assert item.category == "Footwear"
    assert item.price == 215.0
    assert item.keywords == ["jordan", "retro"]


@pytest.mark.parametrize("status_code", [400, 401, 404, 500, 502, 503])
def test_showcase_http_4xx_and_5xx_responses(client, monkeypatch, status_code):
    """Verify /api/run-interactive returns HTTP 502 Bad Gateway with diagnostic status when upstream returns 4xx/5xx."""
    from behavior_lab.client import BehaviorGPTClient
    from behavior_lab.models import InferenceOutput

    def mock_predict(self, history, limit=10, variant_label="original", case_id="case_000"):
        return InferenceOutput(
            case_id=case_id,
            variant=variant_label,
            items=[],
            latency_ms=10.0,
            status_code=status_code,
            error_message=f"HTTP {status_code}: Upstream Server Error",
        )

    monkeypatch.setattr(BehaviorGPTClient, "predict", mock_predict)

    payload = {
        "history_a_items": ["paB08NYK61PJ"],
        "history_b_items": ["paB0CBT2582L"],
        "mode": "live",
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 502
    assert str(status_code) in resp.json().get("detail", "")


def test_showcase_unsupported_method_response_501(client, monkeypatch):
    """Verify handling when upstream returns HTTP 501, and verify FastAPI route method restrictions."""
    from behavior_lab.client import BehaviorGPTClient
    from behavior_lab.models import InferenceOutput

    # 1. Upstream 501 handling
    def mock_501_predict(self, history, limit=10, variant_label="original", case_id="case_000"):
        return InferenceOutput(
            case_id=case_id,
            variant=variant_label,
            items=[],
            latency_ms=8.0,
            status_code=501,
            error_message="HTTP 501: Not Implemented",
        )

    monkeypatch.setattr(BehaviorGPTClient, "predict", mock_501_predict)

    payload = {
        "history_a_items": ["paB08NYK61PJ"],
        "history_b_items": ["paB0CBT2582L"],
        "mode": "live",
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 502
    detail = resp.json().get("detail", "")
    assert "501" in detail

    # 2. Verify FastAPI route method restriction on /api/run-interactive
    interactive_route = next(
        (r for r in app.routes if getattr(r, "path", None) == "/api/run-interactive"),
        None,
    )
    assert interactive_route is not None
    assert interactive_route.methods == {"POST"}


def test_showcase_network_timeout_and_connection_failure(client, monkeypatch):
    """Verify TimeoutException and ConnectError in BehaviorGPTClient map to 502 Bad Gateway."""
    from behavior_lab.client import BehaviorGPTClient
    from behavior_lab.models import BehavioralAction, BehavioralHistory, ActionType
    import httpx

    # 1. TimeoutException
    def mock_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Read timed out after 5.0 seconds")

    test_client = BehaviorGPTClient(mode="live")
    monkeypatch.setattr(test_client._http_client, "post", mock_timeout)

    history = BehavioralHistory(
        history_id="test_timeout",
        archetype="CUSTOM",
        description="Timeout Test History",
        created_at="2026-10-15T00:00:00Z",
        actions=[BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="paB08NYK61PJ", offset_ms=0)],
    )
    out_timeout = test_client.predict(history)
    assert out_timeout.status_code == 500
    assert "TimeoutException" in out_timeout.error_message

    # 2. ConnectError
    def mock_connect_error(*args, **kwargs):
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(test_client._http_client, "post", mock_connect_error)
    out_connect = test_client.predict(history)
    assert out_connect.status_code == 500
    assert "ConnectError" in out_connect.error_message


def test_showcase_invalid_or_empty_response(monkeypatch):
    """Verify BehaviorGPTClient gracefully handles empty, missing, or malformed response bodies."""
    from behavior_lab.client import BehaviorGPTClient
    from behavior_lab.models import BehavioralAction, BehavioralHistory, ActionType
    import httpx

    client = BehaviorGPTClient(mode="live")
    history = BehavioralHistory(
        history_id="test_empty",
        archetype="CUSTOM",
        description="Empty Test History",
        created_at="2026-10-15T00:00:00Z",
        actions=[BehavioralAction(action_id="a1", action_type=ActionType.VIEW_ITEM, product_id="paB08NYK61PJ", offset_ms=0)],
    )

    # Empty JSON object {}
    monkeypatch.setattr(
        client._http_client,
        "post",
        lambda url, **kw: httpx.Response(200, json={}, request=httpx.Request("POST", url)),
    )
    out = client.predict(history)
    assert out.status_code == 200
    assert out.items == []

    # Items with missing fields (should apply defaults gracefully)
    monkeypatch.setattr(
        client._http_client,
        "post",
        lambda url, **kw: httpx.Response(
            200,
            json={"items": [{"id": "item_sparse"}, {"name": "No Price Item", "prob": 0.1}]},
            request=httpx.Request("POST", url),
        ),
    )
    out_sparse = client.predict(history)
    assert out_sparse.status_code == 200
    assert len(out_sparse.items) == 2
    assert out_sparse.items[0].id == "item_sparse"
    assert out_sparse.items[0].price is None
    assert out_sparse.items[1].name == "No Price Item"


def test_showcase_interactive_mock_mode(client):
    """Verify /api/run-interactive succeeds with mode='mock' without hitting external network."""
    payload = {
        "history_a_items": ["paB08NYK61PJ"],
        "history_b_items": ["paB0CBT2582L"],
        "mode": "mock",
        "limit": 5,
    }
    resp = client.post("/api/run-interactive", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert "sequence_a_output" in data
    assert "sequence_b_output" in data
    assert "metrics" in data
    assert data["sequence_a_output"]["status_code"] == 200
    assert data["sequence_b_output"]["status_code"] == 200


def test_showcase_frontend_diagnostics_and_error_states(client):
    """Verify web/index.html includes complete diagnostics, method descriptions, and error states."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # Diagnostic hints for static server HTTP 501
    assert "python -m http.server" in html
    assert "501 Unsupported method ('POST')" in html
    assert "uvicorn behavior_lab.api:app" in html

    # Showcase mode selector
    assert "sandboxMode" in html
    assert "Offline Mock Mode" in html
    assert "Public Showcase Endpoint (Live)" in html

    # Method and endpoint visibility
    assert "POST" in html
    assert "https://behaviorgpt.unboxai.com/api/recommendations" in html

    # Fallback and error states
    assert "fetchShowcaseDirect" in html
    assert "Public Showcase Endpoint — Direct Browser Fallback" in html
    assert "Live Inference Failed" in html
    assert "Live Inference Unsuccessful" in html


def test_regression_protection_sdk_backed_tools(client):
    """Confirm all three official SDK-backed tools remain fully operational without regression."""
    # 1. Behavioral Journey Simulator
    journey_resp = client.post(
        "/api/journey/simulate",
        json={"events": [{"action_type": "view", "product_id": "paB07R17XQD2"}], "mode": "mock", "limit": 3},
    )
    assert journey_resp.status_code == 200
    assert journey_resp.json()["provider"] == "mock"
    assert len(journey_resp.json()["items"]) == 3

    # 2. Counterfactual Playground
    cf_resp = client.post(
        "/api/counterfactual/compare",
        json={
            "original_events": [{"action_type": "view", "product_id": "paB07R17XQD2"}],
            "mutated_events": [{"action_type": "view", "product_id": "paB0DK3YY682"}],
            "mode": "mock",
            "limit": 3,
        },
    )
    assert cf_resp.status_code == 200
    assert cf_resp.json()["provider"] == "mock"
    assert "item_transitions" in cf_resp.json()

    # 3. Similar Item Explorer
    sim_resp = client.get("/api/items/paB0D93VKPPK/similar")
    assert sim_resp.status_code == 200
    assert sim_resp.json()["provider"] == "mock"
    assert len(sim_resp.json()["similar_items"]) == 10


def test_single_origin_architecture(client):
    """Verify single-origin application serving: index routes, health endpoint, and static mount."""
    # / and /index.html serve the dashboard
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "BehaviorGPT Counterfactual Sensitivity Lab" in resp_root.text
    assert "getApiBase" in resp_root.text
    assert "originWarningBanner" in resp_root.text
    assert "http://127.0.0.1:8000" in resp_root.text

    resp_index = client.get("/index.html")
    assert resp_index.status_code == 200
    assert resp_index.text == resp_root.text

    # /health endpoint returns 200 OK
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    data = resp_health.json()
    assert data["status"] == "ok"
    assert data["app"] == "BehaviorGPT Counterfactual Sensitivity Lab"

    # Static assets mount exists at /static without intercepting /api
    static_route = next((r for r in app.routes if getattr(r, "path", None) == "/static"), None)
    assert static_route is not None





