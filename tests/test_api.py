"""Automated tests for FastAPI endpoints in BehaviorGPT Counterfactual Sensitivity Lab."""

import pytest
from fastapi.testclient import TestClient
from behavior_lab.api import app


@pytest.fixture
def client():
    return TestClient(app)


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

