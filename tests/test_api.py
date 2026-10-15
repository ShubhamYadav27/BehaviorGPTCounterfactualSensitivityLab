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
