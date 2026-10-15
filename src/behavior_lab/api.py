"""FastAPI application for BehaviorGPT Counterfactual Sensitivity Lab.

Provides REST endpoints to explore experiment results, inspect paired histories,
view metrics, examine investigated cases, and run live counterfactual queries.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from behavior_lab.case_investigator import CaseInvestigator
from behavior_lab.client import BehaviorGPTClient
from behavior_lab.metrics import SensitivityMetricsCalculator
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    PairedResult,
)

app = FastAPI(
    title="BehaviorGPT Counterfactual Sensitivity Lab",
    description="Empirical research workbench investigating sequential counterfactual sensitivity in Unbox AI's BehaviorGPT.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")


def _load_summary() -> Dict[str, Any]:
    path = DATA_DIR / "experiment_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not found. Run experiment first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_results() -> List[Dict[str, Any]]:
    path = DATA_DIR / "experiment_results.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Results not found. Run experiment first.")
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


@app.get("/api/summary")
def get_summary():
    return _load_summary()


@app.get("/api/cases")
def get_cases(limit: int = 50):
    results = _load_results()
    cases_summary = []
    for r in results[:limit]:
        cases_summary.append({
            "case_id": r["case_id"],
            "archetype": r["archetype"],
            "mutation_type": r["mutation_type"],
            "position_category": r["position_category"],
            "position_index": r["position_index"],
            "history_length": r["history_length"],
            "top1_changed": r["metrics"]["top1_changed"],
            "jaccard_top5": r["metrics"]["jaccard_top5"],
            "rank_displacement": r["metrics"]["rank_displacement_avg"],
            "original_top1": r["original_output"]["items"][0]["name"] if r["original_output"]["items"] else None,
            "mutated_top1": r["mutated_output"]["items"][0]["name"] if r["mutated_output"]["items"] else None,
        })
    return {"total": len(results), "cases": cases_summary}


@app.get("/api/cases/{case_id}")
def get_case_detail(case_id: str):
    results = _load_results()
    for r in results:
        if r["case_id"] == case_id:
            # Also load raw paired spec if available
            spec_path = DATA_DIR / "paired_experiment_cases.jsonl"
            paired_spec = None
            if spec_path.exists():
                with open(spec_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["case_id"] == case_id:
                            paired_spec = data
                            break
            return {"result": r, "spec": paired_spec}
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")


@app.get("/api/investigation")
def get_investigation():
    investigator = CaseInvestigator(results_path=str(DATA_DIR / "experiment_results.jsonl"))
    noteworthy = investigator.find_noteworthy_cases()
    reports = {}
    if noteworthy.get("high_sensitivity"):
        reports["high_sensitivity"] = noteworthy["high_sensitivity"].model_dump()
    if noteworthy.get("high_inertia"):
        reports["high_inertia"] = noteworthy["high_inertia"].model_dump()
    if noteworthy.get("position_contrast"):
        reports["position_contrast"] = noteworthy["position_contrast"].model_dump()

    # Read markdown analysis file if present
    md_content = ""
    md_path = DATA_DIR / "investigated_case.md"
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

    return {
        "reports": reports,
        "markdown_analysis": md_content,
        "investigated_cases": ["case_004 (Early Position Sensitivity)", "case_003 (Action Verb Invariance)"],
    }


class InteractiveHistoryRequest(BaseModel):
    history_a_items: List[str]
    history_b_items: List[str]
    market: str = "US"
    limit: int = 5


@app.post("/api/run-interactive")
def run_interactive(req: InteractiveHistoryRequest):
    """Run interactive live counterfactual comparison between two sequences."""
    client = BehaviorGPTClient(mode="live", market=req.market)

    hist_a = BehavioralHistory(
        history_id="interactive_a",
        archetype="CUSTOM",
        description="Interactive Sequence A",
        actions=[
            BehavioralAction(
                action_id=f"a_{i}",
                action_type=ActionType.VIEW_ITEM,
                product_id=pid,
                offset_ms=i * 1000,
            )
            for i, pid in enumerate(req.history_a_items)
        ],
        created_at="2026-10-15T00:00:00Z",
    )

    hist_b = BehavioralHistory(
        history_id="interactive_b",
        archetype="CUSTOM",
        description="Interactive Sequence B",
        actions=[
            BehavioralAction(
                action_id=f"b_{i}",
                action_type=ActionType.VIEW_ITEM,
                product_id=pid,
                offset_ms=i * 1000,
            )
            for i, pid in enumerate(req.history_b_items)
        ],
        created_at="2026-10-15T00:00:00Z",
    )

    out_a = client.predict(hist_a, limit=req.limit, variant_label="Sequence A")
    out_b = client.predict(hist_b, limit=req.limit, variant_label="Sequence B")
    metrics = SensitivityMetricsCalculator.compute(out_a, out_b)

    return {
        "sequence_a_output": out_a.model_dump(),
        "sequence_b_output": out_b.model_dump(),
        "metrics": metrics.model_dump(),
    }


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dashboard_path = Path("web/index.html")
    if not dashboard_path.exists():
        return HTMLResponse("<h1>Dashboard under construction</h1>", status_code=200)
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
