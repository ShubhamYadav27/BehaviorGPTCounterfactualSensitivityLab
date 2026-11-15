"""FastAPI application for BehaviorGPT Counterfactual Sensitivity Lab.

Provides REST endpoints to explore experiment results, inspect paired histories,
view metrics, examine investigated cases, and run live counterfactual queries.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

from behavior_lab.case_investigator import CaseInvestigator
from behavior_lab.client import BehaviorGPTClient
from behavior_lab.dataset_generator import VERIFIED_ITEMS
from behavior_lab.metrics import SensitivityMetricsCalculator
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    CounterfactualCompareRequest,
    ItemTransitionEvicted,
    ItemTransitionNew,
    ItemTransitionShared,
    ItemTransitionsSummary,
    JourneyEventInput,
    JourneySimulateRequest,
    PairedResult,
    PredictionItem,
    SimilarItemRequest,
)
from behavior_lab.sdk_adapter import resolve_base_timestamp

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application.

    Startup: Initializes application-level AsyncBehaviorGPTSDKAdapter if API key is present.
    Shutdown: Ensures async HTTP connections are cleanly closed.
    """
    app.state.sdk_adapter = None
    try:
        from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter, is_sdk_available

        if is_sdk_available() and os.environ.get("UNBOXAI_API_KEY"):
            app.state.sdk_adapter = AsyncBehaviorGPTSDKAdapter(market="us")
            logger.info("Application-level AsyncBehaviorGPTSDKAdapter pre-initialized.")
    except Exception as ex:
        logger.warning("Could not pre-initialize AsyncBehaviorGPTSDKAdapter: %s", ex)

    yield

    if getattr(app.state, "sdk_adapter", None) is not None:
        await app.state.sdk_adapter.aclose()
        logger.info("Application-level AsyncBehaviorGPTSDKAdapter closed cleanly.")


app = FastAPI(
    title="BehaviorGPT Counterfactual Sensitivity Lab",
    description="Empirical research workbench investigating sequential counterfactual sensitivity in Unbox AI's BehaviorGPT.",
    version="1.0.0",
    lifespan=lifespan,
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


def _events_to_history(
    events: List[JourneyEventInput],
    history_id: str = "journey_seq",
    description: str = "User Behavioral Sequence",
    created_at: Optional[str] = None,
) -> BehavioralHistory:
    """Convert user journey event inputs into a standardized BehavioralHistory."""
    actions: List[BehavioralAction] = []
    for i, ev in enumerate(events):
        offset = ev.offset_ms if ev.offset_ms is not None else i * 1000
        atype_str = (ev.action_type or "view").lower().strip()
        if atype_str in ("view", "view_item", "viewitem"):
            act_type = ActionType.VIEW_ITEM
        elif atype_str in ("order", "purchase"):
            act_type = ActionType.ORDER
        elif atype_str in ("search", "query"):
            act_type = ActionType.SEARCH
        elif atype_str in ("add_to_cart", "addtocart", "cart"):
            act_type = ActionType.ADD_TO_CART
        else:
            act_type = ActionType.VIEW_ITEM

        actions.append(
            BehavioralAction(
                action_id=f"act_{i}",
                action_type=act_type,
                product_id=ev.product_id,
                query=ev.query,
                offset_ms=offset,
            )
        )
    return BehavioralHistory(
        history_id=history_id,
        archetype="CUSTOM",
        description=description,
        actions=actions,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


def _compute_item_transitions(
    out_a: Any,
    out_b: Any,
    limit: int = 10,
) -> ItemTransitionsSummary:
    """Compute explicit shared, evicted, and new item rank/score transitions."""
    items_a = out_a.items[:limit] if hasattr(out_a, "items") else []
    items_b = out_b.items[:limit] if hasattr(out_b, "items") else []

    dict_a = {getattr(it, "id", None) or getattr(it, "product_id", ""): it for it in items_a}
    dict_b = {getattr(it, "id", None) or getattr(it, "product_id", ""): it for it in items_b}

    shared: List[ItemTransitionShared] = []
    for pid, it_a in dict_a.items():
        if pid in dict_b:
            it_b = dict_b[pid]
            s_a = getattr(it_a, "model_score", None) if getattr(it_a, "model_score", None) is not None else getattr(it_a, "score", None)
            s_b = getattr(it_b, "model_score", None) if getattr(it_b, "model_score", None) is not None else getattr(it_b, "score", None)
            delta_s = (s_b - s_a) if (s_a is not None and s_b is not None) else None
            shared.append(
                ItemTransitionShared(
                    product_id=pid,
                    name=getattr(it_a, "name", pid),
                    brand=getattr(it_a, "brand", None),
                    original_rank=it_a.rank,
                    counterfactual_rank=it_b.rank,
                    rank_delta=it_b.rank - it_a.rank,
                    original_score=s_a,
                    counterfactual_score=s_b,
                    score_delta=delta_s,
                )
            )

    evicted: List[ItemTransitionEvicted] = []
    for pid, it_a in dict_a.items():
        if pid not in dict_b:
            s_a = getattr(it_a, "model_score", None) if getattr(it_a, "model_score", None) is not None else getattr(it_a, "score", None)
            evicted.append(
                ItemTransitionEvicted(
                    product_id=pid,
                    name=getattr(it_a, "name", pid),
                    brand=getattr(it_a, "brand", None),
                    original_rank=it_a.rank,
                    original_score=s_a,
                )
            )

    new_items: List[ItemTransitionNew] = []
    for pid, it_b in dict_b.items():
        if pid not in dict_a:
            s_b = getattr(it_b, "model_score", None) if getattr(it_b, "model_score", None) is not None else getattr(it_b, "score", None)
            new_items.append(
                ItemTransitionNew(
                    product_id=pid,
                    name=getattr(it_b, "name", pid),
                    brand=getattr(it_b, "brand", None),
                    counterfactual_rank=it_b.rank,
                    counterfactual_score=s_b,
                )
            )

    return ItemTransitionsSummary(
        shared=shared,
        evicted=evicted,
        new=new_items,
        shared_count=len(shared),
        evicted_count=len(evicted),
        new_count=len(new_items),
    )


@app.get("/api/catalog/items")
def get_catalog_items():
    """Return verified items from retail_catalog_500k grouped by category with friendly metadata."""
    SIMILAR_SUPPORTED = {
        "paB0D93VKPPK", "paB0CJSYB1SY", "paB0CPSK1GVB", "paB0DK3YY682",
        "paB0DJ1VN1LF", "paB0859XT328", "paB0D8CSSQ7J", "paB0CRM723D6"
    }

    FRIENDLY_NAMES = {
        "NIKE_JORDAN4_BRED": "Air Jordan 4 Retro 'Bred'",
        "NIKE_PANTS": "Nike Sportswear Club Fleece Pants",
        "NIKE_PRO_SHORTS": "Nike Pro Compression Shorts",
        "NIKE_DUNK_LOW": "Nike Dunk Low Retro Sneaker",
        "NIKE_AIR_JORDAN_RETRO": "Air Jordan Retro High OG",
        "UNDER_ARMOUR_CHARGED": "Under Armour Charged Assert 9",
        "TIMBERLAND_BOOT": "Timberland White Ledge Waterproof Boot",
        "APPLE_IPAD_109": "Apple iPad 10.9-inch (10th Generation)",
        "SAMSUNG_GALAXY_TAB": "Samsung Galaxy Tab A9+ Tablet",
        "PS5_CONTROLLER_PEARL": "PlayStation 5 DualSense Wireless Controller (White)",
        "XBOX_WIRELESS_CONTROLLER": "Xbox Wireless Controller (Robot White)",
        "COACH_TABBY_BAG": "Coach Tabby Shoulder Bag 26",
        "COACH_SOFT_TABBY": "Coach Soft Tabby Hobo in Pebble Leather",
        "LOLOI_ACCENT_RUG": "Loloi II Layla Traditional Accent Rug",
        "YANKEE_CANDLE": "Yankee Candle Large Jar Candle (Balsam & Cedar)",
        "KEURIG_COFFEE": "Keurig K-Elite Single-Serve Coffee Maker",
    }

    items = []
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for key, info in VERIFIED_ITEMS.items():
        name = FRIENDLY_NAMES.get(key, key.replace("_", " ").title())
        item_entry = {
            "key": key,
            "id": info["id"],
            "name": name,
            "brand": info.get("brand"),
            "category": info.get("cat"),
            "price": info.get("price"),
            "has_similar_items": info["id"] in SIMILAR_SUPPORTED,
        }
        items.append(item_entry)
        cat = info.get("cat", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item_entry)

    return {
        "catalog_id": "retail_catalog_500k",
        "total_items": len(items),
        "items": items,
        "categories": categories,
    }


@app.post("/api/journey/simulate")
async def simulate_journey(req: JourneySimulateRequest):
    """Simulate a behavioral journey using the official SDK (or mock/showcase) and return recommendations."""
    if not req.events:
        raise HTTPException(status_code=400, detail="Journey must contain at least 1 event.")

    for i, ev in enumerate(req.events):
        atype = (ev.action_type or "view").lower().strip()
        if atype in ("search", "query") and not (ev.query and ev.query.strip()):
            raise HTTPException(status_code=400, detail=f"Event #{i+1} is a search action but has no query text.")
        if atype not in ("search", "query") and not (ev.product_id and ev.product_id.strip()):
            raise HTTPException(status_code=400, detail=f"Event #{i+1} is a {ev.action_type} action but has no product_id.")

    hist = _events_to_history(req.events, history_id="journey_sim", description="Simulated Journey")

    is_async = False
    should_close = False
    if req.mode == "sdk":
        from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter, is_sdk_available
        if not is_sdk_available():
            raise HTTPException(status_code=503, detail="Official behaviorgpt SDK is not installed.")
        shared_adapter = getattr(app.state, "sdk_adapter", None)
        if (
            shared_adapter is not None
            and getattr(shared_adapter, "_market", "").lower() == req.market.lower()
            and getattr(shared_adapter, "_catalog_id", "") == req.catalog_id
        ):
            provider = shared_adapter
            should_close = False
        else:
            try:
                provider = AsyncBehaviorGPTSDKAdapter(
                    market=req.market.lower(),
                    catalog_id=req.catalog_id,
                )
                should_close = True
            except Exception as ex:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize Async SDK adapter: {type(ex).__name__}: {str(ex)[:200]}",
                )
        provider_label = "behaviorgpt_sdk_async"
        is_async = True
    elif req.mode == "live":
        provider = BehaviorGPTClient(mode="live", market=req.market, catalog_id=req.catalog_id)
        provider_label = "showcase"
        is_async = False
        should_close = True
    else:  # mock
        provider = BehaviorGPTClient(mode="mock", market=req.market, catalog_id=req.catalog_id)
        provider_label = "mock"
        is_async = False
        should_close = True

    try:
        if is_async:
            output = await provider.predict(hist, limit=req.limit, variant_label="Journey Simulation")
        else:
            output = provider.predict(hist, limit=req.limit, variant_label="Journey Simulation")

        if output.status_code != 200 or output.error_message:
            msg = f"Inference failed ({provider_label})"
            if output.status_code:
                msg += f" [status {output.status_code}]"
            if output.error_message:
                first_line = output.error_message.split("\n")[0][:150]
                msg += f": {first_line}"
            raise HTTPException(status_code=502, detail=msg)

        return {
            "provider": provider_label,
            "model_name": output.model_name,
            "catalog_id": output.catalog_id,
            "latency_ms": output.latency_ms,
            "event_count": len(req.events),
            "items": [it.model_dump() for it in output.items],
            "provenance": {
                "server_timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": req.mode,
                "score_semantics": "uncalibrated_inner_product_model_score",
                "request_market": req.market,
            },
        }
    finally:
        if should_close:
            if is_async and hasattr(provider, "aclose"):
                await provider.aclose()
            elif hasattr(provider, "close"):
                provider.close()


@app.post("/api/counterfactual/compare")
async def compare_counterfactual(req: CounterfactualCompareRequest):
    """Run controlled counterfactual comparison between original and mutated journeys."""
    if not req.original_events or not req.mutated_events:
        raise HTTPException(
            status_code=400,
            detail="Both original and counterfactual sequences must contain at least 1 event.",
        )

    for i, ev in enumerate(req.original_events):
        atype = (ev.action_type or "view").lower().strip()
        if atype in ("search", "query") and not (ev.query and ev.query.strip()):
            raise HTTPException(status_code=400, detail=f"Original event #{i+1} is search but missing query text.")
        if atype not in ("search", "query") and not (ev.product_id and ev.product_id.strip()):
            raise HTTPException(status_code=400, detail=f"Original event #{i+1} is {ev.action_type} but missing product_id.")

    for i, ev in enumerate(req.mutated_events):
        atype = (ev.action_type or "view").lower().strip()
        if atype in ("search", "query") and not (ev.query and ev.query.strip()):
            raise HTTPException(status_code=400, detail=f"Mutated event #{i+1} is search but missing query text.")
        if atype not in ("search", "query") and not (ev.product_id and ev.product_id.strip()):
            raise HTTPException(status_code=400, detail=f"Mutated event #{i+1} is {ev.action_type} but missing product_id.")

    hist_a = _events_to_history(req.original_events, history_id="cf_orig", description="Original Journey")
    hist_b = _events_to_history(req.mutated_events, history_id="cf_mut", description="Counterfactual Journey")

    # Anchor shared base timestamp to guarantee exact temporal equivalence
    shared_base_time = resolve_base_timestamp()

    is_async = False
    should_close = False
    if req.mode == "sdk":
        from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter, is_sdk_available
        if not is_sdk_available():
            raise HTTPException(status_code=503, detail="Official behaviorgpt SDK is not installed.")
        shared_adapter = getattr(app.state, "sdk_adapter", None)
        if (
            shared_adapter is not None
            and getattr(shared_adapter, "_market", "").lower() == req.market.lower()
            and getattr(shared_adapter, "_catalog_id", "") == req.catalog_id
        ):
            provider = shared_adapter
            should_close = False
        else:
            try:
                provider = AsyncBehaviorGPTSDKAdapter(
                    market=req.market.lower(),
                    catalog_id=req.catalog_id,
                )
                should_close = True
            except Exception as ex:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize Async SDK adapter: {type(ex).__name__}: {str(ex)[:200]}",
                )
        provider_label = "behaviorgpt_sdk_async"
        is_async = True
    elif req.mode == "live":
        provider = BehaviorGPTClient(mode="live", market=req.market, catalog_id=req.catalog_id)
        provider_label = "showcase"
        is_async = False
        should_close = True
    else:  # mock
        provider = BehaviorGPTClient(mode="mock", market=req.market, catalog_id=req.catalog_id)
        provider_label = "mock"
        is_async = False
        should_close = True

    try:
        if is_async:
            out_a = await provider.predict(
                hist_a, limit=req.limit, variant_label="Original", base_time=shared_base_time
            )
            out_b = await provider.predict(
                hist_b, limit=req.limit, variant_label="Counterfactual", base_time=shared_base_time
            )
        else:
            out_a = provider.predict(hist_a, limit=req.limit, variant_label="Original")
            out_b = provider.predict(hist_b, limit=req.limit, variant_label="Counterfactual")

        if out_a.status_code != 200 or out_a.error_message:
            msg = f"Original sequence inference failed ({provider_label})"
            if out_a.status_code:
                msg += f" [status {out_a.status_code}]"
            if out_a.error_message:
                first_line = out_a.error_message.split("\n")[0][:120]
                msg += f": {first_line}"
            raise HTTPException(status_code=502, detail=msg)

        if out_b.status_code != 200 or out_b.error_message:
            msg = f"Counterfactual sequence inference failed ({provider_label})"
            if out_b.status_code:
                msg += f" [status {out_b.status_code}]"
            if out_b.error_message:
                first_line = out_b.error_message.split("\n")[0][:120]
                msg += f": {first_line}"
            raise HTTPException(status_code=502, detail=msg)

        metrics = SensitivityMetricsCalculator.compute(out_a, out_b)
        transitions = _compute_item_transitions(out_a, out_b, limit=req.limit)

        return {
            "provider": provider_label,
            "mutation_description": req.mutation_description,
            "sequence_a_output": out_a.model_dump(),
            "sequence_b_output": out_b.model_dump(),
            "metrics": metrics.model_dump(),
            "item_transitions": transitions.model_dump(),
            "provenance": {
                "server_timestamp": datetime.now(timezone.utc).isoformat(),
                "shared_base_timestamp": shared_base_time.isoformat(),
                "mode": req.mode,
                "score_semantics": "uncalibrated_inner_product_model_score",
                "request_market": req.market,
            },
        }
    finally:
        if should_close:
            if is_async and hasattr(provider, "aclose"):
                await provider.aclose()
            elif hasattr(provider, "close"):
                provider.close()


@app.get("/api/items/{item_id}/similar")
async def get_similar_items_endpoint(
    item_id: str,
    limit: int = 10,
    catalog_id: str = "retail_catalog_500k",
    market: str = "us",
    mode: str = "sdk",
):
    """Retrieve items similar to item_id in embedding space using official SDK similar_items()."""
    return await _resolve_similar_items(
        item_id=item_id,
        limit=limit,
        catalog_id=catalog_id,
        market=market,
        mode=mode,
    )


@app.post("/api/items/similar")
async def post_similar_items_endpoint(req: SimilarItemRequest):
    """Retrieve items similar to item_id via POST payload."""
    return await _resolve_similar_items(
        item_id=req.item_id,
        limit=req.limit,
        catalog_id=req.catalog_id,
        market=req.market,
        mode=req.mode,
    )


async def _resolve_similar_items(
    item_id: str,
    limit: int = 10,
    catalog_id: str = "retail_catalog_500k",
    market: str = "us",
    mode: str = "sdk",
) -> Dict[str, Any]:
    if not item_id or not item_id.strip():
        raise HTTPException(status_code=400, detail="item_id is required.")

    clean_id = item_id.strip()

    # Seed item lookup
    seed_info = {"id": clean_id, "name": clean_id, "brand": "Unknown", "category": "General", "price": None}
    for k, v in VERIFIED_ITEMS.items():
        if v["id"] == clean_id:
            seed_info = {
                "id": clean_id,
                "key": k,
                "name": k.replace("_", " ").title(),
                "brand": v.get("brand"),
                "category": v.get("cat"),
                "price": v.get("price"),
            }
            break

    start = time.perf_counter()

    if mode == "sdk":
        from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter, is_sdk_available
        if not is_sdk_available():
            raise HTTPException(status_code=503, detail="Official behaviorgpt SDK is not installed.")

        shared_adapter = getattr(app.state, "sdk_adapter", None)
        should_close = False
        if (
            shared_adapter is not None
            and getattr(shared_adapter, "_market", "").lower() == market.lower()
            and getattr(shared_adapter, "_catalog_id", "") == catalog_id
        ):
            provider = shared_adapter
        else:
            try:
                provider = AsyncBehaviorGPTSDKAdapter(market=market.lower(), catalog_id=catalog_id)
                should_close = True
            except Exception as ex:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize Async SDK adapter: {type(ex).__name__}: {str(ex)[:200]}",
                )

        try:
            items = await provider.similar_items(clean_id, limit=limit, catalog_id=catalog_id)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            provider_label = "behaviorgpt_sdk_async"
        except Exception as ex:
            err_msg = str(ex)
            if "not found" in err_msg.lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Item '{clean_id}' was not found in catalog '{catalog_id}'.",
                )
            logger.exception("Error in SDK similar_items: %s", ex)
            raise HTTPException(
                status_code=502,
                detail=f"SDK similar_items failed: {type(ex).__name__}: {err_msg[:200]}",
            )
        finally:
            if should_close:
                await provider.aclose()

    elif mode == "mock":
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        provider_label = "mock"
        items = [
            PredictionItem(
                id=f"sim_{clean_id}_{r}",
                name=f"Similar Product #{r} to {clean_id}",
                score=round(2.0 - (r * 0.15), 4),
                model_score=round(2.0 - (r * 0.15), 4),
                rank=r,
                brand=seed_info.get("brand", "Generic"),
                category=seed_info.get("category", "General"),
                price=50.0 + (r * 5),
            )
            for r in range(1, limit + 1)
        ]
    else:
        raise HTTPException(
            status_code=400,
            detail="similar_items() is a feature of the official BehaviorGPT SDK (mode='sdk' or mode='mock').",
        )

    return {
        "seed_item": seed_info,
        "similar_items": [it.model_dump() for it in items],
        "total_returned": len(items),
        "limit": limit,
        "catalog_id": catalog_id,
        "provider": provider_label,
        "latency_ms": elapsed_ms,
        "provenance": {
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "score_semantics": "uncalibrated_inner_product_model_score",
        },
    }



@app.get("/api/summary")
def get_summary():
    """Return experiment summary with explicit labeling for unverified local metadata."""
    summary = _load_summary()
    summary["metadata_provenance"] = {
        "model_name_status": "unverified_local_descriptor",
        "catalog_id_status": "unverified_local_descriptor",
        "note": (
            "The model version ('BehaviorGPT-v4-12.5B') and catalog identifier "
            "('retail_catalog_500k') are local descriptor labels based on public "
            "marketing materials. They are not confirmed by server response metadata "
            "from the public showcase endpoint."
        ),
    }
    summary["is_model_name_verified"] = False
    summary["is_catalog_id_verified"] = False
    return summary


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
        "investigated_cases": list(reports.keys()),
    }


class InteractiveHistoryRequest(BaseModel):
    history_a_items: List[str]
    history_b_items: List[str]
    market: str = "US"
    limit: int = 5
    mode: str = "live"  # "live" (showcase), "sdk" (official SDK)


@app.post("/api/run-interactive")
async def run_interactive(req: InteractiveHistoryRequest):
    """Run interactive live counterfactual comparison between two sequences.

    Supports mode="live" (showcase endpoint) or mode="sdk" (official SDK via AsyncBehaviorGPTSDKAdapter).
    Returns HTTP 502 Bad Gateway if either sequence's upstream inference fails,
    avoiding the calculation of false zero-valued sensitivity metrics.
    """
    should_close_provider = False
    is_async = False

    if req.mode == "sdk":
        from behavior_lab.sdk_adapter import AsyncBehaviorGPTSDKAdapter, is_sdk_available

        if not is_sdk_available():
            raise HTTPException(
                status_code=503,
                detail="SDK mode requested but behaviorgpt package is not installed.",
            )
        shared_adapter = getattr(app.state, "sdk_adapter", None)
        if shared_adapter is not None and getattr(shared_adapter, "_market", "").lower() == req.market.lower():
            provider = shared_adapter
            should_close_provider = False
        else:
            try:
                provider = AsyncBehaviorGPTSDKAdapter(market=req.market.lower())
                should_close_provider = True
            except Exception as ex:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize Async SDK: {type(ex).__name__}: {str(ex)[:200]}",
                )
        provider_label = "behaviorgpt_sdk"
        is_async = True
    else:
        provider = BehaviorGPTClient(mode="live", market=req.market)
        provider_label = "showcase"
        is_async = False
        should_close_provider = True

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

    try:
        try:
            if is_async:
                out_a = await provider.predict(hist_a, limit=req.limit, variant_label="Sequence A")
            else:
                out_a = provider.predict(hist_a, limit=req.limit, variant_label="Sequence A")
        except Exception as ex:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream {provider_label} connection failed for Sequence A: {type(ex).__name__}",
            )

        if out_a.status_code != 200 or out_a.error_message:
            msg = f"Upstream {provider_label} returned non-success response for Sequence A"
            if out_a.status_code:
                msg += f" (status {out_a.status_code})"
            if out_a.error_message:
                first_line = out_a.error_message.split("\n")[0][:120]
                msg += f": {first_line}"
            raise HTTPException(status_code=502, detail=msg)

        try:
            if is_async:
                out_b = await provider.predict(hist_b, limit=req.limit, variant_label="Sequence B")
            else:
                out_b = provider.predict(hist_b, limit=req.limit, variant_label="Sequence B")
        except Exception as ex:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream {provider_label} connection failed for Sequence B: {type(ex).__name__}",
            )

        if out_b.status_code != 200 or out_b.error_message:
            msg = f"Upstream {provider_label} returned non-success response for Sequence B"
            if out_b.status_code:
                msg += f" (status {out_b.status_code})"
            if out_b.error_message:
                first_line = out_b.error_message.split("\n")[0][:120]
                msg += f": {first_line}"
            raise HTTPException(status_code=502, detail=msg)

        metrics = SensitivityMetricsCalculator.compute(out_a, out_b)

        return {
            "provider": provider_label,
            "sequence_a_output": out_a.model_dump(),
            "sequence_b_output": out_b.model_dump(),
            "metrics": metrics.model_dump(),
        }
    finally:
        if should_close_provider:
            if is_async and hasattr(provider, "aclose"):
                await provider.aclose()
            elif hasattr(provider, "close"):
                provider.close()


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dashboard_path = Path("web/index.html")
    if not dashboard_path.exists():
        return HTMLResponse("<h1>Dashboard under construction</h1>", status_code=200)
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
