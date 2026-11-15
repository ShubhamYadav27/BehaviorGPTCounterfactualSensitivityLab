"""Pydantic models and schemas for BehaviorGPT Counterfactual Sensitivity Lab."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    VIEW_ITEM = "viewItem"
    ADD_TO_CART = "addToCart"
    SEARCH = "search"
    ORDER = "order"


class MutationType(str, Enum):
    ITEM_REPLACEMENT_CROSS_CATEGORY = "cross_category_swap"
    ITEM_REPLACEMENT_WITHIN_CATEGORY = "within_category_swap"
    ACTION_TYPE_SWAP = "action_type_swap"
    BRAND_SWAP = "brand_swap"
    POSITION_EARLY = "mutation_position_early"
    POSITION_LATE = "mutation_position_late"


class MutationPosition(str, Enum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class BehavioralAction(BaseModel):
    action_id: str
    action_type: ActionType
    product_id: Optional[str] = None
    query: Optional[str] = None
    offset_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_wire_dict(self) -> Dict[str, Any]:
        """Convert to the wire event format expected by BehaviorGPT."""
        if self.action_type == ActionType.SEARCH:
            return {"type": "search", "query": self.query or ""}
        elif self.action_type == ActionType.ADD_TO_CART:
            return {"type": "addToCart", "product": self.product_id or ""}
        elif self.action_type == ActionType.ORDER:
            return {"type": "order", "product": self.product_id or ""}
        else:
            return {"type": "viewItem", "product": self.product_id or ""}


class BehavioralHistory(BaseModel):
    history_id: str
    archetype: str
    description: str
    actions: List[BehavioralAction]
    market: str = "US"
    created_at: str
    is_synthetic: bool = True

    @property
    def length(self) -> int:
        return len(self.actions)


class MutationSpec(BaseModel):
    case_id: str
    mutation_type: MutationType
    position_category: MutationPosition
    position_index: int
    original_action: BehavioralAction
    mutated_action: BehavioralAction
    rationale: str


class PairedExperimentCase(BaseModel):
    case_id: str
    history_id: str
    archetype: str
    original_history: BehavioralHistory
    mutated_history: BehavioralHistory
    mutation_spec: MutationSpec


class PredictionItem(BaseModel):
    id: str
    name: str
    score: float
    model_score: Optional[float] = None
    prob: Optional[float] = None
    rank: int
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "USD"
    keywords: List[str] = Field(default_factory=list)
    top_terms: List[Dict[str, Any]] = Field(default_factory=list)


class InferenceOutput(BaseModel):
    case_id: str
    variant: str  # "original" or "mutated"
    items: List[PredictionItem]
    latency_ms: float
    model_name: str = "BehaviorGPT-v4-12.5B"
    catalog_id: str = "retail_catalog_500k"
    is_mock: bool = False
    provider: str = "showcase"  # "showcase" or "behaviorgpt_sdk"
    status_code: int = 200
    error_message: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None

    @property
    def top1(self) -> Optional[PredictionItem]:
        return self.items[0] if self.items else None

    @property
    def item_ids(self) -> List[str]:
        return [it.id for it in self.items]


class SensitivityMetrics(BaseModel):
    top1_changed: bool
    top1_original_id: Optional[str] = None
    top1_mutated_id: Optional[str] = None
    top1_prob_delta: Optional[float] = None  # Populated only when calibrated probabilities are present
    top1_score_delta: Optional[float] = None  # Delta between raw model scores
    jaccard_top3: float
    jaccard_top5: float
    jaccard_top10: float
    shared_item_count: int = 0
    rank_displacement_shared: Optional[float] = None  # Rank displacement over shared items (None when disjoint)
    rank_displacement_penalized: float = 0.0  # Penalized rank displacement including evicted & new items
    rank_displacement_avg: Optional[float] = None  # Kept for backward compatibility (None when disjoint)
    spearman_correlation: Optional[float] = None
    prob_tvd_shared: Optional[float] = None  # Total variation distance on shared items (valid on probs in [0, 1])
    score_l1_shared: Optional[float] = None  # Mean absolute L1 delta on raw model scores for shared items
    price_mean_delta: Optional[float] = None
    category_shifted: bool = False


class PairedResult(BaseModel):
    case_id: str
    archetype: str
    mutation_type: MutationType
    position_category: MutationPosition
    position_index: int
    history_length: int
    original_output: InferenceOutput
    mutated_output: InferenceOutput
    metrics: SensitivityMetrics


class ExperimentSummary(BaseModel):
    experiment_id: str
    timestamp: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    model_name: str
    catalog_id: str
    execution_mode: str  # "live", "mock", or "sdk"
    provider: str = "showcase"  # "showcase" or "behaviorgpt_sdk"
    top1_change_rate: float
    avg_jaccard_top5: float
    avg_jaccard_top10: float
    avg_rank_displacement: float
    avg_rank_displacement_penalized: Optional[float] = None
    avg_prob_tvd: float
    avg_score_l1: Optional[float] = None
    by_mutation_type: Dict[str, Dict[str, float]]
    by_position: Dict[str, Dict[str, float]]
    investigated_case_id: Optional[str] = None


class JourneyEventInput(BaseModel):
    action_type: str = "view"  # "view", "search", "order", "addToCart"
    product_id: Optional[str] = None
    query: Optional[str] = None
    offset_ms: Optional[int] = None


class JourneySimulateRequest(BaseModel):
    events: List[JourneyEventInput]
    limit: int = 10
    market: str = "us"
    mode: str = "sdk"  # "sdk", "live", "mock"
    catalog_id: str = "retail_catalog_500k"


class CounterfactualCompareRequest(BaseModel):
    original_events: List[JourneyEventInput]
    mutated_events: List[JourneyEventInput]
    limit: int = 10
    market: str = "us"
    mode: str = "sdk"  # "sdk", "live", "mock"
    mutation_description: Optional[str] = "Counterfactual Single-Action Mutation"
    catalog_id: str = "retail_catalog_500k"


class SimilarItemRequest(BaseModel):
    item_id: str
    limit: int = 10
    catalog_id: str = "retail_catalog_500k"
    market: str = "us"
    mode: str = "sdk"


class ItemTransitionShared(BaseModel):
    product_id: str
    name: str
    brand: Optional[str] = None
    original_rank: int
    counterfactual_rank: int
    rank_delta: int  # counterfactual_rank - original_rank
    original_score: Optional[float] = None
    counterfactual_score: Optional[float] = None
    score_delta: Optional[float] = None


class ItemTransitionEvicted(BaseModel):
    product_id: str
    name: str
    brand: Optional[str] = None
    original_rank: int
    original_score: Optional[float] = None


class ItemTransitionNew(BaseModel):
    product_id: str
    name: str
    brand: Optional[str] = None
    counterfactual_rank: int
    counterfactual_score: Optional[float] = None


class ItemTransitionsSummary(BaseModel):
    shared: List[ItemTransitionShared] = Field(default_factory=list)
    evicted: List[ItemTransitionEvicted] = Field(default_factory=list)
    new: List[ItemTransitionNew] = Field(default_factory=list)
    shared_count: int = 0
    evicted_count: int = 0
    new_count: int = 0

