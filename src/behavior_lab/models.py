"""Pydantic models and schemas for BehaviorGPT Counterfactual Sensitivity Lab."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    VIEW_ITEM = "viewItem"
    ADD_TO_CART = "addToCart"
    SEARCH = "search"


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
    status_code: int = 200
    error_message: Optional[str] = None

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
    top1_prob_delta: Optional[float] = None
    jaccard_top3: float
    jaccard_top5: float
    jaccard_top10: float
    rank_displacement_avg: float
    spearman_correlation: Optional[float] = None
    prob_tvd_shared: Optional[float] = None  # Total variation distance on shared items
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
    execution_mode: str  # "live" or "mock"
    top1_change_rate: float
    avg_jaccard_top5: float
    avg_jaccard_top10: float
    avg_rank_displacement: float
    avg_prob_tvd: float
    by_mutation_type: Dict[str, Dict[str, float]]
    by_position: Dict[str, Dict[str, float]]
    investigated_case_id: Optional[str] = None
