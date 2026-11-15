"""Sensitivity and divergence metrics for BehaviorGPT counterfactual comparisons.

Defines exact mathematical formulations for:
- Top-1 Invariance / Flip Rate
- Jaccard Similarity (Top-3, Top-5, Top-10)
- Mean Rank Displacement
- Spearman Rank Correlation on shared items
- Probability Divergence (TVD on shared probabilities)
- Mean Price Delta & Category Drift
"""

from typing import List, Optional, Set
from scipy.stats import spearmanr
from behavior_lab.models import (
    InferenceOutput,
    PredictionItem,
    SensitivityMetrics,
)


class SensitivityMetricsCalculator:
    """Computes rigorous sensitivity metrics between paired model inference outputs."""

    @classmethod
    def compute(
        cls, orig_out: InferenceOutput, mut_out: InferenceOutput
    ) -> SensitivityMetrics:
        orig_items = orig_out.items
        mut_items = mut_out.items

        if not orig_items or not mut_items:
            return SensitivityMetrics(
                top1_changed=False,
                jaccard_top3=0.0,
                jaccard_top5=0.0,
                jaccard_top10=0.0,
                shared_item_count=0,
                rank_displacement_shared=None,
                rank_displacement_penalized=0.0,
                rank_displacement_avg=None,
            )

        top1_orig = orig_items[0]
        top1_mut = mut_items[0]
        top1_changed = top1_orig.id != top1_mut.id

        # Probability delta (only if calibrated probabilities exist in [0, 1])
        top1_prob_delta = None
        if (
            top1_orig.prob is not None
            and top1_mut.prob is not None
            and 0.0 <= top1_orig.prob <= 1.0
            and 0.0 <= top1_mut.prob <= 1.0
        ):
            top1_prob_delta = abs(top1_orig.prob - top1_mut.prob)

        # Raw model score delta (for uncalibrated model scores)
        orig_score = getattr(top1_orig, "model_score", None) if getattr(top1_orig, "model_score", None) is not None else top1_orig.score
        mut_score = getattr(top1_mut, "model_score", None) if getattr(top1_mut, "model_score", None) is not None else top1_mut.score
        top1_score_delta = abs(orig_score - mut_score) if orig_score is not None and mut_score is not None else None

        jaccard_3 = cls._jaccard([it.id for it in orig_items[:3]], [it.id for it in mut_items[:3]])
        jaccard_5 = cls._jaccard([it.id for it in orig_items[:5]], [it.id for it in mut_items[:5]])
        jaccard_10 = cls._jaccard([it.id for it in orig_items[:10]], [it.id for it in mut_items[:10]])

        # Rank mappings
        rank_orig = {it.id: it.rank for it in orig_items}
        rank_mut = {it.id: it.rank for it in mut_items}
        shared_ids = set(rank_orig.keys()) & set(rank_mut.keys())
        shared_count = len(shared_ids)

        # 1. Rank displacement conditioned on shared items (None if 0 shared items)
        rank_disp_shared: Optional[float] = None
        spearman_corr: Optional[float] = None
        if shared_count > 0:
            displacements = [abs(rank_orig[sid] - rank_mut[sid]) for sid in shared_ids]
            rank_disp_shared = sum(displacements) / shared_count

            if shared_count >= 3:
                ranks_a = [rank_orig[sid] for sid in shared_ids]
                ranks_b = [rank_mut[sid] for sid in shared_ids]
                res = spearmanr(ranks_a, ranks_b)
                spearman_corr = float(res.correlation) if res.correlation is not None else None

        # 2. Penalized rank displacement across all items in either top-K list
        # Items evicted from or newly appearing in top-K receive penalty rank K + 1
        all_ids = set(rank_orig.keys()) | set(rank_mut.keys())
        k_val = max(len(orig_items), len(mut_items), 1)
        penalty_rank = k_val + 1

        penalized_disp_sum = 0.0
        for item_id in all_ids:
            r_a = rank_orig.get(item_id, penalty_rank)
            r_b = rank_mut.get(item_id, penalty_rank)
            penalized_disp_sum += abs(r_a - r_b)
        rank_disp_penalized = penalized_disp_sum / len(all_ids) if all_ids else 0.0

        # Probability Total Variation Distance (TVD) on shared items (strictly for probs in [0, 1])
        valid_prob_orig = {
            it.id: it.prob for it in orig_items
            if it.prob is not None and 0.0 <= it.prob <= 1.0
        }
        valid_prob_mut = {
            it.id: it.prob for it in mut_items
            if it.prob is not None and 0.0 <= it.prob <= 1.0
        }
        prob_shared_ids = set(valid_prob_orig.keys()) & set(valid_prob_mut.keys())

        prob_tvd = None
        if prob_shared_ids:
            prob_tvd = 0.5 * sum(abs(valid_prob_orig[sid] - valid_prob_mut[sid]) for sid in prob_shared_ids)

        # Raw score L1 distance on shared items (for uncalibrated model scores)
        score_orig = {
            it.id: (getattr(it, "model_score", None) if getattr(it, "model_score", None) is not None else it.score)
            for it in orig_items
        }
        score_mut = {
            it.id: (getattr(it, "model_score", None) if getattr(it, "model_score", None) is not None else it.score)
            for it in mut_items
        }
        score_shared_ids = set(score_orig.keys()) & set(score_mut.keys())
        score_l1 = None
        if score_shared_ids:
            score_l1 = sum(abs(score_orig[sid] - score_mut[sid]) for sid in score_shared_ids) / len(score_shared_ids)

        # Price drift
        prices_orig = [it.price for it in orig_items if it.price is not None]
        prices_mut = [it.price for it in mut_items if it.price is not None]
        price_delta = None
        if prices_orig and prices_mut:
            mean_orig = sum(prices_orig) / len(prices_orig)
            mean_mut = sum(prices_mut) / len(prices_mut)
            price_delta = mean_mut - mean_orig

        # Category shift
        cat_orig = top1_orig.category
        cat_mut = top1_mut.category
        category_shifted = bool(cat_orig and cat_mut and cat_orig != cat_mut)

        return SensitivityMetrics(
            top1_changed=top1_changed,
            top1_original_id=top1_orig.id,
            top1_mutated_id=top1_mut.id,
            top1_prob_delta=round(top1_prob_delta, 6) if top1_prob_delta is not None else None,
            top1_score_delta=round(top1_score_delta, 6) if top1_score_delta is not None else None,
            jaccard_top3=round(jaccard_3, 4),
            jaccard_top5=round(jaccard_5, 4),
            jaccard_top10=round(jaccard_10, 4),
            shared_item_count=shared_count,
            rank_displacement_shared=round(rank_disp_shared, 4) if rank_disp_shared is not None else None,
            rank_displacement_penalized=round(rank_disp_penalized, 4),
            rank_displacement_avg=round(rank_disp_shared, 4) if rank_disp_shared is not None else None,
            spearman_correlation=round(spearman_corr, 4) if spearman_corr is not None else None,
            prob_tvd_shared=round(prob_tvd, 6) if prob_tvd is not None else None,
            score_l1_shared=round(score_l1, 6) if score_l1 is not None else None,
            price_mean_delta=round(price_delta, 2) if price_delta is not None else None,
            category_shifted=category_shifted,
        )

    @staticmethod
    def _jaccard(list_a: List[str], list_b: List[str]) -> float:
        set_a, set_b = set(list_a), set(list_b)
        union = set_a | set_b
        if not union:
            return 1.0  # Both empty lists are identical sets
        if not set_a or not set_b:
            return 0.0  # One empty, one non-empty
        return len(set_a & set_b) / len(union)
