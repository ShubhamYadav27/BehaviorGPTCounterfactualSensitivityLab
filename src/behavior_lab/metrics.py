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
                rank_displacement_avg=0.0,
            )

        top1_orig = orig_items[0]
        top1_mut = mut_items[0]
        top1_changed = top1_orig.id != top1_mut.id

        top1_prob_delta = None
        if top1_orig.prob is not None and top1_mut.prob is not None:
            top1_prob_delta = abs(top1_orig.prob - top1_mut.prob)

        jaccard_3 = cls._jaccard([it.id for it in orig_items[:3]], [it.id for it in mut_items[:3]])
        jaccard_5 = cls._jaccard([it.id for it in orig_items[:5]], [it.id for it in mut_items[:5]])
        jaccard_10 = cls._jaccard([it.id for it in orig_items[:10]], [it.id for it in mut_items[:10]])

        # Rank displacement & Spearman correlation on shared items
        rank_orig = {it.id: it.rank for it in orig_items}
        rank_mut = {it.id: it.rank for it in mut_items}
        shared_ids = set(rank_orig.keys()) & set(rank_mut.keys())

        rank_disp = 0.0
        spearman_corr = None
        if shared_ids:
            displacements = [abs(rank_orig[sid] - rank_mut[sid]) for sid in shared_ids]
            rank_disp = sum(displacements) / len(displacements)

            if len(shared_ids) >= 3:
                ranks_a = [rank_orig[sid] for sid in shared_ids]
                ranks_b = [rank_mut[sid] for sid in shared_ids]
                res = spearmanr(ranks_a, ranks_b)
                spearman_corr = float(res.correlation) if res.correlation is not None else None

        # Probability Total Variation Distance (TVD) on shared items
        prob_orig = {it.id: it.prob for it in orig_items if it.prob is not None}
        prob_mut = {it.id: it.prob for it in mut_items if it.prob is not None}
        prob_shared_ids = set(prob_orig.keys()) & set(prob_mut.keys())

        prob_tvd = None
        if prob_shared_ids:
            prob_tvd = 0.5 * sum(abs(prob_orig[sid] - prob_mut[sid]) for sid in prob_shared_ids)

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
            top1_prob_delta=top1_prob_delta,
            jaccard_top3=round(jaccard_3, 4),
            jaccard_top5=round(jaccard_5, 4),
            jaccard_top10=round(jaccard_10, 4),
            rank_displacement_avg=round(rank_disp, 4),
            spearman_correlation=round(spearman_corr, 4) if spearman_corr is not None else None,
            prob_tvd_shared=round(prob_tvd, 6) if prob_tvd is not None else None,
            price_mean_delta=round(price_delta, 2) if price_delta is not None else None,
            category_shifted=category_shifted,
        )

    @staticmethod
    def _jaccard(list_a: List[str], list_b: List[str]) -> float:
        set_a, set_b = set(list_a), set(list_b)
        union = set_a | set_b
        if not union:
            return 1.0
        return len(set_a & set_b) / len(union)
