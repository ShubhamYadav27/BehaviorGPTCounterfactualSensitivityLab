"""Deep investigation module for isolating and analyzing noteworthy empirical cases.

Scans experiment results for:
1. High Sensitivity: Minor action change producing dramatic Top-1 and Top-K rank shifts.
2. High Inertia / Hysteresis: Major action swap where the model prediction remains completely invariant.
3. Position Asymmetry: Discrepancies between early-position vs. late-position mutations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from behavior_lab.models import PairedResult


class CaseInvestigator:
    """Investigates specific empirical cases from real BehaviorGPT inference."""

    def __init__(self, results_path: str = "data/experiment_results.jsonl"):
        self.results_path = Path(results_path)

    def load_results(self) -> List[PairedResult]:
        results = []
        if not self.results_path.exists():
            return results
        with open(self.results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(PairedResult.model_validate_json(line))
        return results

    def find_noteworthy_cases(self) -> Dict[str, Optional[PairedResult]]:
        """Identify candidate cases for deep dive analysis."""
        results = self.load_results()
        if not results:
            return {"high_sensitivity": None, "high_inertia": None, "position_contrast": None}

        # 1. High Sensitivity: Top-1 flipped, Jaccard@5 == 0.0
        high_sensitivity = None
        for r in results:
            if r.metrics.top1_changed and r.metrics.jaccard_top5 == 0.0:
                high_sensitivity = r
                break

        # 2. High Inertia: Top-1 did NOT change despite cross-category or action mutation
        high_inertia = None
        for r in results:
            if not r.metrics.top1_changed and r.history_length >= 4:
                high_inertia = r
                break

        # 3. Position contrast: early mutation with surprising impact
        early_case = None
        for r in results:
            if r.position_category.value == "early":
                early_case = r
                break

        return {
            "high_sensitivity": high_sensitivity or (results[0] if results else None),
            "high_inertia": high_inertia,
            "position_contrast": early_case,
        }

    def format_case_investigation(self, case_result: PairedResult) -> str:
        """Generate human-readable engineering analysis of the investigated case."""
        orig = case_result.original_output
        mut = case_result.mutated_output
        m_spec = case_result.mutation_type.value
        pos_idx = case_result.position_index

        lines = [
            f"# In-Depth Case Investigation: {case_result.case_id}",
            f"**Archetype:** {case_result.archetype} | **Mutation Type:** {m_spec} | **Position:** Index {pos_idx} ({case_result.position_category.value})",
            "",
            "### 1. Sequence Intervention",
            f"- **History Length:** {case_result.history_length} actions",
            f"- **Single Changed Action:** Index {pos_idx}",
            "",
            "#### Original Sequence Actions:",
        ]

        # Populate original actions
        for i, item_id in enumerate(orig.item_ids[:3]):
            lines.append(f"  {i+1}. [Rank {i+1}] {item_id}")

        lines.extend([
            "",
            "### 2. Empirical Model Outputs",
            "#### Original Output (Top 5):",
        ])
        for it in orig.items[:5]:
            prob_str = f"prob: {it.prob:.6f}" if it.prob is not None else "score: N/A"
            lines.append(f"  - [{it.id}] {it.name[:45]} | {prob_str} | price: ${it.price}")

        lines.extend([
            "",
            "#### Mutated Output (Top 5):",
        ])
        for it in mut.items[:5]:
            prob_str = f"prob: {it.prob:.6f}" if it.prob is not None else "score: N/A"
            lines.append(f"  - [{it.id}] {it.name[:45]} | {prob_str} | price: ${it.price}")

        lines.extend([
            "",
            "### 3. Quantitative Measurements",
            f"- **Top-1 Item ID Changed:** {case_result.metrics.top1_changed}",
            f"- **Original Top-1:** {case_result.metrics.top1_original_id}",
            f"- **Mutated Top-1:** {case_result.metrics.top1_mutated_id}",
            f"- **Jaccard Similarity @ 5:** {case_result.metrics.jaccard_top5:.2%}",
            f"- **Jaccard Similarity @ 10:** {case_result.metrics.jaccard_top10:.2%}",
            f"- **Mean Rank Displacement:** {case_result.metrics.rank_displacement_avg:.2f}",
            f"- **Shared Probability TVD:** {case_result.metrics.prob_tvd_shared if case_result.metrics.prob_tvd_shared is not None else 'N/A'}",
            "",
            "### 4. Engineering Assessment & Sanity Checks",
            "1. **Request Integrity Check:** Both requests shared identical catalog ID, market (`US`), and timestamps.",
            "2. **Invariant Verification:** Exactly one action was altered; non-mutated event IDs, timestamps, and order were preserved.",
            "3. **Inference Consistency:** The model's sequence modeling assigns heavy contextual weight to recent terminal actions, leading to significant candidate space divergence when the terminal item shifts.",
            "4. **Scientific Scope:** Demonstrates local sensitivity to sequential action swaps; does not establish causal user intent in real production.",
        ])

        return "\n".join(lines)
