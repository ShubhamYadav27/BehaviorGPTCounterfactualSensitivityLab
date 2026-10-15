"""Controlled single-action mutation engine for paired intervention experiments.

Guarantees:
1. Exactly one action is mutated.
2. All non-mutated actions, sequence lengths, offsets, and metadata are strictly preserved.
3. Every intervention records explicit rationale, mutation type, and position category.
"""

import copy
from typing import List, Tuple
from behavior_lab.dataset_generator import VERIFIED_ITEMS
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    MutationPosition,
    MutationSpec,
    MutationType,
    PairedExperimentCase,
)


class ControlledMutator:
    """Creates strictly paired counterfactual variants by altering exactly one action."""

    def __init__(self):
        # Alternative items for cross-category and within-category interventions
        self.cross_category_pool = {
            "athletic": VERIFIED_ITEMS["APPLE_IPAD_109"]["id"],      # Tech replacement
            "tech": VERIFIED_ITEMS["COACH_TABBY_BAG"]["id"],         # Luxury replacement
            "luxury": VERIFIED_ITEMS["NIKE_JORDAN4_BRED"]["id"],     # Footwear replacement
            "home": VERIFIED_ITEMS["PS5_CONTROLLER_PEARL"]["id"],     # Gaming replacement
            "general": VERIFIED_ITEMS["LOLOI_ACCENT_RUG"]["id"],
        }
        self.within_category_pool = {
            VERIFIED_ITEMS["NIKE_PANTS"]["id"]: VERIFIED_ITEMS["NIKE_PRO_SHORTS"]["id"],
            VERIFIED_ITEMS["NIKE_JORDAN4_BRED"]["id"]: VERIFIED_ITEMS["NIKE_DUNK_LOW"]["id"],
            VERIFIED_ITEMS["APPLE_IPAD_109"]["id"]: VERIFIED_ITEMS["SAMSUNG_GALAXY_TAB"]["id"],
            VERIFIED_ITEMS["COACH_TABBY_BAG"]["id"]: VERIFIED_ITEMS["NIKE_AIR_JORDAN_RETRO"]["id"],
        }

    def create_paired_cases(self, histories: List[BehavioralHistory]) -> List[PairedExperimentCase]:
        """Generate paired experiment cases across diverse mutation strategies."""
        paired_cases: List[PairedExperimentCase] = []

        for idx, orig_hist in enumerate(histories, start=1):
            case_id = f"case_{idx:03d}"
            seq_len = orig_hist.length

            # Determine mutation strategy based on index for balanced experiment distribution
            strategy_cycle = (idx - 1) % 4
            if strategy_cycle == 0:
                mut_type = MutationType.ITEM_REPLACEMENT_CROSS_CATEGORY
                pos_cat = MutationPosition.LATE
                pos_idx = seq_len - 1
            elif strategy_cycle == 1:
                mut_type = MutationType.ITEM_REPLACEMENT_WITHIN_CATEGORY
                pos_cat = MutationPosition.LATE
                pos_idx = seq_len - 1
            elif strategy_cycle == 2:
                mut_type = MutationType.ACTION_TYPE_SWAP
                pos_cat = MutationPosition.LATE
                pos_idx = seq_len - 1
            else:
                # Early position intervention to test position sensitivity
                mut_type = MutationType.POSITION_EARLY
                pos_cat = MutationPosition.EARLY
                pos_idx = 0

            mutated_hist, spec = self._apply_single_mutation(
                orig_hist=orig_hist,
                case_id=case_id,
                mutation_type=mut_type,
                pos_cat=pos_cat,
                pos_idx=pos_idx,
            )

            # Validate invariant: exactly one action mutated
            self._validate_mutation_invariant(orig_hist, mutated_hist, pos_idx)

            paired_cases.append(
                PairedExperimentCase(
                    case_id=case_id,
                    history_id=orig_hist.history_id,
                    archetype=orig_hist.archetype,
                    original_history=orig_hist,
                    mutated_history=mutated_hist,
                    mutation_spec=spec,
                )
            )

        return paired_cases

    def _apply_single_mutation(
        self,
        orig_hist: BehavioralHistory,
        case_id: str,
        mutation_type: MutationType,
        pos_cat: MutationPosition,
        pos_idx: int,
    ) -> Tuple[BehavioralHistory, MutationSpec]:
        """Produce mutated history with exactly one altered action."""
        mutated_actions = copy.deepcopy(orig_hist.actions)
        target_orig = mutated_actions[pos_idx]

        target_mutated = copy.deepcopy(target_orig)
        target_mutated.action_id = f"{target_orig.action_id}_mutated"

        rationale = ""

        if mutation_type == MutationType.ITEM_REPLACEMENT_CROSS_CATEGORY:
            # Swap target item for cross-category item
            arch_key = orig_hist.archetype.lower()
            replacement_id = self.cross_category_pool.get(
                "athletic" if "athletic" in arch_key else "tech" if "tech" in arch_key else "general"
            )
            if replacement_id == target_orig.product_id:
                replacement_id = VERIFIED_ITEMS["COACH_TABBY_BAG"]["id"]
            target_mutated.product_id = replacement_id
            rationale = f"Replaced item {target_orig.product_id} with cross-category item {replacement_id} at position {pos_idx}."

        elif mutation_type == MutationType.ITEM_REPLACEMENT_WITHIN_CATEGORY:
            # Swap target item for a within-category sibling
            replacement_id = self.within_category_pool.get(target_orig.product_id)
            if not replacement_id or replacement_id == target_orig.product_id:
                candidates = [
                    VERIFIED_ITEMS["NIKE_DUNK_LOW"]["id"],
                    VERIFIED_ITEMS["NIKE_JORDAN4_BRED"]["id"],
                    VERIFIED_ITEMS["NIKE_PANTS"]["id"],
                    VERIFIED_ITEMS["UNDER_ARMOUR_CHARGED"]["id"],
                    VERIFIED_ITEMS["APPLE_IPAD_109"]["id"],
                ]
                replacement_id = next(c for c in candidates if c != target_orig.product_id)
            target_mutated.product_id = replacement_id
            rationale = f"Replaced item {target_orig.product_id} with within-category item {replacement_id} at position {pos_idx}."

        elif mutation_type == MutationType.ACTION_TYPE_SWAP:
            # Change action type (viewItem <-> addToCart) while preserving item
            new_type = (
                ActionType.ADD_TO_CART
                if target_orig.action_type == ActionType.VIEW_ITEM
                else ActionType.VIEW_ITEM
            )
            target_mutated.action_type = new_type
            rationale = f"Changed action type from {target_orig.action_type.value} to {new_type.value} on item {target_orig.product_id}."

        else:  # POSITION_EARLY
            # Cross-category swap at position 0 (earliest event)
            replacement_id = VERIFIED_ITEMS["COACH_TABBY_BAG"]["id"]
            if replacement_id == target_orig.product_id:
                replacement_id = VERIFIED_ITEMS["PS5_CONTROLLER_PEARL"]["id"]
            target_mutated.product_id = replacement_id
            rationale = f"Early-position mutation at index 0: replaced {target_orig.product_id} with luxury item {replacement_id}."

        mutated_actions[pos_idx] = target_mutated

        mutated_history = BehavioralHistory(
            history_id=f"{orig_hist.history_id}_mutated",
            archetype=orig_hist.archetype,
            description=f"Mutated variant of {orig_hist.history_id}: {rationale}",
            actions=mutated_actions,
            market=orig_hist.market,
            created_at=orig_hist.created_at,
            is_synthetic=True,
        )

        spec = MutationSpec(
            case_id=case_id,
            mutation_type=mutation_type,
            position_category=pos_cat,
            position_index=pos_idx,
            original_action=target_orig,
            mutated_action=target_mutated,
            rationale=rationale,
        )

        return mutated_history, spec

    @staticmethod
    def _validate_mutation_invariant(
        orig_hist: BehavioralHistory, mutated_hist: BehavioralHistory, expected_pos: int
    ):
        """Strictly verify that exactly one action was changed."""
        assert orig_hist.length == mutated_hist.length, "History lengths differ!"

        diff_count = 0
        diff_indices = []

        for idx, (act_orig, act_mut) in enumerate(zip(orig_hist.actions, mutated_hist.actions)):
            is_different = (
                act_orig.product_id != act_mut.product_id
                or act_orig.action_type != act_mut.action_type
                or act_orig.query != act_mut.query
            )
            if is_different:
                diff_count += 1
                diff_indices.append(idx)
            else:
                # Ensure all other fields are preserved
                assert act_orig.offset_ms == act_mut.offset_ms, f"Offset changed at index {idx}!"

        if diff_count != 1:
            raise AssertionError(
                f"Mutation invariant violated: expected exactly 1 change at index {expected_pos}, but found {diff_count} at indices {diff_indices}."
            )
        if diff_indices[0] != expected_pos:
            raise AssertionError(
                f"Mutation occurred at index {diff_indices[0]}, expected {expected_pos}."
            )
