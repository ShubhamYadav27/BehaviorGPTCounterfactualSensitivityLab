"""Deterministic synthetic behavioral dataset generator for BehaviorGPT sensitivity analysis.

Generates 40 reproducible behavioral histories across distinct archetypes
using verified catalog items from the BehaviorGPT retail catalog.
"""

import random
from typing import List
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
)

# Curated verified items from BehaviorGPT retail_catalog_500k
VERIFIED_ITEMS = {
    # Footwear & Athletics
    "NIKE_JORDAN4_BRED": {"id": "paB07R17XQD2", "brand": "Jordan", "cat": "Footwear", "price": 115.0},
    "NIKE_PANTS": {"id": "paB08NYK61PJ", "brand": "Nike", "cat": "Athletic Apparel", "price": 45.0},
    "NIKE_PRO_SHORTS": {"id": "paB0CBT2582L", "brand": "Nike", "cat": "Athletic Apparel", "price": 30.0},
    "NIKE_DUNK_LOW": {"id": "paB0D93VKPPK", "brand": "Nike", "cat": "Footwear", "price": 55.0},
    "NIKE_AIR_JORDAN_RETRO": {"id": "paB0DD21212N", "brand": "Nike", "cat": "Footwear", "price": 153.0},
    "UNDER_ARMOUR_CHARGED": {"id": "paB0CJSYB1SY", "brand": "Under Armour", "cat": "Footwear", "price": 37.0},
    "TIMBERLAND_BOOT": {"id": "paB0CPSK1GVB", "brand": "Timberland", "cat": "Footwear", "price": 35.0},

    # Consumer Electronics & Gaming
    "APPLE_IPAD_109": {"id": "paB0DK3YY682", "brand": "Apple", "cat": "Electronics", "price": 277.0},
    "SAMSUNG_GALAXY_TAB": {"id": "paB0DJ5796F6", "brand": "Samsung", "cat": "Electronics", "price": 450.0},
    "PS5_CONTROLLER_PEARL": {"id": "paB0DJ1VN1LF", "brand": "PlayStation", "cat": "Gaming", "price": 66.0},

    # Luxury Fashion & Accessories
    "COACH_TABBY_BAG": {"id": "paB0D8CSSQ7J", "brand": "Coach", "cat": "Luxury Bags", "price": 395.0},

    # Home & Kitchen
    "LOLOI_ACCENT_RUG": {"id": "paB0CRLY5MYP", "brand": "Loloi", "cat": "Home", "price": 81.0},
}


class SyntheticHistoryGenerator:
    """Generates deterministic synthetic user histories with guaranteed repeatability."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_dataset(self, total_histories: int = 40) -> List[BehavioralHistory]:
        """Generate a deterministic synthetic dataset of behavioral sequences."""
        rng = random.Random(self.seed)
        histories: List[BehavioralHistory] = []

        archetypes = [
            ("ATHLETIC_RUNNER", "Consumer focused on sports footwear and workout apparel"),
            ("TECH_ENTHUSIAST", "Consumer browsing tablets, gaming hardware, and gadgets"),
            ("LUXURY_SHOPPER", "High-ticket accessories and designer apparel explorer"),
            ("HOME_DECORATOR", "Consumer browsing accent rugs, home decor, and lifestyle goods"),
            ("CROSS_SHOPPER", "Multi-category shopper browsing footwear then consumer tech"),
        ]

        # Item pools by category
        cat_items = {
            "athletic": [
                VERIFIED_ITEMS["NIKE_PANTS"]["id"],
                VERIFIED_ITEMS["NIKE_PRO_SHORTS"]["id"],
                VERIFIED_ITEMS["NIKE_DUNK_LOW"]["id"],
                VERIFIED_ITEMS["UNDER_ARMOUR_CHARGED"]["id"],
                VERIFIED_ITEMS["NIKE_JORDAN4_BRED"]["id"],
                VERIFIED_ITEMS["NIKE_AIR_JORDAN_RETRO"]["id"],
            ],
            "tech": [
                VERIFIED_ITEMS["APPLE_IPAD_109"]["id"],
                VERIFIED_ITEMS["SAMSUNG_GALAXY_TAB"]["id"],
                VERIFIED_ITEMS["PS5_CONTROLLER_PEARL"]["id"],
            ],
            "luxury": [
                VERIFIED_ITEMS["COACH_TABBY_BAG"]["id"],
                VERIFIED_ITEMS["NIKE_AIR_JORDAN_RETRO"]["id"],
            ],
            "home": [
                VERIFIED_ITEMS["LOLOI_ACCENT_RUG"]["id"],
                VERIFIED_ITEMS["TIMBERLAND_BOOT"]["id"],
            ],
        }

        for i in range(1, total_histories + 1):
            history_id = f"hist_{i:03d}"
            arch_key, arch_desc = archetypes[(i - 1) % len(archetypes)]

            # Sequence length between 2 and 6 actions
            seq_len = 2 + (i % 5)  # produces lengths 2, 3, 4, 5, 6 deterministically

            actions: List[BehavioralAction] = []
            pool = cat_items["athletic"]
            if arch_key == "TECH_ENTHUSIAST":
                pool = cat_items["tech"]
            elif arch_key == "LUXURY_SHOPPER":
                pool = cat_items["luxury"]
            elif arch_key == "HOME_DECORATOR":
                pool = cat_items["home"]
            elif arch_key == "CROSS_SHOPPER":
                pool = cat_items["athletic"] + cat_items["tech"]

            for step in range(seq_len):
                # 80% viewItem, 20% addToCart
                act_type = ActionType.VIEW_ITEM if rng.random() > 0.3 else ActionType.ADD_TO_CART
                chosen_item = rng.choice(pool)

                action = BehavioralAction(
                    action_id=f"{history_id}_act_{step + 1}",
                    action_type=act_type,
                    product_id=chosen_item,
                    offset_ms=step * 1000,
                    metadata={"step_index": step, "archetype": arch_key},
                )
                actions.append(action)

            history = BehavioralHistory(
                history_id=history_id,
                archetype=arch_key,
                description=f"{arch_desc} (Length: {seq_len})",
                actions=actions,
                market="US",
                created_at="2026-10-15T12:00:00Z",
                is_synthetic=True,
            )
            histories.append(history)

        return histories
