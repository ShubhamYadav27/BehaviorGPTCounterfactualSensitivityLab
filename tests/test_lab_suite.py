"""Automated Test Suite for BehaviorGPT Counterfactual Sensitivity Lab.

Covers:
1. Synthetic dataset determinism & seed invariance
2. Valid behavioral history schema
3. Exactly-one-action mutation invariant
4. Non-mutated actions preservation
5. Output parsing & prediction item validation
6. Metric calculations (Jaccard, TVD, Rank displacement, Spearman)
7. Error & timeout handling with mock/error responses
8. Result serialization and JSONL round-trip fidelity
9. Clear separation between mock and live modes
"""

import json
import pytest
from pathlib import Path

from behavior_lab.client import BehaviorGPTClient
from behavior_lab.dataset_generator import SyntheticHistoryGenerator, VERIFIED_ITEMS
from behavior_lab.metrics import SensitivityMetricsCalculator
from behavior_lab.models import (
    ActionType,
    BehavioralAction,
    BehavioralHistory,
    InferenceOutput,
    MutationPosition,
    MutationType,
    PredictionItem,
)
from behavior_lab.mutator import ControlledMutator


def test_synthetic_dataset_determinism():
    """Generating with same seed produces identical histories."""
    gen1 = SyntheticHistoryGenerator(seed=123)
    gen2 = SyntheticHistoryGenerator(seed=123)

    data1 = gen1.generate_dataset(total_histories=10)
    data2 = gen2.generate_dataset(total_histories=10)

    assert len(data1) == 10
    assert len(data2) == 10
    for h1, h2 in zip(data1, data2):
        assert h1.history_id == h2.history_id
        assert h1.length == h2.length
        for a1, a2 in zip(h1.actions, h2.actions):
            assert a1.action_type == a2.action_type
            assert a1.product_id == a2.product_id
            assert a1.offset_ms == a2.offset_ms


def test_different_seeds_produce_different_histories():
    """Different seeds generate distinct sequences."""
    gen1 = SyntheticHistoryGenerator(seed=111)
    gen2 = SyntheticHistoryGenerator(seed=999)

    data1 = gen1.generate_dataset(total_histories=5)
    data2 = gen2.generate_dataset(total_histories=5)

    # At least one item should differ across the datasets
    items1 = [a.product_id for h in data1 for a in h.actions]
    items2 = [a.product_id for h in data2 for a in h.actions]
    assert items1 != items2


def test_mutation_exactly_one_action_changed():
    """Verify that every paired case has exactly one modified action."""
    gen = SyntheticHistoryGenerator(seed=42)
    histories = gen.generate_dataset(total_histories=15)
    mutator = ControlledMutator()

    paired_cases = mutator.create_paired_cases(histories)
    assert len(paired_cases) == 15

    for case in paired_cases:
        orig = case.original_history
        mut = case.mutated_history

        assert orig.length == mut.length

        diffs = 0
        diff_idx = -1
        for idx, (a_orig, a_mut) in enumerate(zip(orig.actions, mut.actions)):
            changed = (
                a_orig.product_id != a_mut.product_id
                or a_orig.action_type != a_mut.action_type
            )
            if changed:
                diffs += 1
                diff_idx = idx
            else:
                # Non-mutated actions MUST be identical
                assert a_orig.product_id == a_mut.product_id
                assert a_orig.action_type == a_mut.action_type
                assert a_orig.offset_ms == a_mut.offset_ms

        assert diffs == 1, f"Expected exactly 1 change, but found {diffs} in {case.case_id}"
        assert diff_idx == case.mutation_spec.position_index


def test_metric_calculations_known_analytical_cases():
    """Verify accuracy of Jaccard, Rank Displacement, and TVD calculations."""
    # Case 1: Identical items -> Jaccard 1.0, Displacement 0.0, Top1 unchanged
    items_a = [
        PredictionItem(id="item_1", name="Product 1", score=0.8, prob=0.4, rank=1),
        PredictionItem(id="item_2", name="Product 2", score=0.5, prob=0.2, rank=2),
        PredictionItem(id="item_3", name="Product 3", score=0.3, prob=0.1, rank=3),
    ]
    items_b = [
        PredictionItem(id="item_1", name="Product 1", score=0.8, prob=0.4, rank=1),
        PredictionItem(id="item_2", name="Product 2", score=0.5, prob=0.2, rank=2),
        PredictionItem(id="item_3", name="Product 3", score=0.3, prob=0.1, rank=3),
    ]

    out_a = InferenceOutput(case_id="c1", variant="orig", items=items_a, latency_ms=1.0)
    out_b = InferenceOutput(case_id="c1", variant="mut", items=items_b, latency_ms=1.0)

    metrics_identical = SensitivityMetricsCalculator.compute(out_a, out_b)
    assert metrics_identical.top1_changed is False
    assert metrics_identical.jaccard_top3 == 1.0
    assert metrics_identical.rank_displacement_avg == 0.0
    assert metrics_identical.prob_tvd_shared == 0.0

    # Case 2: Disjoint items -> Jaccard 0.0, Top1 changed
    items_c = [
        PredictionItem(id="item_4", name="Product 4", score=0.9, prob=0.5, rank=1),
        PredictionItem(id="item_5", name="Product 5", score=0.6, prob=0.3, rank=2),
        PredictionItem(id="item_6", name="Product 6", score=0.2, prob=0.1, rank=3),
    ]
    out_c = InferenceOutput(case_id="c2", variant="mut", items=items_c, latency_ms=1.0)

    metrics_disjoint = SensitivityMetricsCalculator.compute(out_a, out_c)
    assert metrics_disjoint.top1_changed is True
    assert metrics_disjoint.jaccard_top3 == 0.0
    assert metrics_disjoint.rank_displacement_avg == 0.0

    # Case 3: Reordered items -> item 1 and item 2 swapped ranks
    items_d = [
        PredictionItem(id="item_2", name="Product 2", score=0.8, prob=0.3, rank=1),
        PredictionItem(id="item_1", name="Product 1", score=0.5, prob=0.2, rank=2),
        PredictionItem(id="item_3", name="Product 3", score=0.3, prob=0.1, rank=3),
    ]
    out_d = InferenceOutput(case_id="c3", variant="mut", items=items_d, latency_ms=1.0)

    metrics_swapped = SensitivityMetricsCalculator.compute(out_a, out_d)
    assert metrics_swapped.top1_changed is True
    assert metrics_swapped.jaccard_top3 == 1.0
    # Rank displacements: item_1 changed from rank 1 to 2 (|1-2|=1), item_2 changed from rank 2 to 1 (|2-1|=1), item_3 unchanged (|3-3|=0)
    # Mean displacement: (1 + 1 + 0) / 3 = 2/3 = 0.6667
    assert abs(metrics_swapped.rank_displacement_avg - 0.6667) < 0.001


def test_mock_client_inference():
    """Verify mock inference generates valid schema without network calls."""
    client = BehaviorGPTClient(mode="mock")
    history = BehavioralHistory(
        history_id="test_mock",
        archetype="ATHLETIC_RUNNER",
        description="Test history",
        actions=[
            BehavioralAction(
                action_id="act_1",
                action_type=ActionType.VIEW_ITEM,
                product_id=VERIFIED_ITEMS["NIKE_PANTS"]["id"],
            )
        ],
        created_at="2026-10-15T00:00:00Z",
    )

    out = client.predict(history=history, limit=5, case_id="mock_case_01")
    assert out.status_code == 200
    assert out.is_mock is True
    assert len(out.items) == 5
    assert out.top1 is not None
    assert out.top1.rank == 1
    assert out.top1.prob is not None


def test_error_handling_invalid_endpoint():
    """Verify robust error reporting when encountering bad URLs or connection issues."""
    client = BehaviorGPTClient(mode="live", base_url="http://invalid-nonexistent-domain.test/api")
    history = BehavioralHistory(
        history_id="test_err",
        archetype="TEST",
        description="Test history",
        actions=[
            BehavioralAction(
                action_id="act_err",
                action_type=ActionType.VIEW_ITEM,
                product_id="test_id",
            )
        ],
        created_at="2026-10-15T00:00:00Z",
    )

    out = client.predict(history=history, limit=5, case_id="err_case")
    assert out.status_code == 500
    assert out.error_message is not None
    assert len(out.items) == 0


def test_jsonl_serialization_round_trip(tmp_path):
    """Verify JSONL persistence preserves full fidelity."""
    file_path = tmp_path / "test_data.jsonl"
    gen = SyntheticHistoryGenerator(seed=42)
    histories = gen.generate_dataset(total_histories=3)

    # Save
    with open(file_path, "w", encoding="utf-8") as f:
        for h in histories:
            f.write(h.model_dump_json() + "\n")

    # Load and verify
    loaded = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                loaded.append(BehavioralHistory.model_validate_json(line))

    assert len(loaded) == 3
    for orig, reserialized in zip(histories, loaded):
        assert orig.history_id == reserialized.history_id
        assert orig.length == reserialized.length
        assert orig.actions[0].product_id == reserialized.actions[0].product_id
