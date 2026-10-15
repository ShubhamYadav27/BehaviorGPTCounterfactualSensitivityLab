"""Experiment orchestration engine for BehaviorGPT Counterfactual Sensitivity Lab.

Runs paired experiments, computes metrics, generates summaries,
and serializes artifacts into JSONL / JSON formats.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

from behavior_lab.client import BehaviorGPTClient
from behavior_lab.dataset_generator import SyntheticHistoryGenerator
from behavior_lab.metrics import SensitivityMetricsCalculator
from behavior_lab.models import (
    BehavioralHistory,
    ExperimentSummary,
    PairedExperimentCase,
    PairedResult,
)
from behavior_lab.mutator import ControlledMutator


class ExperimentRunner:
    """Orchestrates paired counterfactual sensitivity experiments."""

    def __init__(
        self,
        mode: str = "live",
        data_dir: str = "data",
        limit: int = 10,
        seed: int = 42,
    ):
        self.mode = mode
        self.data_dir = Path(data_dir)
        self.limit = limit
        self.seed = seed
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run_full_suite(self, total_histories: int = 40) -> ExperimentSummary:
        """Execute full experimental protocol from synthetic generation to result persistence."""
        print(f"[{datetime.now(tz=UTC).isoformat()}] Generating {total_histories} synthetic histories (seed={self.seed})...")
        generator = SyntheticHistoryGenerator(seed=self.seed)
        histories = generator.generate_dataset(total_histories=total_histories)

        print(f"[{datetime.now(tz=UTC).isoformat()}] Constructing paired single-action mutations...")
        mutator = ControlledMutator()
        paired_cases = mutator.create_paired_cases(histories)

        # Save input datasets
        self._save_jsonl(self.data_dir / "synthetic_histories.jsonl", [h.model_dump() for h in histories])
        self._save_jsonl(self.data_dir / "paired_experiment_cases.jsonl", [c.model_dump() for c in paired_cases])

        print(f"[{datetime.now(tz=UTC).isoformat()}] Running inference in '{self.mode}' mode across {len(paired_cases)} pairs...")
        results: List[PairedResult] = []
        successful = 0
        failed = 0

        with BehaviorGPTClient(mode=self.mode) as client:
            for i, case in enumerate(paired_cases, start=1):
                print(f"  Executing pair {i}/{len(paired_cases)}: {case.case_id} ({case.archetype} | {case.mutation_spec.mutation_type.value})...")
                orig_out = client.predict(
                    history=case.original_history,
                    limit=self.limit,
                    variant_label="original",
                    case_id=case.case_id,
                )
                mut_out = client.predict(
                    history=case.mutated_history,
                    limit=self.limit,
                    variant_label="mutated",
                    case_id=case.case_id,
                )

                if orig_out.status_code == 200 and mut_out.status_code == 200:
                    successful += 1
                else:
                    failed += 1

                metrics = SensitivityMetricsCalculator.compute(orig_out, mut_out)

                result = PairedResult(
                    case_id=case.case_id,
                    archetype=case.archetype,
                    mutation_type=case.mutation_spec.mutation_type,
                    position_category=case.mutation_spec.position_category,
                    position_index=case.mutation_spec.position_index,
                    history_length=case.original_history.length,
                    original_output=orig_out,
                    mutated_output=mut_out,
                    metrics=metrics,
                )
                results.append(result)

        # Save paired results
        self._save_jsonl(self.data_dir / "experiment_results.jsonl", [r.model_dump() for r in results])

        # Compute aggregate summary
        summary = self._compute_summary(results, successful, failed)
        with open(self.data_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))

        print(f"[{datetime.now(tz=UTC).isoformat()}] Experiment run completed! Successful: {successful}, Failed: {failed}")
        print(f"  Top-1 Flip Rate: {summary.top1_change_rate:.1%}")
        print(f"  Mean Jaccard@5: {summary.avg_jaccard_top5:.3f}")
        print(f"  Mean Rank Displacement: {summary.avg_rank_displacement:.2f}")

        return summary

    def _compute_summary(
        self, results: List[PairedResult], successful: int, failed: int
    ) -> ExperimentSummary:
        total = len(results)
        if total == 0:
            raise ValueError("No results to summarize")

        top1_changed_count = sum(1 for r in results if r.metrics.top1_changed)
        top1_rate = top1_changed_count / total

        avg_j5 = sum(r.metrics.jaccard_top5 for r in results) / total
        avg_j10 = sum(r.metrics.jaccard_top10 for r in results) / total
        avg_disp = sum(r.metrics.rank_displacement_avg for r in results) / total

        tvd_vals = [r.metrics.prob_tvd_shared for r in results if r.metrics.prob_tvd_shared is not None]
        avg_tvd = sum(tvd_vals) / len(tvd_vals) if tvd_vals else 0.0

        # Breakdowns by mutation type
        by_mut: Dict[str, Dict[str, float]] = {}
        for r in results:
            m_key = r.mutation_type.value
            if m_key not in by_mut:
                by_mut[m_key] = {"count": 0, "top1_flips": 0, "sum_j5": 0.0, "sum_disp": 0.0}
            by_mut[m_key]["count"] += 1
            if r.metrics.top1_changed:
                by_mut[m_key]["top1_flips"] += 1
            by_mut[m_key]["sum_j5"] += r.metrics.jaccard_top5
            by_mut[m_key]["sum_disp"] += r.metrics.rank_displacement_avg

        for m_key, data in by_mut.items():
            cnt = data["count"]
            data["top1_flip_rate"] = round(data["top1_flips"] / cnt, 3)
            data["avg_jaccard_5"] = round(data["sum_j5"] / cnt, 3)
            data["avg_displacement"] = round(data["sum_disp"] / cnt, 3)

        # Breakdowns by position
        by_pos: Dict[str, Dict[str, float]] = {}
        for r in results:
            p_key = r.position_category.value
            if p_key not in by_pos:
                by_pos[p_key] = {"count": 0, "top1_flips": 0, "sum_j5": 0.0}
            by_pos[p_key]["count"] += 1
            if r.metrics.top1_changed:
                by_pos[p_key]["top1_flips"] += 1
            by_pos[p_key]["sum_j5"] += r.metrics.jaccard_top5

        for p_key, data in by_pos.items():
            cnt = data["count"]
            data["top1_flip_rate"] = round(data["top1_flips"] / cnt, 3)
            data["avg_jaccard_5"] = round(data["sum_j5"] / cnt, 3)

        # Identify an interesting candidate case:
        # e.g., cross-category swap that completely flipped top1, or subtle swap that preserved top1
        interesting_id = None
        for r in results:
            if r.metrics.top1_changed and r.metrics.jaccard_top5 < 0.2:
                interesting_id = r.case_id
                break
        if not interesting_id and results:
            interesting_id = results[0].case_id

        return ExperimentSummary(
            experiment_id=f"exp_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(tz=UTC).isoformat(),
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            model_name="BehaviorGPT-v4-12.5B",
            catalog_id="retail_catalog_500k",
            execution_mode=self.mode,
            top1_change_rate=round(top1_rate, 4),
            avg_jaccard_top5=round(avg_j5, 4),
            avg_jaccard_top10=round(avg_j10, 4),
            avg_rank_displacement=round(avg_disp, 4),
            avg_prob_tvd=round(avg_tvd, 6),
            by_mutation_type=by_mut,
            by_position=by_pos,
            investigated_case_id=interesting_id,
        )

    @staticmethod
    def _save_jsonl(path: Path, items: List[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
