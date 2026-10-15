"""Execution script to run the full BehaviorGPT Counterfactual Sensitivity Lab experiment."""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from behavior_lab.runner import ExperimentRunner
from behavior_lab.case_investigator import CaseInvestigator


def main():
    parser = argparse.ArgumentParser(description="Run BehaviorGPT Counterfactual Sensitivity Experiment")
    parser.add_argument("--mode", choices=["live", "mock"], default="live", help="Inference mode (live or mock)")
    parser.add_argument("--count", type=int, default=40, help="Total synthetic paired histories to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    args = parser.parse_args()

    print(f"================================================================")
    print(f" BehaviorGPT Counterfactual Sensitivity Lab: Experiment Runner")
    print(f" Mode: {args.mode.upper()} | Count: {args.count} | Seed: {args.seed}")
    print(f"================================================================")

    runner = ExperimentRunner(mode=args.mode, seed=args.seed)
    summary = runner.run_full_suite(total_histories=args.count)

    print("\n----------------------------------------------------------------")
    print(" EXPERIMENT SUMMARY RESULTS:")
    print(f"  Total Pairs:              {summary.total_cases}")
    print(f"  Successful Inferences:    {summary.successful_cases}")
    print(f"  Failed Inferences:        {summary.failed_cases}")
    print(f"  Top-1 Flip Rate:          {summary.top1_change_rate:.1%}")
    print(f"  Avg Jaccard Similarity@5: {summary.avg_jaccard_top5:.3f}")
    print(f"  Avg Jaccard Similarity@10:{summary.avg_jaccard_top10:.3f}")
    print(f"  Avg Rank Displacement:    {summary.avg_rank_displacement:.2f}")
    print("----------------------------------------------------------------")

    # Run case investigator
    investigator = CaseInvestigator(results_path="data/experiment_results.jsonl")
    noteworthy = investigator.find_noteworthy_cases()
    if noteworthy.get("high_sensitivity"):
        print("\n=== INVESTIGATED CASE: HIGH SENSITIVITY ===")
        print(investigator.format_case_investigation(noteworthy["high_sensitivity"]))
        with open("data/investigated_case.md", "w", encoding="utf-8") as f:
            f.write(investigator.format_case_investigation(noteworthy["high_sensitivity"]))
        print("\nSaved deep case analysis to data/investigated_case.md")


if __name__ == "__main__":
    main()
