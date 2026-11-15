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
    parser.add_argument(
        "--mode",
        choices=["live", "mock", "sdk"],
        default="live",
        help="Inference mode: 'live' (showcase API), 'mock' (offline), or 'sdk' (official BehaviorGPT SDK)",
    )
    parser.add_argument("--count", type=int, default=40, help="Total synthetic paired histories to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Target data directory (defaults to 'data/sdk_results' in sdk mode, 'data' otherwise)",
    )
    args = parser.parse_args()

    print(f"================================================================")
    print(f" BehaviorGPT Counterfactual Sensitivity Lab: Experiment Runner")
    print(f" Mode: {args.mode.upper()} | Count: {args.count} | Seed: {args.seed}")
    print(f"================================================================")

    runner = ExperimentRunner(mode=args.mode, seed=args.seed, data_dir=args.data_dir)
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
    if summary.avg_rank_displacement_penalized is not None:
        print(f"  Avg Penalized Disp:       {summary.avg_rank_displacement_penalized:.2f}")
    print("----------------------------------------------------------------")

    # Run case investigator on runner's output directory
    results_file = runner.data_dir / "experiment_results.jsonl"
    investigator = CaseInvestigator(results_path=str(results_file))
    paired_cases = investigator.load_paired_cases()
    noteworthy = investigator.find_noteworthy_cases()
    if noteworthy.get("high_sensitivity"):
        case = noteworthy["high_sensitivity"]
        paired_case = paired_cases.get(case.case_id)
        print("\n=== INVESTIGATED CASE: HIGH SENSITIVITY ===")
        print(investigator.format_case_investigation(case, paired_case=paired_case))
        investigated_file = runner.data_dir / "investigated_case.md"
        with open(investigated_file, "w", encoding="utf-8") as f:
            f.write(investigator.format_case_investigation(case, paired_case=paired_case))
        print(f"\nSaved deep case analysis to {investigated_file}")


if __name__ == "__main__":
    main()
