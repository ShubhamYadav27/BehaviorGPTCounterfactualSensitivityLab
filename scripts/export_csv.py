"""Export JSONL experiment results into CSV format for tabular analysis."""

import csv
import json
from pathlib import Path

RESULTS_JSONL = Path("data/experiment_results.jsonl")
RESULTS_CSV = Path("data/experiment_results.csv")


def export_csv():
    if not RESULTS_JSONL.exists():
        print("data/experiment_results.jsonl does not exist")
        return

    rows = []
    with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["metrics"]
            orig_top1 = r["original_output"]["items"][0] if r["original_output"]["items"] else {}
            mut_top1 = r["mutated_output"]["items"][0] if r["mutated_output"]["items"] else {}

            rows.append({
                "case_id": r["case_id"],
                "archetype": r["archetype"],
                "mutation_type": r["mutation_type"],
                "position_category": r["position_category"],
                "position_index": r["position_index"],
                "history_length": r["history_length"],
                "top1_changed": m["top1_changed"],
                "original_top1_id": orig_top1.get("id"),
                "original_top1_name": orig_top1.get("name"),
                "original_top1_prob": orig_top1.get("prob"),
                "mutated_top1_id": mut_top1.get("id"),
                "mutated_top1_name": mut_top1.get("name"),
                "mutated_top1_prob": mut_top1.get("prob"),
                "jaccard_top3": m["jaccard_top3"],
                "jaccard_top5": m["jaccard_top5"],
                "jaccard_top10": m["jaccard_top10"],
                "rank_displacement_avg": m["rank_displacement_avg"],
                "spearman_correlation": m.get("spearman_correlation"),
                "prob_tvd_shared": m.get("prob_tvd_shared"),
                "price_mean_delta": m.get("price_mean_delta"),
                "category_shifted": m.get("category_shifted"),
            })

    if not rows:
        return

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} records to {RESULTS_CSV}")


if __name__ == "__main__":
    export_csv()
