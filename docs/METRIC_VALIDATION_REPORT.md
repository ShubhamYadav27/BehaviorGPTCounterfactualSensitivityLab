# Metric Validation & Algorithmic Audit Report

**Date:** 2026-11-15  
**Auditor:** Principal Backend Engineer, Senior ML Engineer, AI Research Scientist  
**Scope:** `src/behavior_lab/metrics.py`, `src/behavior_lab/models.py`, `src/behavior_lab/sdk_adapter.py`  
**Status:** COMPLETE (Stage A Audit)  

---

## 1. Executive Summary: The `case_001` Metric Anomaly

In paired counterfactual case `case_001` (from historical data `data/experiment_results.jsonl`), the original and counterfactual inference requests returned **completely disjoint Top-10 recommendation sets**:

```json
{
  "case_id": "case_001",
  "archetype": "ATHLETIC_RUNNER",
  "mutation_type": "cross_category_swap",
  "metrics": {
    "top1_changed": true,
    "jaccard_top3": 0.0,
    "jaccard_top5": 0.0,
    "jaccard_top10": 0.0,
    "rank_displacement_avg": 0.0,
    "spearman_correlation": null,
    "prob_tvd_shared": null
  }
}
```

### The Contradiction:
* **`jaccard_top10 = 0.0`**: Not a single item in the original Top-10 appeared in the mutated Top-10 ($\text{intersection} = \emptyset$).
* **`rank_displacement_avg = 0.0`**: A rank displacement of $0.0$ ordinarily denotes **zero perturbation / perfect ranking stability** (i.e. every item maintained its exact rank).
* **The Root Cause in `metrics.py` (Lines 58–64)**:
  ```python
  shared_ids = set(rank_orig.keys()) & set(rank_mut.keys())

  rank_disp = 0.0
  spearman_corr = None
  if shared_ids:
      displacements = [abs(rank_orig[sid] - rank_mut[sid]) for sid in shared_ids]
      rank_disp = sum(displacements) / len(displacements)
  ```
  When `shared_ids` is empty (`set()`), the conditional `if shared_ids:` evaluates to `False`. The variable `rank_disp` was pre-initialized to `0.0`, so it returned `0.0`.
  Furthermore, `SensitivityMetrics.rank_displacement_avg` was typed as `float` (instead of `Optional[float]`), preventing it from returning `None`.

---

## 2. Comprehensive Metric-by-Metric Mathematical Audit

### 2.1 Jaccard Similarity ($\text{Top-3}, \text{Top-5}, \text{Top-10}$)

* **Current Implementation (`metrics.py`, Lines 110–116)**:
  ```python
  @staticmethod
  def _jaccard(list_a: List[str], list_b: List[str]) -> float:
      set_a, set_b = set(list_a), set(list_b)
      union = set_a | set_b
      if not union:
          return 1.0
      return len(set_a & set_b) / len(union)
  ```
* **Edge Case Analysis**:
  1. **Disjoint Sets**: If $A \cap B = \emptyset$, $\text{len}(A \cap B) = 0$, returns `0.0`. Correct.
  2. **Identical Sets**: If $A = B$, $\text{len}(A \cap B) = \text{len}(A \cup B)$, returns `1.0`. Correct.
  3. **Empty Sets (Both Empty)**: If $A = \emptyset$ and $B = \emptyset$ (e.g. server failure or empty catalog), returns `1.0`.
     * *Scientific Concern*: If an upstream inference fails on both variants and returns 0 items, Jaccard returns $1.0$, giving a false illusion of perfect stability.
     * *Remedy*: Return `None` or guard at the caller level when `not orig_items or not mut_items`. Lines 31–38 currently return `0.0` when either list is empty, which partially mitigates this.
  4. **Unequal Truncation**: If $A$ has 10 items and $B$ only returned 4 items, the denominator is $\text{len}(A \cup B) \le 14$. The denominator correctly scales with the actual union size.

### 2.2 Mean Rank Displacement

* **Current Formulation**:
  $$\text{MRD}_{\text{shared}} = \frac{1}{|I_{\text{shared}}|} \sum_{i \in I_{\text{shared}}} |r_{\text{orig}}(i) - r_{\text{mut}}(i)|$$
  where $I_{\text{shared}} = S_{\text{orig}}^K \cap S_{\text{mut}}^K$.
* **Critical Flaws**:
  1. **Vacuous Truth / Division by Zero**: When $I_{\text{shared}} = \emptyset$, $|I_{\text{shared}}| = 0$. Pre-initializing `rank_disp = 0.0` treats complete absence of shared items as zero displacement.
  2. **Severe Survivor Bias**: By measuring displacement *only* over items that survive in both Top-K lists, the metric actually penalizes mild rank adjustments more heavily than complete list eviction!
     * *Example A*: Item 1 drops to Rank 2 ($|1 - 2| = 1.0$). $\text{MRD} = 1.0$.
     * *Example B*: Item 1 drops to Rank 500 (completely leaves Top-10). It is omitted from $I_{\text{shared}}$. If no items survive, $\text{MRD} = 0.0$!
* **Proposed Formulations for Stage B**:
  1. **`rank_displacement_shared: Optional[float]`**: Return `None` when $|I_{\text{shared}}| = 0$, consistent with `spearman_correlation` and `prob_tvd_shared`.
  2. **`rank_displacement_penalized: float`** (Full Top-K Rank Drift): Assign a penalty rank of $K + 1$ (e.g., $11$ for Top-10) to items that drop out of the list:
     $$\text{MRD}_{\text{penalized}} = \frac{1}{2K} \left( \sum_{i \in S_{\text{orig}}^K} |r_{\text{orig}}(i) - r_{\text{mut}}'(i)| + \sum_{j \in S_{\text{mut}}^K} |r_{\text{orig}}'(j) - r_{\text{mut}}(j)| \right)$$
     where $r'(x) = K + 1$ if $x \notin \text{Top-K}$.
     For completely disjoint Top-10 lists, this produces maximal displacement ($\approx 5.5 - 6.0$), truthfully reflecting total list turnover.

### 2.3 Spearman Rank Correlation

* **Current Implementation (`metrics.py`, Lines 66–70)**:
  ```python
  if len(shared_ids) >= 3:
      ranks_a = [rank_orig[sid] for sid in shared_ids]
      ranks_b = [rank_mut[sid] for sid in shared_ids]
      res = spearmanr(ranks_a, ranks_b)
      spearman_corr = float(res.correlation) if res.correlation is not None else None
  ```
* **Evaluation**:
  * Correctly requires $\ge 3$ points (Spearman correlation with $N < 3$ is statistically degenerate).
  * Correctly returns `None` when shared items $< 3$.
  * Handles NaN returns from `scipy.stats.spearmanr`.

### 2.4 Total Variation Distance (TVD) & Uncalibrated Scores

* **Current Implementation (`metrics.py`, Lines 72–79)**:
  ```python
  prob_orig = {it.id: it.prob for it in orig_items if it.prob is not None}
  prob_mut = {it.id: it.prob for it in mut_items if it.prob is not None}
  prob_shared_ids = set(prob_orig.keys()) & set(prob_mut.keys())

  prob_tvd = None
  if prob_shared_ids:
      prob_tvd = 0.5 * sum(abs(prob_orig[sid] - prob_mut[sid]) for sid in prob_shared_ids)
  ```
* **Critical Finding: Probability vs. Model Score Confusion**:
  * In showcase mode, the API returned `"prob": 0.006...` (values in $[0, 1]$ representing probability mass).
  * In official SDK mode, `UnboxAIResponse.items[i].score` returns **uncalibrated dot-product / logit values** (e.g., `1.2469`, `1.5093`).
  * In `sdk_adapter.py` (Line 103), the adapter assigned:
    ```python
    score=item.score,
    prob=item.score,  # FLAW: Uncalibrated score assigned to prob field
    ```
  * When `prob` is $> 1.0$, computing `0.5 * sum(|score_a - score_b|)` is **NOT a Total Variation Distance** ($\text{TVD} \in [0, 1]$ is strictly defined for probability distributions $\sum p_i = 1$).
  * Computing TVD on uncalibrated, un-normalized scores produces arbitrary float deltas ($> 1.0$) that violate the statistical definition of Total Variation Distance.

---

## 3. Implemented Schema & Metric Adjustments (Stage B1)

The required fixes have been fully implemented in `src/behavior_lab/models.py`, `src/behavior_lab/metrics.py`, and `src/behavior_lab/sdk_adapter.py`:

1. **`SensitivityMetrics` Schema**:
   * `rank_displacement_shared: Optional[float] = None`: Mean rank displacement evaluated strictly over shared items ($I_{\text{shared}} = S_{\text{orig}}^K \cap S_{\text{mut}}^K$). Evaluates to `None` when disjoint.
   * `rank_displacement_penalized: float = 0.0`: Penalized rank displacement across all items in either list ($S_{\text{orig}}^K \cup S_{\text{mut}}^K$), assigning penalty rank $K + 1$ to evicted or newly appearing items.
   * `rank_displacement_avg: Optional[float] = None`: Aliased to `rank_displacement_shared` for backward compatibility. Evaluates to `None` when disjoint.
   * `shared_item_count: int`: Explicit count of shared items ($|I_{\text{shared}}|$).
   * `score_l1_shared: Optional[float] = None`: Mean absolute L1 delta on raw model scores for shared items.
   * `top1_score_delta: Optional[float] = None`: Absolute difference between raw model scores for top-1 items.
   * `prob_tvd_shared: Optional[float] = None`: Strictly gated to items with valid probabilities in $[0, 1]$.
2. **`PredictionItem` Schema**:
   * `score: float`: General score field.
   * `model_score: Optional[float] = None`: Explicit raw uncalibrated model score from SDK.
   * `prob: Optional[float] = None`: Reserved exclusively for calibrated probabilities; set to `None` for official SDK outputs.

---

## 4. Implementation Verification & Empirical Proof

### 4.1 Resolution of the `case_001` Anomaly
When two recommendation lists of length $K=5$ are completely disjoint (as in `case_001`):
* `shared_item_count = 0`
* `rank_displacement_shared = None` (no longer `0.0`)
* `rank_displacement_avg = None` (no longer `0.0`)
* `rank_displacement_penalized = 3.0` (with $K=5$, penalty rank is $6$; for all 10 items, $|r - 6|$ yields sum 30, $30/10 = 3.0$)

Verified in automated test `test_disjoint_lists_rank_displacement_not_misleading_zero` in `tests/test_sdk_adapter.py`.

### 4.2 Gating of Probability TVD vs. Score L1
* Total Variation Distance is strictly evaluated on valid probabilities: `0.0 <= prob <= 1.0`.
* For official BehaviorGPT SDK outputs where raw scores are uncalibrated similarities (e.g. `1.2469`), `prob` is `None` and `prob_tvd_shared` is `None`.
* Score shift is accurately captured via `score_l1_shared` and `top1_score_delta`.

### 4.3 Full Test Suite Status
All 36 unit and regression tests in the repository pass with 100% success rate:
* `tests/test_api.py`: 13/13 passed.
* `tests/test_lab_suite.py`: 7/7 passed.
* `tests/test_sdk_adapter.py`: 16/16 passed.
