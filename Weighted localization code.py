"""
Custom Metrics Evaluator for 2D Grid Cell Retrieval
===================================================

This script evaluates model predictions against ground truth cell coordinates
using a suite of custom retrieval metrics:

1. Top_N_Overlap_Percentage: Percentage of top N retrieved cells matching ground truth.
2. Top3_Precision_Score: Precision across top 3 retrieved cells (Hits / 3).
3. Rank_Weighted_Top3_Score: Weighted reward giving higher priority to Rank 1 > Rank 2 > Rank 3.
   - Rank 1 Hit = 3 pts (50.0%)
   - Rank 2 Hit = 2 pts (33.3%)
   - Rank 3 Hit = 1 pt  (16.7%)
4. Priority_Cell_Target_Reward: Reward for retrieving primary target cell (Cell 1 / Cell 2-3).
5. Overall_Composite_Score: Average across all 4 custom metrics.

Usage:
------
Python:
    python evaluate_custom_metrics.py --predictions path/to/predictions.csv --ground_truth path/to/ground_truth.xlsx

Or edit the CONFIGURABLE FILE PATHS section at the top of this script directly.
"""

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
from typing import Set, List, Dict, Tuple

# ==============================================================================
# CONFIGURABLE FILE PATHS (Set your input file paths here)
# ==============================================================================
DEFAULT_PREDICTIONS_FILE = "top10gridcells.csv"
DEFAULT_GROUND_TRUTH_FILE = "cleanestworkbook_with_cell_coords.xlsx"
DEFAULT_OUTPUT_EXCEL = "custom_metrics_evaluation_results.xlsx"
DEFAULT_OUTPUT_CSV = "custom_metrics_evaluation_results.csv"

# ==============================================================================
# COORDINATE PARSER & METRIC EVALUATION FUNCTIONS
# ==============================================================================
def parse_coords(coord_str: str) -> Set[Tuple[str, str]]:
    """
    Parses coordinate strings into a set of (row, col) string tuples.
    Supports formats like 'Cell (1, 2)', '(1, 2)', 'Cell (1,2)', etc.
    """
    if pd.isna(coord_str) or not coord_str or str(coord_str).strip().upper() in ["", "NAN", "N/A"]:
        return set()
    s = str(coord_str)
    matches = re.findall(r"Cell\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", s, re.IGNORECASE)
    if matches:
        return set(matches)
    raw_tuples = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    return set(raw_tuples)

def evaluate_single_query(
    ground_truth_cell_map: Dict[int, Set[Tuple[str, str]]],
    all_gt_coords: Set[Tuple[str, str]],
    retrieved_rank_strings: List[str]
) -> Dict[str, float]:
    """
    Evaluates a single query against ground truth using the custom metrics suite.
    """
    # Parse retrieved cell ranks into ordered unique tuples
    rank_sets = [parse_coords(r) for r in retrieved_rank_strings]
    ordered_coords = []
    for r_set in rank_sets:
        for c_tup in r_set:
            if c_tup not in ordered_coords:
                ordered_coords.append(c_tup)

    num_gt_cells = len(all_gt_coords)

    # --------------------------------------------------------------------------
    # METRIC 1: Top-N Overlap Percentage
    # --------------------------------------------------------------------------
    N_eval = max(1, num_gt_cells)
    top_n_retrieved = set(ordered_coords[:N_eval])
    overlap_count = len(top_n_retrieved & all_gt_coords)
    top_n_overlap_pct = round((overlap_count / N_eval) * 100.0, 2) if N_eval > 0 else 0.0

    # --------------------------------------------------------------------------
    # METRIC 2: Top-3 Precision Score
    # --------------------------------------------------------------------------
    top3_retrieved = set(ordered_coords[:3])
    top3_matches = len(top3_retrieved & all_gt_coords)
    top3_precision_score = round((top3_matches / 3.0) * 100.0, 2)

    # --------------------------------------------------------------------------
    # METRIC 3: Custom Rank-Weighted Top-3 Score
    # Weights: Rank 1 = 3 pts (50%), Rank 2 = 2 pts (33.3%), Rank 3 = 1 pt (16.7%)
    # Max possible = 6 pts (100.0%)
    # --------------------------------------------------------------------------
    r1_hit = 1 if len(ordered_coords) > 0 and ordered_coords[0] in all_gt_coords else 0
    r2_hit = 1 if len(ordered_coords) > 1 and ordered_coords[1] in all_gt_coords else 0
    r3_hit = 1 if len(ordered_coords) > 2 and ordered_coords[2] in all_gt_coords else 0

    rank_weighted_pts = (3 * r1_hit) + (2 * r2_hit) + (1 * r3_hit)
    rank_weighted_top3_score = round((rank_weighted_pts / 6.0) * 100.0, 2)

    # --------------------------------------------------------------------------
    # METRIC 4: Priority Cell Target Reward (Cell 1 vs Cell 2/3 Target)
    # --------------------------------------------------------------------------
    has_cell_1 = len(ground_truth_cell_map.get(1, set())) > 0
    has_cell_2_3 = len(ground_truth_cell_map.get(2, set())) > 0 or len(ground_truth_cell_map.get(3, set())) > 0

    primary_target = set()
    if has_cell_1:
        primary_target = ground_truth_cell_map[1]
    elif has_cell_2_3:
        primary_target = ground_truth_cell_map.get(2, set()) | ground_truth_cell_map.get(3, set())

    priority_reward = 0.0
    if primary_target:
        if len(ordered_coords) > 0 and ordered_coords[0] in primary_target:
            priority_reward = 100.0
        elif len(ordered_coords) > 1 and ordered_coords[1] in primary_target:
            priority_reward = 75.0
        elif len(ordered_coords) > 2 and ordered_coords[2] in primary_target:
            priority_reward = 50.0
        elif any(c in primary_target for c in ordered_coords[:5]):
            priority_reward = 35.0

    # --------------------------------------------------------------------------
    # OVERALL COMPOSITE SCORE
    # --------------------------------------------------------------------------
    composite_score = round((top_n_overlap_pct + top3_precision_score + rank_weighted_top3_score + priority_reward) / 4.0, 2)

    return {
        "Top_N_Overlap_Percentage": top_n_overlap_pct,
        "Top3_Precision_Score": top3_precision_score,
        "Rank_Weighted_Top3_Score": rank_weighted_top3_score,
        "Priority_Cell_Target_Reward": priority_reward,
        "Overall_Composite_Score": composite_score
    }

# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate 2D Grid Cell Retrieval Custom Metrics.")
    parser.add_argument("--predictions", type=str, default=DEFAULT_PREDICTIONS_FILE, help="Path to predictions CSV/Excel file")
    parser.add_argument("--ground_truth", type=str, default=DEFAULT_GROUND_TRUTH_FILE, help="Path to ground truth Excel/CSV file")
    parser.add_argument("--out_excel", type=str, default=DEFAULT_OUTPUT_EXCEL, help="Output path for evaluation Excel")
    parser.add_argument("--out_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Output path for evaluation CSV")

    args = parser.parse_args()

    pred_file = args.predictions
    gt_file = args.ground_truth

    # Load Ground Truth
    if not os.path.exists(gt_file):
        print(f"[ERROR] Ground truth file '{gt_file}' not found.")
        sys.exit(1)

    df_gt = pd.read_csv(gt_file) if gt_file.endswith(".csv") else pd.read_excel(gt_file)

    # Load Predictions
    if not os.path.exists(pred_file):
        print(f"[ERROR] Predictions file '{pred_file}' not found.")
        sys.exit(1)

    df_pred = pd.read_csv(pred_file) if pred_file.endswith(".csv") else pd.read_excel(pred_file)

    print("=" * 80)
    print("CUSTOM METRICS EVALUATION PIPELINE FOR 2D GRID CELL RETRIEVAL")
    print("=" * 80)
    print(f"Ground Truth File: {gt_file} ({len(df_gt)} rows)")
    print(f"Predictions File:  {pred_file} ({len(df_pred)} rows)")

    # Build predictions map: query_id -> list of Rank 1..10 strings
    pred_ranks_map = {}
    for _, row in df_pred.iterrows():
        qid = str(row.get("query_id") or row.get("Query_ID") or row.get("id", "")).strip()
        ranks = []
        for r in range(1, 11):
            c_val = str(row.get(f"Rank_{r}_Cell_Coord") or row.get(f"Rank_{r}") or row.get(f"rank_{r}", "")).strip()
            if c_val and c_val.upper() not in ["NAN", "N/A"]:
                ranks.append(c_val)
        pred_ranks_map[qid] = ranks

    results = []

    for _, row in df_gt.iterrows():
        qid = str(row.get("query_id") or row.get("Query_ID") or row.get("id", "")).strip()

        # Parse cell_1_coords .. cell_7_coords from ground truth
        cell_gt_map = {}
        all_gt_coords = set()

        for c_i in range(1, 8):
            c_str = str(row.get(f"cell_{c_i}_coords") or row.get(f"cell_{c_i}", ""))
            c_set = parse_coords(c_str)
            cell_gt_map[c_i] = c_set
            all_gt_coords.update(c_set)

        retrieved_ranks = pred_ranks_map.get(qid, [])
        scores = evaluate_single_query(cell_gt_map, all_gt_coords, retrieved_ranks)
        scores["query_id"] = qid
        results.append(scores)

    df_res = pd.DataFrame(results)

    # Reorder columns with query_id first
    cols = ["query_id", "Top_N_Overlap_Percentage", "Top3_Precision_Score", "Rank_Weighted_Top3_Score", "Priority_Cell_Target_Reward", "Overall_Composite_Score"]
    df_res = df_res[cols]

    # Save Results
    df_res.to_excel(args.out_excel, index=False)
    df_res.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 80)
    print("DATASET OVERALL AVERAGE METRIC SCORES")
    print("=" * 80)
    print(f"  Avg Top_N_Overlap_Percentage:      {df_res['Top_N_Overlap_Percentage'].mean():.2f}%")
    print(f"  Avg Top3_Precision_Score:          {df_res['Top3_Precision_Score'].mean():.2f}%")
    print(f"  Avg Rank_Weighted_Top3_Score:       {df_res['Rank_Weighted_Top3_Score'].mean():.2f}%")
    print(f"  Avg Priority_Cell_Target_Reward:   {df_res['Priority_Cell_Target_Reward'].mean():.2f}%")
    print(f"  Avg Overall_Composite_Score:        {df_res['Overall_Composite_Score'].mean():.2f}%")
    print("=" * 80)
    print(f"\n[SUCCESS] Evaluation results saved to:\n  1. {args.out_excel}\n  2. {args.out_csv}\n")

if __name__ == "__main__":
    main()
