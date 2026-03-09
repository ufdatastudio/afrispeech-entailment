#!/usr/bin/env python3
"""
Compute evaluation metrics grouped by difficulty level (easy/medium/hard).

This script extends the standard evaluation to support grouping by hypothesis difficulty.
It produces metrics tables that show performance separately for easy, medium, and hard hypotheses.

Expected input CSV columns:
  dataset, task, alm, difficulty, gold, pred

Example:
  python evaluation_by_difficulty.py \
    --predictions_csv predictions_with_difficulty.csv \
    --out_dir results_tables_by_difficulty

Output:
  - metrics_nli_by_difficulty.csv
  - table_nli_by_difficulty.tex
  etc.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


@dataclass
class TaskSpec:
    name: str
    label_order: Optional[List[str]] = None
    named_class_acc: bool = False
    class_acc_prefix: str = "CACC"
    nli_acc_names: Optional[Dict[str, str]] = None


def infer_label_order(y_true: pd.Series, y_pred: pd.Series) -> List[str]:
    """Infer label order from data."""
    labels = sorted(set(y_true.dropna().astype(str)).union(set(y_pred.dropna().astype(str))))
    return labels


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Compute per-class accuracy."""
    out = {}
    for lab in labels:
        mask = (y_true == lab)
        if mask.sum() == 0:
            out[lab] = np.nan
        else:
            out[lab] = float((y_pred[mask] == lab).mean())
    return out


def compute_metrics_by_difficulty(df: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    """
    Compute metrics grouped by (task, dataset, alm, difficulty).
    
    Args:
        df: DataFrame with columns: task, dataset, alm, difficulty, gold, pred
        spec: TaskSpec with label configuration
        
    Returns:
        DataFrame with one row per (task, dataset, alm, difficulty) group
    """
    required_cols = {"dataset", "task", "alm", "difficulty", "gold", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    group_cols = ["task", "dataset", "alm", "difficulty"]
    
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.dropna(subset=["gold", "pred"]).copy()
        if g.empty:
            continue

        y_true = g["gold"].astype(str).to_numpy()
        y_pred = g["pred"].astype(str).to_numpy()

        labels = spec.label_order or infer_label_order(g["gold"], g["pred"])

        # Compute accuracy metrics
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )

        class_acc = per_class_accuracy(y_true, y_pred, labels)

        # Build row
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "N": int(len(g)),
            "ACC": float(acc),
            "P_macro": float(p),
            "R_macro": float(r),
            "F1_macro": float(f1),
        })

        # Add per-class accuracy columns
        if spec.named_class_acc and spec.nli_acc_names:
            for lab, colname in spec.nli_acc_names.items():
                row[colname] = class_acc.get(lab, np.nan)
        else:
            for lab in labels:
                safe_lab = lab.replace(" ", "_")
                row[f"{spec.class_acc_prefix}_{safe_lab}"] = class_acc.get(lab, np.nan)

        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Sort by task, dataset, alm, then by difficulty order (easy, medium, hard)
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    if "difficulty" in out_df.columns:
        out_df["difficulty_order"] = out_df["difficulty"].map(difficulty_order)
        out_df = out_df.sort_values(
            by=["task", "dataset", "alm", "difficulty_order"],
            na_position="last"
        ).drop(columns=["difficulty_order"]).reset_index(drop=True)
    
    return out_df


def to_latex_table_by_difficulty(task_df: pd.DataFrame, task_name: str, out_path: str) -> None:
    """
    Export a LaTeX table with difficulty breakdown.
    
    Format:
    Dataset | ALM | Difficulty | ACC | P | R | F1 | EACC | NACC | CACC
    """
    # Rename columns for display
    rename_map = {
        "P_macro": "P",
        "R_macro": "R",
        "F1_macro": "F1",
    }
    latex_df = task_df.rename(columns=rename_map).copy()

    # Column order
    first_cols = ["dataset", "alm", "difficulty"]
    metric_cols = ["ACC", "P", "R", "F1"]
    other_cols = [c for c in latex_df.columns 
                  if c not in (first_cols + metric_cols + ["task", "N"])]
    cols = first_cols + metric_cols + other_cols

    # Format floats
    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{x:.4f}"
        return str(x)

    latex_df = latex_df[cols].copy()
    for c in latex_df.columns:
        latex_df[c] = latex_df[c].map(fmt)

    caption = f"Zero-shot performance on {task_name} (grouped by difficulty level)."
    label = f"tab:{task_name.lower().replace(' ', '_')}_difficulty"

    latex = latex_df.to_latex(
        index=False,
        escape=True,
        column_format="l" * len(cols),
        caption=caption,
        label=label
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics grouped by hypothesis difficulty level."
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="CSV with columns: dataset, task, alm, difficulty, gold, pred"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for metrics CSVs and LaTeX tables"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)

    # Task specifications
    task_specs: Dict[str, TaskSpec] = {
        "nli": TaskSpec(
            name="NLI",
            label_order=["entailment", "neutral", "contradiction"],
            named_class_acc=True,
            nli_acc_names={"entailment": "EACC", "neutral": "NACC", "contradiction": "CACC"},
        ),
        "consistency": TaskSpec(
            name="Consistency",
            label_order=["consistent", "inconsistent"],
            named_class_acc=False,
            class_acc_prefix="ACC",
        ),
        "intent": TaskSpec(
            name="Intent",
            label_order=None,
            named_class_acc=False,
            class_acc_prefix="ACC",
        ),
        "plausibility": TaskSpec(
            name="Plausibility",
            label_order=["plausible", "implausible"],
            named_class_acc=False,
            class_acc_prefix="ACC",
        ),
        "commonsense": TaskSpec(
            name="Commonsense",
            label_order=["yes", "no"],
            named_class_acc=False,
            class_acc_prefix="ACC",
        ),
    }

    # Compute metrics for each task
    tasks_in_data = sorted(df["task"].dropna().astype(str).unique().tolist())

    for task in tasks_in_data:
        if task not in task_specs:
            task_specs[task] = TaskSpec(name=task, label_order=None)

        spec = task_specs[task]
        sub = df[df["task"].astype(str) == task].copy()
        
        # Check if difficulty column exists
        if "difficulty" not in sub.columns:
            print(f"[SKIP] {task}: no 'difficulty' column in data")
            continue
        
        metrics_df = compute_metrics_by_difficulty(sub, spec)

        # Save metrics CSV
        metrics_csv_path = os.path.join(args.out_dir, f"metrics_{task}_by_difficulty.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Save LaTeX table
        latex_path = os.path.join(args.out_dir, f"table_{task}_by_difficulty.tex")
        to_latex_table_by_difficulty(metrics_df, spec.name, latex_path)

        print(f"[OK] {task}: wrote {metrics_csv_path}")
        print(f"[OK] {task}: wrote {latex_path}")
        print(metrics_df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
