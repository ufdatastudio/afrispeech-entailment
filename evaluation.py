#!/usr/bin/env python3
"""
Compute Table-4 style evaluation metrics for audio-semantic tasks.

Produces:
- One metrics CSV per task
- One LaTeX table per task (grouped by task; includes dataset column)

Expected input CSV columns:
  dataset, task, alm, gold, pred
Optional:
  llm_judge

Example:
  python eval_tables.py \
    --predictions_csv predictions_all_tasks.csv \
    --out_dir results_tables

Notes:
- For multi-class tasks (e.g., NLI), computes ACC, macro-P/R/F1, and per-class accuracies.
- For binary tasks, also computes ACC, macro-P/R/F1, and per-class accuracies for both labels.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


@dataclass
class TaskSpec:
    name: str
    label_order: Optional[List[str]] = None  # if None, inferred from data
    # Whether to compute special named per-class accuracy columns (EACC/NACC/CACC for NLI-like tasks)
    named_class_acc: bool = False
    class_acc_prefix: str = "CACC"  # generic prefix if not named_class_acc
    # For NLI-style naming
    nli_acc_names: Optional[Dict[str, str]] = None  # e.g., {"entailment":"EACC", ...}


def infer_label_order(y_true: pd.Series, y_pred: pd.Series) -> List[str]:
    labels = sorted(set(y_true.dropna().astype(str)).union(set(y_pred.dropna().astype(str))))
    return labels


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    out = {}
    for lab in labels:
        mask = (y_true == lab)
        if mask.sum() == 0:
            out[lab] = np.nan
        else:
            out[lab] = float((y_pred[mask] == lab).mean())
    return out


def compute_metrics(df: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    """
    Compute metrics grouped by (dataset, alm, [llm_judge], task).
    """
    required_cols = {"dataset", "task", "alm", "gold", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    group_cols = ["task", "dataset", "alm"]
    if "llm_judge" in df.columns:
        group_cols.append("llm_judge")

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.dropna(subset=["gold", "pred"]).copy()
        if g.empty:
            continue

        y_true = g["gold"].astype(str).to_numpy()
        y_pred = g["pred"].astype(str).to_numpy()

        labels = spec.label_order or infer_label_order(g["gold"], g["pred"])

        acc = accuracy_score(y_true, y_pred)

        # macro P/R/F1 (works for binary and multi-class)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )

        class_acc = per_class_accuracy(y_true, y_pred, labels)

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "N": int(len(g)),
            "ACC": float(acc),
            "P_macro": float(p),
            "R_macro": float(r),
            "F1_macro": float(f1),
        })

        # Add per-class accuracy columns.
        if spec.named_class_acc and spec.nli_acc_names:
            # e.g., entailment->EACC, neutral->NACC, contradiction->CACC
            for lab, colname in spec.nli_acc_names.items():
                row[colname] = class_acc.get(lab, np.nan)
        else:
            # generic class accuracy columns
            for lab in labels:
                safe_lab = lab.replace(" ", "_")
                row[f"{spec.class_acc_prefix}_{safe_lab}"] = class_acc.get(lab, np.nan)

        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Sort nicely if present
    sort_cols = [c for c in ["task", "dataset", "alm", "llm_judge"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    return out_df


def to_latex_table(task_df: pd.DataFrame, task_name: str, out_path: str) -> None:
    """
    Export a LaTeX table roughly matching Table-4 structure:
    Dataset | ALM | (optional LLM judge) | ACC | P | R | F1 | per-class accs...
    """
    # Rename columns to match paper style
    rename_map = {
        "P_macro": "P",
        "R_macro": "R",
        "F1_macro": "F1",
    }
    latex_df = task_df.rename(columns=rename_map).copy()

    # Keep core columns first
    first_cols = ["dataset", "alm"]
    if "llm_judge" in latex_df.columns:
        first_cols.append("llm_judge")
    metric_cols = ["ACC", "P", "R", "F1"]

    # everything else after
    other_cols = [c for c in latex_df.columns if c not in (first_cols + metric_cols + ["task", "N"])]
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

    caption = f"Zero-shot performance on {task_name} (grouped by task)."
    label = f"tab:{task_name.lower().replace(' ', '_')}_results"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", type=str, required=True, help="CSV with dataset/task/alm/gold/pred (+ optional llm_judge).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for per-task metrics + LaTeX tables.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)

    # Define task specs.
    # Adjust label names below to EXACTLY match your gold/pred label strings.
    task_specs: Dict[str, TaskSpec] = {
        "nli": TaskSpec(
            name="NLI",
            label_order=["entailment", "neutral", "contradiction"],
            named_class_acc=True,
            nli_acc_names={"entailment": "EACC", "neutral": "NACC", "contradiction": "CACC"},
        ),
        "consistency": TaskSpec(
            name="Consistency",
            # Example binary labels; change to your actual labels
            label_order=["consistent", "inconsistent"],
            named_class_acc=False,
            class_acc_prefix="ACC",
        ),
        "intent": TaskSpec(
            name="Intent",
            # If you have fixed intent labels, set them here; else leave None to infer.
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
            # If binary: yes/no (or supported/unsupported). If 3-way, set list of 3 labels.
            label_order=["yes", "no"],
            named_class_acc=False,
            class_acc_prefix="ACC",
        ),
    }

    # Compute and export per-task tables
    tasks_in_data = sorted(df["task"].dropna().astype(str).unique().tolist())

    for task in tasks_in_data:
        if task not in task_specs:
            # fallback: infer labels
            task_specs[task] = TaskSpec(name=task, label_order=None)

        spec = task_specs[task]
        sub = df[df["task"].astype(str) == task].copy()
        metrics_df = compute_metrics(sub, spec)

        # Save metrics as CSV
        metrics_csv_path = os.path.join(args.out_dir, f"metrics_{task}.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Save LaTeX table
        latex_path = os.path.join(args.out_dir, f"table_{task}.tex")
        to_latex_table(metrics_df, spec.name, latex_path)

        print(f"[OK] {task}: wrote {metrics_csv_path} and {latex_path}")


if __name__ == "__main__":
    main()
