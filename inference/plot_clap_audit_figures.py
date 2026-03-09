#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_float(value: str):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["baseline_accuracy"] = parse_float(row.get("baseline_accuracy"))
            row["baseline_macro_f1"] = parse_float(row.get("baseline_macro_f1"))
            row["threshold_best_accuracy"] = parse_float(row.get("threshold_best_accuracy"))
            row["threshold_best_macro_f1"] = parse_float(row.get("threshold_best_macro_f1"))
            row["auroc_pos_score"] = parse_float(row.get("auroc_pos_score"))
            rows.append(row)
    return rows


def save_binary_gain_plot(rows: List[Dict], metric_base: str, metric_thr: str, title: str, ylabel: str, out_path: str):
    binary_rows = [r for r in rows if r.get("type") == "binary"]
    binary_rows.sort(key=lambda r: (r.get("model_family", ""), r.get("task", "")))

    labels = [f"{r['model_family']}\n{r['task']}" for r in binary_rows]
    base_vals = [r.get(metric_base) for r in binary_rows]
    thr_vals = [r.get(metric_thr) for r in binary_rows]

    x = list(range(len(binary_rows)))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(binary_rows) * 0.45), 5.5))
    ax.bar([i - width / 2 for i in x], base_vals, width=width, label="Argmax", color="#6c8ebf")
    ax.bar([i + width / 2 for i in x], thr_vals, width=width, label="Best Threshold", color="#93c47d")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_binary_auroc_plot(rows: List[Dict], out_path: str):
    binary_rows = [r for r in rows if r.get("type") == "binary"]
    binary_rows.sort(key=lambda r: (r.get("model_family", ""), r.get("task", "")))

    labels = [f"{r['model_family']}\n{r['task']}" for r in binary_rows]
    auroc_vals = [r.get("auroc_pos_score") for r in binary_rows]
    colors = ["#d6604d" if (v is not None and v < 0.55) else "#4393c3" for v in auroc_vals]

    x = list(range(len(binary_rows)))
    fig, ax = plt.subplots(figsize=(max(12, len(binary_rows) * 0.45), 5.5))
    ax.bar(x, auroc_vals, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_title("Binary Tasks: AUROC of Positive-Class Score")
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_multiclass_accuracy_plot(rows: List[Dict], out_path: str):
    multi_rows = [r for r in rows if r.get("type") == "multiclass"]
    multi_rows.sort(key=lambda r: (r.get("model_family", ""), r.get("task", "")))

    labels = [f"{r['model_family']}\n{r['task']}" for r in multi_rows]
    acc_vals = [r.get("baseline_accuracy") for r in multi_rows]

    x = list(range(len(multi_rows)))
    fig, ax = plt.subplots(figsize=(max(10, len(multi_rows) * 0.75), 5))
    ax.bar(x, acc_vals, color="#8e7cc3")
    ax.axhline(1 / 3, color="black", linestyle="--", linewidth=1.0, label="3-way chance")
    ax.set_title("Multiclass NLI: Argmax Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CLAP calibration audit figures from summary CSV.")
    parser.add_argument("--csv", default="outputs/clap_audit/clap_calibration_summary.csv")
    parser.add_argument("--out-dir", default="outputs/clap_audit/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(args.csv)

    save_binary_gain_plot(
        rows,
        metric_base="baseline_accuracy",
        metric_thr="threshold_best_accuracy",
        title="Binary Tasks: Argmax vs Threshold Accuracy",
        ylabel="Accuracy",
        out_path=os.path.join(args.out_dir, "binary_accuracy_argmax_vs_threshold.png"),
    )
    save_binary_gain_plot(
        rows,
        metric_base="baseline_macro_f1",
        metric_thr="threshold_best_macro_f1",
        title="Binary Tasks: Argmax vs Threshold Macro-F1",
        ylabel="Macro-F1",
        out_path=os.path.join(args.out_dir, "binary_macro_f1_argmax_vs_threshold.png"),
    )
    save_binary_auroc_plot(rows, os.path.join(args.out_dir, "binary_auroc_by_task.png"))
    save_multiclass_accuracy_plot(rows, os.path.join(args.out_dir, "multiclass_nli_accuracy.png"))

    print(f"Saved figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
