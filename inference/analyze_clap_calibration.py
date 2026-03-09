#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


ACCENT_GOLD_TO_BOOL = {
    "ACCENT_INVARIANT": "TRUE",
    "ACCENT_TEST": "FALSE",
}

PREFERRED_POSITIVE_LABELS = [
    "TRUE",
    "CONSISTENT",
    "PLAUSIBLE",
    "SUPPORTED",
    "ENTAILMENT",
]


@dataclass
class BinaryRecord:
    gold: str
    pred_argmax: str
    score_pos: float
    score_neg: float


def safe_mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def safe_std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return safe_mean(f1s)


def balanced_accuracy_binary(gold_bin: Sequence[int], pred_bin: Sequence[int]) -> float:
    if not gold_bin:
        return float("nan")
    tp = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 0)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


def pick_positive_label(labels: Sequence[str]) -> str:
    label_set = set(labels)
    for preferred in PREFERRED_POSITIVE_LABELS:
        if preferred in label_set:
            return preferred
    return sorted(labels)[0]


def map_gold(gold: str, labels: Sequence[str]) -> Optional[str]:
    if gold in labels:
        return gold
    mapped = ACCENT_GOLD_TO_BOOL.get(gold)
    if mapped in labels:
        return mapped
    return None


def auc_roc(scores: Sequence[float], y_true: Sequence[int]) -> float:
    pos = [s for s, y in zip(scores, y_true) if y == 1]
    neg = [s for s, y in zip(scores, y_true) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def d_prime(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    if not pos_scores or not neg_scores:
        return float("nan")
    mu1 = safe_mean(pos_scores)
    mu0 = safe_mean(neg_scores)
    s1 = safe_std(pos_scores)
    s0 = safe_std(neg_scores)
    pooled = math.sqrt((s1 * s1 + s0 * s0) / 2.0)
    if pooled == 0:
        return float("nan")
    return (mu1 - mu0) / pooled


def candidate_thresholds(values: Sequence[float]) -> List[float]:
    uniq = sorted(set(values))
    if not uniq:
        return [0.0]
    cands = [uniq[0] - 1e-6]
    for a, b in zip(uniq, uniq[1:]):
        cands.append((a + b) / 2.0)
    cands.append(uniq[-1] + 1e-6)
    return cands


def parse_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def analyze_binary(rows: Sequence[Dict], labels: Sequence[str]) -> Dict:
    pos_label = pick_positive_label(labels)
    neg_label = [l for l in labels if l != pos_label][0]

    records: List[BinaryRecord] = []
    skipped = 0
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            skipped += 1
            continue
        gold = map_gold(str(row.get("gold")), labels)
        if gold is None:
            skipped += 1
            continue
        try:
            score_pos = float(scores[pos_label])
            score_neg = float(scores[neg_label])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        pred_argmax = pos_label if score_pos >= score_neg else neg_label
        records.append(BinaryRecord(gold=gold, pred_argmax=pred_argmax, score_pos=score_pos, score_neg=score_neg))

    gold = [r.gold for r in records]
    pred_argmax = [r.pred_argmax for r in records]
    y_true = [1 if r.gold == pos_label else 0 for r in records]
    score_pos = [r.score_pos for r in records]
    margins = [abs(r.score_pos - r.score_neg) for r in records]

    baseline_acc = accuracy(gold, pred_argmax)
    baseline_f1 = macro_f1(gold, pred_argmax, [neg_label, pos_label])
    baseline_bal_acc = balanced_accuracy_binary(y_true, [1 if p == pos_label else 0 for p in pred_argmax])

    best = {
        "threshold": None,
        "macro_f1": -1.0,
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
    }
    for t in candidate_thresholds(score_pos):
        pred_t = [pos_label if s >= t else neg_label for s in score_pos]
        macro = macro_f1(gold, pred_t, [neg_label, pos_label])
        if macro > best["macro_f1"]:
            best["threshold"] = t
            best["macro_f1"] = macro
            best["accuracy"] = accuracy(gold, pred_t)
            best["balanced_accuracy"] = balanced_accuracy_binary(y_true, [1 if p == pos_label else 0 for p in pred_t])

    margin_cands = candidate_thresholds(margins)
    coverage_best = {
        "min_cov": 0.8,
        "margin_threshold": None,
        "coverage": 0.0,
        "kept_accuracy": float("nan"),
    }
    for mt in margin_cands:
        kept = [i for i, m in enumerate(margins) if m >= mt]
        if not kept:
            continue
        cov = len(kept) / len(records) if records else 0.0
        if cov < coverage_best["min_cov"]:
            continue
        kept_acc = accuracy([gold[i] for i in kept], [pred_argmax[i] for i in kept])
        if math.isnan(coverage_best["kept_accuracy"]) or kept_acc > coverage_best["kept_accuracy"]:
            coverage_best["margin_threshold"] = mt
            coverage_best["coverage"] = cov
            coverage_best["kept_accuracy"] = kept_acc

    pos_scores_for_true = [s for s, y in zip(score_pos, y_true) if y == 1]
    pos_scores_for_false = [s for s, y in zip(score_pos, y_true) if y == 0]

    correct_margins = [m for m, g, p in zip(margins, gold, pred_argmax) if g == p]
    wrong_margins = [m for m, g, p in zip(margins, gold, pred_argmax) if g != p]

    return {
        "n": len(records),
        "n_skipped": skipped,
        "labels": [neg_label, pos_label],
        "positive_label": pos_label,
        "baseline_accuracy": baseline_acc,
        "baseline_macro_f1": baseline_f1,
        "baseline_balanced_accuracy": baseline_bal_acc,
        "threshold_best": best,
        "margin_selective": coverage_best,
        "auroc_pos_score": auc_roc(score_pos, y_true),
        "d_prime_pos_score": d_prime(pos_scores_for_true, pos_scores_for_false),
        "mean_margin_correct": safe_mean(correct_margins),
        "mean_margin_wrong": safe_mean(wrong_margins),
        "mean_pos_score_true": safe_mean(pos_scores_for_true),
        "mean_pos_score_false": safe_mean(pos_scores_for_false),
    }


def analyze_multiclass(rows: Sequence[Dict], labels: Sequence[str]) -> Dict:
    kept = []
    skipped = 0
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict):
            skipped += 1
            continue
        gold = map_gold(str(row.get("gold")), labels)
        if gold is None:
            skipped += 1
            continue
        try:
            scored = [(label, float(scores[label])) for label in labels]
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        pred = scored_sorted[0][0]
        margin = scored_sorted[0][1] - scored_sorted[1][1]
        kept.append((gold, pred, margin))

    gold = [x[0] for x in kept]
    pred = [x[1] for x in kept]
    margins = [x[2] for x in kept]
    correct_margins = [m for (g, p, m) in kept if g == p]
    wrong_margins = [m for (g, p, m) in kept if g != p]

    return {
        "n": len(kept),
        "n_skipped": skipped,
        "labels": list(labels),
        "baseline_accuracy": accuracy(gold, pred),
        "baseline_macro_f1": macro_f1(gold, pred, labels),
        "mean_top1_top2_margin": safe_mean(margins),
        "mean_margin_correct": safe_mean(correct_margins),
        "mean_margin_wrong": safe_mean(wrong_margins),
    }


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.4f}"


def discover_files(roots: Sequence[str]) -> List[str]:
    paths = []
    for root in roots:
        paths.extend(glob.glob(os.path.join(root, "*", "results", "*.jsonl")))
    return sorted([p for p in paths if not p.endswith("_errors.jsonl")])


def analyze_file(path: str) -> Dict:
    rows = parse_jsonl(path)
    sample = next((r for r in rows if isinstance(r.get("scores"), dict)), None)
    if sample is None:
        return {"path": path, "type": "invalid", "n": 0, "n_skipped": len(rows)}

    labels = sorted(sample["scores"].keys())
    model_family = path.split(os.sep)[1] if len(path.split(os.sep)) > 1 else "unknown"
    task = path.split(os.sep)[2] if len(path.split(os.sep)) > 2 else "unknown"

    out = {
        "path": path,
        "model_family": model_family,
        "task": task,
    }
    if len(labels) == 2:
        out["type"] = "binary"
        out.update(analyze_binary(rows, labels))
    else:
        out["type"] = "multiclass"
        out.update(analyze_multiclass(rows, labels))
    return out


def write_csv(results: Sequence[Dict], out_csv: str) -> None:
    fields = [
        "model_family",
        "task",
        "path",
        "type",
        "n",
        "n_skipped",
        "baseline_accuracy",
        "baseline_macro_f1",
        "baseline_balanced_accuracy",
        "threshold_best_macro_f1",
        "threshold_best_accuracy",
        "threshold_best_balanced_accuracy",
        "threshold_best_threshold",
        "margin_selective_coverage",
        "margin_selective_kept_accuracy",
        "margin_selective_threshold",
        "auroc_pos_score",
        "d_prime_pos_score",
        "mean_top1_top2_margin",
        "mean_margin_correct",
        "mean_margin_wrong",
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "model_family": r.get("model_family"),
                "task": r.get("task"),
                "path": r.get("path"),
                "type": r.get("type"),
                "n": r.get("n"),
                "n_skipped": r.get("n_skipped"),
                "baseline_accuracy": r.get("baseline_accuracy"),
                "baseline_macro_f1": r.get("baseline_macro_f1"),
                "baseline_balanced_accuracy": r.get("baseline_balanced_accuracy"),
                "threshold_best_macro_f1": r.get("threshold_best", {}).get("macro_f1"),
                "threshold_best_accuracy": r.get("threshold_best", {}).get("accuracy"),
                "threshold_best_balanced_accuracy": r.get("threshold_best", {}).get("balanced_accuracy"),
                "threshold_best_threshold": r.get("threshold_best", {}).get("threshold"),
                "margin_selective_coverage": r.get("margin_selective", {}).get("coverage"),
                "margin_selective_kept_accuracy": r.get("margin_selective", {}).get("kept_accuracy"),
                "margin_selective_threshold": r.get("margin_selective", {}).get("margin_threshold"),
                "auroc_pos_score": r.get("auroc_pos_score"),
                "d_prime_pos_score": r.get("d_prime_pos_score"),
                "mean_top1_top2_margin": r.get("mean_top1_top2_margin"),
                "mean_margin_correct": r.get("mean_margin_correct"),
                "mean_margin_wrong": r.get("mean_margin_wrong"),
            }
            writer.writerow(row)


def write_markdown(results: Sequence[Dict], out_md: str) -> None:
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    binaries = [r for r in results if r.get("type") == "binary"]
    multiclass = [r for r in results if r.get("type") == "multiclass"]

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# CLAP Calibration and Separability Audit\n\n")
        f.write("This report compares default argmax decoding against simple post-hoc thresholding for binary tasks, and reports score separability diagnostics.\n\n")

        f.write("## Binary Tasks (Threshold + Separability)\n\n")
        f.write("| Model | Task | N | Argmax Acc | Argmax Macro-F1 | Best Thr Acc | Best Thr Macro-F1 | AUROC(pos score) | d' | Margin(✓) | Margin(✗) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in binaries:
            thr = r.get("threshold_best", {})
            f.write(
                "| {mf} | {task} | {n} | {a0} | {f0} | {a1} | {f1} | {auroc} | {dprime} | {mc} | {mw} |\n".format(
                    mf=r.get("model_family"),
                    task=r.get("task"),
                    n=r.get("n", 0),
                    a0=fmt(r.get("baseline_accuracy")),
                    f0=fmt(r.get("baseline_macro_f1")),
                    a1=fmt(thr.get("accuracy")),
                    f1=fmt(thr.get("macro_f1")),
                    auroc=fmt(r.get("auroc_pos_score")),
                    dprime=fmt(r.get("d_prime_pos_score")),
                    mc=fmt(r.get("mean_margin_correct")),
                    mw=fmt(r.get("mean_margin_wrong")),
                )
            )

        f.write("\n## Multiclass Tasks (Argmax + Margin Diagnostics)\n\n")
        f.write("| Model | Task | N | Argmax Acc | Argmax Macro-F1 | Mean Top1-Top2 Margin | Margin(✓) | Margin(✗) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in multiclass:
            f.write(
                "| {mf} | {task} | {n} | {a0} | {f0} | {m} | {mc} | {mw} |\n".format(
                    mf=r.get("model_family"),
                    task=r.get("task"),
                    n=r.get("n", 0),
                    a0=fmt(r.get("baseline_accuracy")),
                    f0=fmt(r.get("baseline_macro_f1")),
                    m=fmt(r.get("mean_top1_top2_margin")),
                    mc=fmt(r.get("mean_margin_correct")),
                    mw=fmt(r.get("mean_margin_wrong")),
                )
            )

        if binaries:
            gain_acc = [
                r["threshold_best"]["accuracy"] - r["baseline_accuracy"]
                for r in binaries
                if r.get("threshold_best") and not math.isnan(r["baseline_accuracy"])
            ]
            gain_f1 = [
                r["threshold_best"]["macro_f1"] - r["baseline_macro_f1"]
                for r in binaries
                if r.get("threshold_best") and not math.isnan(r["baseline_macro_f1"])
            ]
            f.write("\n## Aggregate Binary Calibration Effect\n\n")
            f.write(f"- Mean accuracy gain from thresholding: **{fmt(safe_mean(gain_acc))}**\n")
            f.write(f"- Mean macro-F1 gain from thresholding: **{fmt(safe_mean(gain_f1))}**\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CLAP/MSCLAP calibration and separability from result JSONLs.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["outputs/CLAP", "outputs/MSCLAP", "outputs/MSCLAP_2022"],
        help="Root output directories to scan.",
    )
    parser.add_argument("--out-csv", default="outputs/clap_audit/clap_calibration_summary.csv")
    parser.add_argument("--out-md", default="outputs/clap_audit/CLAP_CALIBRATION_AUDIT.md")
    parser.add_argument("--out-json", default="outputs/clap_audit/clap_calibration_summary.json")
    args = parser.parse_args()

    files = discover_files(args.roots)
    results = [analyze_file(path) for path in files]

    write_csv(results, args.out_csv)
    write_markdown(results, args.out_md)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Analyzed {len(files)} files")
    print(f"CSV: {args.out_csv}")
    print(f"MD : {args.out_md}")
    print(f"JSON: {args.out_json}")


if __name__ == "__main__":
    main()
