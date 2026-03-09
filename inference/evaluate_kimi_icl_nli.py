#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


ROOT = Path("/orange/ufdatastudios/c.okocha/afrispeech-entailment")
ICL_OUT = ROOT / "outputs" / "ICL_overlay_nli"
RESULT_OUT = ICL_OUT / "kimi_eval"

VARIANTS = ["audio_only", "audio_plus_transcript"]
DATASETS = ["afri200", "medical"]
SHOTS = [1, 3, 5, 10]
LABELS = ["entailment", "neutral", "contradiction"]


def normalize_label(value: str) -> str:
    if value is None:
        return "unparseable"
    text = str(value).upper().strip()
    for lab in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]:
        if re.search(rf"\b{lab}\b", text):
            return lab.lower()
    return "unparseable"


def strip_icl_prefix(hyp: str) -> str:
    if not isinstance(hyp, str):
        return ""
    marker = "Target Hypothesis:"
    end_marker = "\nAnswer:"
    if marker in hyp:
        after = hyp.split(marker, 1)[1]
        if end_marker in after:
            return after.split(end_marker, 1)[0].strip()
    return hyp.strip()


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["gold_norm"].tolist()
    y_pred = df["pred_norm"].tolist()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    _, recalls, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "N": int(len(df)),
        "ACC": float(acc),
        "P_macro": float(p),
        "R_macro": float(r),
        "F1_macro": float(f1),
        "EACC": float(recalls[0]),
        "NACC": float(recalls[1]),
        "CACC": float(recalls[2]),
        "unparseable_rate": float((df["pred_norm"] == "unparseable").mean()),
    }


def main() -> None:
    RESULT_OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for variant in VARIANTS:
        for dataset in DATASETS:
            for shot in SHOTS:
                pred_file = ICL_OUT / variant / "kimi" / dataset / f"shot{shot}" / "predictions.jsonl"
                if not pred_file.exists():
                    missing.append(str(pred_file))
                    continue

                with pred_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        rows.append(
                            {
                                "variant": variant,
                                "dataset": dataset,
                                "shot": shot,
                                "model": "kimi",
                                "item_id": obj.get("item_id", ""),
                                "file_name": obj.get("file_name", ""),
                                "hypothesis_prefixed": obj.get("hypothesis", ""),
                                "hypothesis": strip_icl_prefix(obj.get("hypothesis", "")),
                                "gold": obj.get("gold", ""),
                                "pred": obj.get("pred", ""),
                                "pred_raw": obj.get("pred_raw", ""),
                                "error": obj.get("error", ""),
                                "gold_norm": normalize_label(obj.get("gold", "")),
                                "pred_norm": normalize_label(obj.get("pred_raw", "") or obj.get("pred", "")),
                                "source_jsonl": str(pred_file),
                            }
                        )

    if missing:
        print("WARNING: Missing files:")
        for m in missing:
            print(m)

    if not rows:
        raise SystemExit("No Kimi ICL predictions found.")

    df = pd.DataFrame(rows)
    agg_csv = RESULT_OUT / "kimi_icl_predictions_aggregated.csv"
    df.to_csv(agg_csv, index=False)

    by_vsd_rows = []
    for (variant, shot, dataset), g in df.groupby(["variant", "shot", "dataset"]):
        met = compute_metrics(g)
        by_vsd_rows.append({"variant": variant, "shot": int(shot), "dataset": dataset, **met})
    by_vsd = pd.DataFrame(by_vsd_rows).sort_values(["variant", "dataset", "shot"]).reset_index(drop=True)
    by_vsd_csv = RESULT_OUT / "kimi_icl_metrics_by_variant_shot_dataset.csv"
    by_vsd.to_csv(by_vsd_csv, index=False)

    by_vs_rows = []
    for (variant, shot), g in df.groupby(["variant", "shot"]):
        met = compute_metrics(g)
        by_vs_rows.append({"variant": variant, "shot": int(shot), **met})
    by_vs = pd.DataFrame(by_vs_rows).sort_values(["variant", "shot"]).reset_index(drop=True)
    by_vs_csv = RESULT_OUT / "kimi_icl_metrics_by_variant_shot_overall.csv"
    by_vs.to_csv(by_vs_csv, index=False)

    overall_rows = []
    for variant, g in df.groupby("variant"):
        met = compute_metrics(g)
        overall_rows.append({"variant": variant, **met})
    overall = pd.DataFrame(overall_rows).sort_values(["variant"]).reset_index(drop=True)
    overall_csv = RESULT_OUT / "kimi_icl_metrics_overall_by_variant.csv"
    overall.to_csv(overall_csv, index=False)

    with (RESULT_OUT / "kimi_icl_metrics_by_variant_shot_overall.tex").open("w", encoding="utf-8") as f:
        f.write(by_vs.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True,
                               caption="Kimi ICL NLI results by variant and shot (Afri200+Medical pooled).",
                               label="tab:kimi_icl_variant_shot_overall"))

    with (RESULT_OUT / "kimi_icl_metrics_by_variant_shot_dataset.tex").open("w", encoding="utf-8") as f:
        f.write(by_vsd.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True,
                                caption="Kimi ICL NLI results by variant, dataset, and shot.",
                                label="tab:kimi_icl_variant_shot_dataset"))

    print(f"Wrote: {agg_csv}")
    print(f"Wrote: {by_vsd_csv}")
    print(f"Wrote: {by_vs_csv}")
    print(f"Wrote: {overall_csv}")
    print("\nBy variant + shot (overall):")
    print(by_vs.to_string(index=False))


if __name__ == "__main__":
    main()
