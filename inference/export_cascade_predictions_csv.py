#!/usr/bin/env python3
"""Aggregate cascade JSONL prediction files into one evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path


def _match_label(text: str, labels: tuple[str, ...]) -> str:
    for label in sorted(labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label
    return ""


def parse_from_raw(task: str, pred_raw: str) -> str:
    if not pred_raw:
        return ""
    text = str(pred_raw).upper()

    if task == "nli":
        return _match_label(text, ("ENTAILMENT", "CONTRADICTION", "NEUTRAL")).lower()
    if task == "consistency":
        return _match_label(text, ("INCONSISTENT", "CONSISTENT")).lower()
    if task == "plausibility":
        return _match_label(text, ("IMPLAUSIBLE", "PLAUSIBLE")).lower()
    if task == "restraint":
        return _match_label(text, ("UNSUPPORTED", "SUPPORTED")).lower()
    if task == "accent_drift":
        return _match_label(text, ("FALSE", "TRUE")).lower()
    return ""


def normalize_label(task: str, label: str) -> str:
    if label is None:
        return ""
    text = str(label).strip()
    if not text:
        return ""

    if task in {"nli", "consistency", "plausibility", "restraint", "accent_drift"}:
        return text.lower()
    return text


def resolve_prediction(task: str, pred: str, pred_raw: str) -> str:
    # pred_raw is the main source of truth for completed runs.
    raw_norm = parse_from_raw(task, pred_raw)
    if raw_norm:
        return raw_norm

    # Fallback to parsed `pred` only if pred_raw cannot be mapped.
    pred_norm = normalize_label(task, pred)
    if pred_norm and pred_norm != "unparseable":
        return pred_norm

    return pred_norm


def infer_dataset(task_key: str) -> str:
    if task_key.startswith("afri200_"):
        return "AfriSpeech200"
    if task_key.startswith("medical_"):
        return "Medical"
    if task_key.startswith("general_"):
        return "AfriSpeechGeneral"
    if task_key.startswith("afrinames_"):
        return "AfriNames"
    return "Unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cascade JSONL predictions to CSV")
    parser.add_argument(
        "--input_glob",
        default="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/cascade_asr_text_llm/*/*/*.jsonl",
        help="Glob for cascade JSONL files",
    )
    parser.add_argument(
        "--output_csv",
        default="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/cascade_asr_text_llm/predictions_all_tasks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    rows = []
    for file_path in files:
        rel_parts = Path(file_path).parts
        # .../cascade_asr_text_llm/{asr_model}/{text_llm}/{task_key}.jsonl
        asr_model = rel_parts[-3]
        text_llm = rel_parts[-2]
        task_key = Path(file_path).stem

        if task_key.endswith("_nli"):
            task = "nli"
        elif task_key.endswith("_consistency"):
            task = "consistency"
        elif task_key.endswith("_plausibility"):
            task = "plausibility"
        elif task_key.endswith("_restraint"):
            task = "restraint"
        elif task_key.endswith("_accent_drift"):
            task = "accent_drift"
        else:
            task = "unknown"

        dataset = infer_dataset(task_key)
        alm_name = f"cascade_{asr_model}_{text_llm}"

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "alm": alm_name,
                        "asr_model": asr_model,
                        "text_llm": text_llm,
                        "task_key": task_key,
                        "item_id": obj.get("item_id", ""),
                        "file_name": obj.get("file_name", ""),
                        "hypothesis": obj.get("hypothesis", ""),
                        "gold": normalize_label(task, obj.get("gold", "")),
                        "pred": resolve_prediction(task, obj.get("pred", ""), obj.get("pred_raw", "")),
                        "pred_raw": obj.get("pred_raw", ""),
                        "error": obj.get("error", ""),
                        "source_jsonl": file_path,
                    }
                )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = [
        "task",
        "dataset",
        "alm",
        "asr_model",
        "text_llm",
        "task_key",
        "item_id",
        "file_name",
        "hypothesis",
        "gold",
        "pred",
        "pred_raw",
        "error",
        "source_jsonl",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
