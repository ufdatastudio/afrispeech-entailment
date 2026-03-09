#!/usr/bin/env python3
"""
Convert inference output JSONL to evaluation CSV format with difficulty levels.

This script takes output JSONL files from ALM inference and converts them to
CSV format suitable for evaluation_by_difficulty.py.

Usage:
    python convert_to_difficulty_csv.py \
      --input_jsonl outputs/interview_nli_AudioFlamingo3Local.jsonl \
      --dataset interview \
      --task nli \
      --alm AudioFlamingo3Local \
      --output_csv predictions_interview_nli_AudioFlamingo3Local.csv

Or in batch:
    for jsonl in outputs/interview_nli_*.jsonl; do
        alm=$(basename $jsonl | sed 's/interview_nli_//' | sed 's/.jsonl//')
        python convert_to_difficulty_csv.py \
          --input_jsonl "$jsonl" \
          --dataset interview \
          --task nli \
          --alm "$alm" \
          --output_csv "predictions_interview_nli_${alm}.csv"
    done
"""

import json
import argparse
import pandas as pd
from pathlib import Path


def jsonl_to_csv_with_difficulty(
    input_jsonl: str,
    dataset: str,
    task: str,
    alm: str,
    output_csv: str
) -> None:
    """
    Convert JSONL inference output to CSV format with difficulty.
    
    Expects JSONL records with:
    - hypothesis (text)
    - gold (label)
    - pred (prediction)
    - difficulty (easy/medium/hard)
    
    Creates CSV with columns:
    - task, dataset, alm, difficulty, gold, pred
    """
    
    records = []
    
    with open(input_jsonl, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Failed to parse JSON: {e}")
                continue
            
            # Check required fields
            required = {'gold', 'pred', 'difficulty'}
            missing = required - set(data.keys())
            if missing:
                print(f"[WARN] Line {line_num}: Missing fields {missing}, skipping")
                continue
            
            # Normalize labels to lowercase
            gold = str(data.get('gold', '')).lower().strip()
            pred = str(data.get('pred', '')).lower().strip()
            difficulty = str(data.get('difficulty', '')).lower().strip()
            
            # Skip if any field is empty
            if not all([gold, pred, difficulty]):
                print(f"[WARN] Line {line_num}: Empty gold/pred/difficulty, skipping")
                continue
            
            records.append({
                'task': task,
                'dataset': dataset,
                'alm': alm,
                'difficulty': difficulty,
                'gold': gold,
                'pred': pred
            })
    
    if not records:
        print(f"[ERROR] No valid records found in {input_jsonl}")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"[OK] Converted {len(records)} records from {input_jsonl}")
    print(f"[OK] Saved to {output_csv}")
    print(f"\nSummary:")
    print(df['difficulty'].value_counts().sort_index())
    print(f"\nDataFrame head:")
    print(df.head(10))


def main():
    parser = argparse.ArgumentParser(
        description="Convert ALM inference JSONL output to evaluation CSV with difficulty levels"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file from ALM inference"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'interview', 'medical')"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task type (e.g., 'nli', 'consistency')"
    )
    parser.add_argument(
        "--alm",
        type=str,
        required=True,
        help="Audio Language Model name (e.g., 'AudioFlamingo3Local', 'Kimi')"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input_jsonl).exists():
        print(f"[ERROR] Input file not found: {args.input_jsonl}")
        return
    
    jsonl_to_csv_with_difficulty(
        input_jsonl=args.input_jsonl,
        dataset=args.dataset,
        task=args.task,
        alm=args.alm,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
