#!/usr/bin/env python3
"""
Postprocess AF3 ICL outputs: fix 'pred' field by parsing canonical label from 'pred_raw'.
Works for all variants, datasets, and shots.
"""
import os
import re
import json
from pathlib import Path

def parse_label(text):
    if not text:
        return "UNPARSEABLE"
    text = str(text).strip().upper()
    for label in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
        if re.search(rf"\\b{label}\\b", text):
            return label
    return "UNPARSEABLE"

def fix_file(fp):
    changed = 0
    out_lines = []
    with open(fp, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pred_raw = obj.get("pred_raw")
            new_pred = parse_label(pred_raw)
            if obj.get("pred") != new_pred:
                obj["pred"] = new_pred
                changed += 1
            out_lines.append(json.dumps(obj))
    if changed:
        with open(fp, "w") as f:
            f.write("\n".join(out_lines) + "\n")
    return changed, len(out_lines)

def main():
    root = Path("outputs/ICL_overlay_nli")
    for variant in ["audio_only", "audio_plus_transcript"]:
        af3_dir = root / variant / "af3"
        if not af3_dir.exists():
            continue
        for dataset in af3_dir.iterdir():
            if not dataset.is_dir():
                continue
            for shot_dir in dataset.iterdir():
                pred_file = shot_dir / "predictions.jsonl"
                if pred_file.exists():
                    changed, total = fix_file(pred_file)
                    print(f"{pred_file}: fixed {changed}/{total}")

if __name__ == "__main__":
    main()
