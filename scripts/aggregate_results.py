import glob
import json
import pandas as pd
import os
import re

def get_dataset_name(path):
    if "general" in path:
        return "AfriSpeech-General"
    if "medical" in path:
        return "Medical"
    if "afrispeech200" in path:
        return "AfriSpeech-200"
    return "Unknown"

def get_variant_name(path):
    if "original" in path:
        return "AudioFlamingo3_original"
    if "v2_explicit" in path:
        return "AudioFlamingo3_v2_explicit"
    if "v3_simple" in path:
        return "AudioFlamingo3_v3_simple"
    return "AudioFlamingo3"

base_dir = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo3Local"
output_csv = os.path.join(base_dir, "predictions_all_af3.csv")

files = glob.glob(os.path.join(base_dir, "**/*.jsonl"), recursive=True)

records = []

print(f"Found {len(files)} JSONL files.")

for f in files:
    dataset = get_dataset_name(f)
    alm = get_variant_name(f)
    task = "consistency" # All files seem to be consistency task based on previous steps
    
    # Verify task from filename if needed, but we know we ran consistency jobs.
    
    with open(f) as fin:
        for line in fin:
            try:
                data = json.loads(line)
                
                # Use pred_raw as requested by user, but normalize to uppercase
                # pred_raw might be None or not string, handle safely
                pred_raw = data.get("pred_raw", "")
                if pred_raw is None: 
                    pred_v = ""
                else:
                    pred_v = str(pred_raw).strip().upper()
                
                # If pred_raw is empty, fallback to pred? User said "if not you can use Predraw".
                # Assuming pred_raw is present.
                
                gold = data.get("gold", "").strip().upper()
                
                records.append({
                    "dataset": dataset,
                    "task": task,
                    "alm": alm,
                    "gold": gold,
                    "pred": pred_v,
                    "file": f
                })
            except Exception as e:
                print(f"Error reading line in {f}: {e}")

df = pd.DataFrame(records)
print(f"Aggregated {len(df)} records.")
print(df.head())

df.to_csv(output_csv, index=False)
print(f"Saved aggregated predictions to {output_csv}")
