#!/usr/bin/env python3
"""
Fix gold labels in output JSONL files by extracting them from input JSONL.

Usage:
    python fix_gold_labels.py --input_jsonl <input> --output_jsonl <output> --task <task_type>
"""

import json
import argparse
from pathlib import Path


def extract_hypotheses_with_labels(record, task):
    """Extract hypotheses with their gold labels from input JSONL record."""
    hypotheses = []
    
    if "output" in record and isinstance(record["output"], dict):
        output = record["output"]
        
        if task == "nli":
            for key in ["entailment", "contradiction", "neutral"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        hyp_text = str(hyp) if not isinstance(hyp, dict) else hyp.get("hypothesis", "")
                        if hyp_text:
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "consistency":
            for key in ["consistent", "inconsistent"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        hyp_text = str(hyp) if not isinstance(hyp, dict) else hyp.get("hypothesis", "")
                        if hyp_text:
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "plausibility":
            for key in ["plausible", "implausible"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        hyp_text = str(hyp) if not isinstance(hyp, dict) else hyp.get("hypothesis", "")
                        if hyp_text:
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "restraint":
            for key in ["supported", "unsupported"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        hyp_text = str(hyp) if not isinstance(hyp, dict) else hyp.get("hypothesis", "")
                        if hyp_text:
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "accent_drift":
            for key in ["accent_invariant", "accent_sensitive_lures"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        hyp_text = str(hyp) if not isinstance(hyp, dict) else hyp.get("hypothesis", "")
                        if hyp_text:
                            label = "ACCENT_TEST" if key == "accent_sensitive_lures" else "ACCENT_INVARIANT"
                            hypotheses.append((hyp_text.strip(), label))
    
    return hypotheses


def main():
    parser = argparse.ArgumentParser(description="Fix gold labels in output JSONL files")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL with gold labels")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL to fix")
    parser.add_argument("--task", type=str, required=True,
                       choices=["nli", "consistency", "plausibility", "restraint", "accent_drift"],
                       help="Task type")
    
    args = parser.parse_args()
    
    # Read input JSONL to build hypothesis -> gold_label mapping
    print(f"Reading input JSONL: {args.input_jsonl}")
    hyp_to_gold = {}
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                hyps_with_labels = extract_hypotheses_with_labels(record, args.task)
                for hyp_text, gold_label in hyps_with_labels:
                    hyp_to_gold[hyp_text] = gold_label
    
    print(f"Found {len(hyp_to_gold)} hypothesis->gold mappings")
    
    # Read and update output JSONL
    print(f"Reading output JSONL: {args.output_jsonl}")
    updated_lines = []
    matched = 0
    unmatched = 0
    unmatched_examples = []
    
    with open(args.output_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                hypothesis = obj.get("hypothesis", "").strip()
                
                if hypothesis in hyp_to_gold:
                    obj["gold"] = hyp_to_gold[hypothesis]
                    matched += 1
                else:
                    unmatched += 1
                    if len(unmatched_examples) < 5:
                        unmatched_examples.append(hypothesis[:80])
                
                updated_lines.append(json.dumps(obj))
    
    print(f"\nMatched: {matched}, Unmatched: {unmatched}")
    if unmatched_examples:
        print(f"\nFirst few unmatched hypotheses (for debugging):")
        for ex in unmatched_examples:
            print(f"  - {ex}...")
    
    # Write updated output
    print(f"\nWriting updated output to {args.output_jsonl}...")
    with open(args.output_jsonl, 'w') as f:
        for line in updated_lines:
            f.write(line + "\n")
    
    print(f"✓ Updated {matched} entries with gold labels!")


if __name__ == "__main__":
    main()

