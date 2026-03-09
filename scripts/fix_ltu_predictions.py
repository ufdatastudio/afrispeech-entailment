#!/usr/bin/env python3
"""
Fix LTU predictions by properly extracting labels from pred_raw sentences.
"""
import json
import re

def extract_label_from_text(text: str) -> str:
    """Extract NLI label from LTU's verbose output."""
    if not text:
        return "UNPARSEABLE"
    
    text_lower = text.lower().strip()
    
    # Common patterns in LTU outputs
    if any(phrase in text_lower for phrase in [
        "does not contain information that would entail or contradict",
        "does not contain any information that would contradict or entail",
        "neutral and does not seem to be contradicted",
        "does not provide any evidence to support or refute",
        "cannot be determined from the audio",
        "no clear indication",
        "neither supports nor contradicts"
    ]):
        return "NEUTRAL"
    
    # Look for explicit entailment/contradiction mentions
    if any(word in text_lower for word in ["entail", "support", "confirm", "agree"]):
        # But check if it's in a negative context
        if any(neg in text_lower for neg in ["does not entail", "not entail", "no entail", "cannot entail"]):
            return "NEUTRAL"
        return "ENTAILMENT"
    
    if any(word in text_lower for word in ["contradict", "refute", "disagree", "oppose"]):
        # But check if it's in a negative context  
        if any(neg in text_lower for neg in ["does not contradict", "not contradict", "no contradict", "cannot contradict"]):
            return "NEUTRAL"
        return "CONTRADICTION"
    
    # Fallback: look for explicit labels
    text_upper = text.upper()
    if "ENTAILMENT" in text_upper:
        return "ENTAILMENT"
    elif "CONTRADICTION" in text_upper:
        return "CONTRADICTION" 
    elif "NEUTRAL" in text_upper:
        return "NEUTRAL"
    
    return "UNPARSEABLE"

def main():
    input_file = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/results/LTU_interview_nli_sampling.jsonl"
    output_file = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/results/LTU_interview_nli.jsonl"
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            total_count += 1
            
            # Extract better prediction
            old_pred = data.get('pred', '')
            new_pred = extract_label_from_text(data.get('pred_raw', ''))
            
            if old_pred != new_pred:
                fixed_count += 1
                print(f"Fixed: '{old_pred}' -> '{new_pred}' | Raw: {data.get('pred_raw', '')[:80]}...")
            
            # Update prediction
            data['pred'] = new_pred
            
            f_out.write(json.dumps(data) + "\n")
    
    print(f"\nFixed {fixed_count}/{total_count} predictions")
    
    # Show new distribution
    print("\n=== New Prediction Distribution ===")
    with open(output_file, 'r') as f:
        preds = [json.loads(line)['pred'] for line in f]
    
    from collections import Counter
    for pred, count in Counter(preds).most_common():
        print(f"{pred:>15}: {count:>3}")

if __name__ == "__main__":
    main()