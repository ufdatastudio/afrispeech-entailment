#!/usr/bin/env python3
"""
Add interview_nli prompt to Kimi's infer_jsonl.py
"""
import re
import sys
from pathlib import Path
from datetime import datetime

def main():
    kimi_script = Path("/orange/ufdatastudios/c.okocha/Kimi-Audio/infer_jsonl.py")
    
    if not kimi_script.exists():
        print(f"❌ Error: {kimi_script} not found")
        return 1
    
    print(f"Reading {kimi_script}...")
    with open(kimi_script, 'r') as f:
        content = f.read()
    
    # Check if already has interview_nli
    if 'interview_nli' in content:
        print("✅ interview_nli prompt already present")
        return 0
    
    print("Adding interview_nli prompt...")
    
    # Create backup
    backup_path = kimi_script.with_suffix(f'.py.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"   Backup created: {backup_path}")
    
    # Find the PROMPTS dictionary and add interview_nli
    interview_prompt = '''    "interview_nli": """You are evaluating an audio interview about stuttering and speech experiences.

Listen to what the speaker says in the audio, then determine the relationship between the audio and the following statement.

Classification:
- ENTAILMENT: The statement is supported by what is said in the audio
- CONTRADICTION: The statement contradicts what is said in the audio
- NEUTRAL: The statement is neither supported nor contradicted by the audio

Important:
- Base your answer ONLY on the audio content
- Do not make assumptions beyond what is explicitly stated
- Consider what the speaker actually says about their experiences

Respond with exactly one label:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}""",

'''
    
    # Find the accent_drift entry and add interview_nli after it
    # Look for the pattern: "accent_drift": """...""",
    # and insert interview_nli before the closing brace of PROMPTS
    
    # Try to find the end of PROMPTS dictionary
    prompts_match = re.search(r'(PROMPTS\s*=\s*\{.*?)(\n\})', content, re.DOTALL)
    
    if not prompts_match:
        print("❌ Could not find PROMPTS dictionary")
        return 1
    
    # Insert interview_nli before the closing brace
    new_content = content[:prompts_match.end(1)] + '\n' + interview_prompt + content[prompts_match.start(2):]
    
    # Also add to normalize_label function
    nli_pattern = r'(if task == "nli":.*?match = _match_label\(t, \("ENTAILMENT", "CONTRADICTION", "NEUTRAL"\)\))'
    nli_replacement = r'\1\n    elif task == "interview_nli":\n        match = _match_label(t, ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"))'
    
    new_content = re.sub(nli_pattern, nli_replacement, new_content)
    
    # Write updated content
    with open(kimi_script, 'w') as f:
        f.write(new_content)
    
    print("✅ Added interview_nli prompt to Kimi's infer_jsonl.py")
    print(f"   Original backed up to: {backup_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
