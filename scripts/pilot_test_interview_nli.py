#!/usr/bin/env python3
"""
Pilot test for Interview NLI prompt validation.

Tests the interview_nli prompt on 3 representative audio samples:
- 08f (easy/medium/hard hypotheses)
- 09m (varied difficulty)
- 506 (simple child speech)

This validates prompt effectiveness before full batch inference.
"""
import json
import sys
from pathlib import Path

# Audio samples selection (representative of different types)
PILOT_SAMPLES = ["08f", "09m", "506"]

def extract_pilot_samples():
    """Extract hypotheses for pilot samples from interview_nli_hypotheses.jsonl"""
    
    input_file = Path("Entailment/interview_nli_hypotheses.jsonl")
    output_file = Path("Entailment/interview_nli_hypotheses_PILOT.jsonl")
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        return False
    
    pilot_data = []
    total_hypotheses = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['audio_id'] in PILOT_SAMPLES:
                pilot_data.append(data)
                total_hypotheses += len(data['hypotheses'])
                print(f"✓ Found {data['audio_id']}: {len(data['hypotheses'])} hypotheses")
    
    if not pilot_data:
        print(f"❌ Error: No pilot samples found")
        return False
    
    # Write pilot JSONL
    with open(output_file, 'w') as f:
        for item in pilot_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Created {output_file}")
    print(f"   - {len(pilot_data)} audio samples")
    print(f"   - {total_hypotheses} total hypotheses to test")
    print(f"   - Expected predictions: {total_hypotheses}")
    
    # Print difficulty breakdown
    print("\n📊 Difficulty Breakdown:")
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    label_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
    
    for item in pilot_data:
        for hyp in item['hypotheses']:
            difficulty_counts[hyp['difficulty']] += 1
            label_counts[hyp['label']] += 1
    
    for diff, count in difficulty_counts.items():
        print(f"   - {diff}: {count}")
    
    print("\n🏷️  Label Distribution:")
    for label, count in label_counts.items():
        print(f"   - {label}: {count}")
    
    return True

def verify_audio_files():
    """Verify that pilot audio files exist"""
    audio_dir = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview")
    
    print("\n🎵 Verifying Audio Files:")
    all_exist = True
    for sample_id in PILOT_SAMPLES:
        audio_file = audio_dir / f"{sample_id}.wav"
        if audio_file.exists():
            size_mb = audio_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ {sample_id}.wav ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {sample_id}.wav NOT FOUND")
            all_exist = False
    
    return all_exist

def print_next_steps():
    """Print instructions for running inference"""
    print("\n" + "="*70)
    print("🚀 NEXT STEPS: Run Pilot Inference")
    print("="*70)
    
    print("\n1️⃣  Update task_config.json (if not already done):")
    print('   Add entry: "interview_nli_pilot" pointing to interview_nli_hypotheses_PILOT.jsonl')
    
    print("\n2️⃣  Option A: Run with existing model (e.g., Kimi):")
    print("   cd /orange/ufdatastudios/c.okocha/Kimi-Audio")
    print("   # Copy inference template and customize for interview_nli")
    print("   python infer_jsonl.py \\")
    print("       --task interview_nli \\")
    print("       --audio_dir /orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview \\")
    print("       --hypotheses_file /orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses_PILOT.jsonl \\")
    print("       --output_file outputs/interview_nli_pilot_results.jsonl")
    
    print("\n   Option B: Quick Python test (manual):")
    print("   # Use the prompt template and manually test with 1-2 samples")
    print("   # See INTERVIEW_NLI_PROMPT_VARIANTS.md for prompt text")
    
    print("\n3️⃣  Evaluate Results:")
    print("   python convert_to_difficulty_csv.py \\")
    print("       --jsonl outputs/Kimi/interview_nli/results/kimi_interview_nli_pilot.jsonl \\")
    print("       --output pilot_results.csv \\")
    print("       --task interview_nli \\")
    print("       --dataset interview \\")
    print("       --alm Kimi")
    
    print("\n   python evaluation_by_difficulty.py \\")
    print("       --csv pilot_results.csv \\")
    print("       --output_csv pilot_metrics.csv")
    
    print("\n4️⃣  Analyze Results:")
    print("   - Check accuracy by difficulty (easy/medium/hard)")
    print("   - Identify error patterns (over-inference, audio misunderstanding, etc.)")
    print("   - Compare against baseline expectations")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("="*70)
    print("🧪 PILOT TEST SETUP: Interview NLI Prompt Validation")
    print("="*70)
    
    # Step 1: Extract pilot samples
    if not extract_pilot_samples():
        sys.exit(1)
    
    # Step 2: Verify audio files
    if not verify_audio_files():
        print("\n⚠️  Warning: Some audio files missing!")
    
    # Step 3: Print next steps
    print_next_steps()
    
    print("\n✅ Pilot test setup complete!")
