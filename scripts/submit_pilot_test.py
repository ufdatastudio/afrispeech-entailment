#!/usr/bin/env python3
"""
Submit SLURM job for pilot interview NLI inference.

This script submits a pilot test to validate the interview_nli prompt
on 3 representative audio samples before running full batch inference.
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("🚀 Submitting Pilot Test for Interview NLI")
    print("="*70)
    
    # Check if pilot JSONL exists
    pilot_jsonl = Path("Entailment/interview_nli_hypotheses_PILOT.jsonl")
    if not pilot_jsonl.exists():
        print(f"\n❌ Error: {pilot_jsonl} not found")
        print("   Run first: python3 pilot_test_interview_nli.py")
        return 1
    
    print(f"\n✅ Pilot JSONL found: {pilot_jsonl}")
    
    # Count hypotheses
    with open(pilot_jsonl, 'r') as f:
        import json
        samples = [json.loads(line) for line in f]
        total_hyp = sum(len(s['hypotheses']) for s in samples)
    
    print(f"   - {len(samples)} audio samples")
    print(f"   - {total_hyp} hypotheses to test")
    
    # SLURM script
    slurm_script = Path("inference/run_pilot_interview_nli.sh")
    if not slurm_script.exists():
        print(f"\n❌ Error: {slurm_script} not found")
        return 1
    
    print(f"\n📄 SLURM script: {slurm_script}")
    
    # Submit job
    print("\n🎯 Submitting job to SLURM...")
    
    try:
        result = subprocess.run(
            ["sbatch", str(slurm_script)],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # Extract job ID
        import re
        job_match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if job_match:
            job_id = job_match.group(1)
            print(f"✅ Job submitted successfully!")
            print(f"   Job ID: {job_id}")
            print(f"\n📊 Monitor progress:")
            print(f"   squeue -u $USER")
            print(f"   tail -f outputs/Kimi/interview_nli_pilot/logs/pilot_{job_id}.out")
            print(f"\n📁 Output location:")
            print(f"   outputs/Kimi/interview_nli_pilot/results/kimi_interview_nli_pilot.jsonl")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error submitting job: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return 1
    except FileNotFoundError:
        print("❌ Error: sbatch command not found")
        print("   Are you on a SLURM cluster?")
        return 1

if __name__ == "__main__":
    sys.exit(main())
