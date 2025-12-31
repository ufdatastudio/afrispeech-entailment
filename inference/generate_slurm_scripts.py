#!/usr/bin/env python3
"""
Generate SLURM scripts for all tasks from task_config.json.

Usage:
    python generate_slurm_scripts.py \
        --model_name Kimi \
        --model_path moonshotai/Kimi-Audio-7B-Instruct \
        --project_dir /orange/ufdatastudios/c.okocha/Kimi-Audio \
        --output_dir /orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts
"""
import json
import argparse
from pathlib import Path


def load_task_config(config_path: str) -> dict:
    """Load task configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_slurm_script(
    task_name: str,
    task_config: dict,
    model_name: str,
    model_path: str,
    project_dir: str,
    template_path: str,
    variant: str = None
) -> str:
    """Generate SLURM script content for a task."""
    
    # Read template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders (order matters - do specific replacements first)
    script = template.replace("MODEL-PROJECT", project_dir)
    script = script.replace("MODEL-TASK-DATASET", f"{model_name.lower()}-{task_name}")
    
    # Replace MODEL_NAME_PLACEHOLDER first (before any other MODEL replacements)
    script = script.replace("MODEL_NAME_PLACEHOLDER", model_name)
    
    # Replace MODEL_PATH_HERE BEFORE replacing MODEL (to avoid MODEL_PATH_HERE -> Kimi_PATH_HERE)
    script = script.replace("MODEL_PATH_HERE", model_path)
    
    # Replace other MODEL placeholders
    script = script.replace("MODEL", model_name)
    
    # Task-specific replacements
    script = script.replace("TASK_TYPE", task_config['task'])  # nli, consistency, etc.
    script = script.replace("TASK_NAME", task_name)  # medical_nli, parliament_consistency, etc.
    
    # Set paths directly from config (replace placeholders)
    script = script.replace(
        'JSONL_PATH="JSONL_PATH_PLACEHOLDER"',
        f'JSONL_PATH="{task_config["jsonl_path"]}"'
    )
    script = script.replace(
        'AUDIO_DIR="AUDIO_DIR_PLACEHOLDER"',
        f'AUDIO_DIR="{task_config["audio_dir"]}"'
    )
    
    # Set output paths
    output_prefix = task_config.get('output_prefix', task_name)
    script = script.replace(
        'OUTPUT_PREFIX="OUTPUT_PREFIX"',
        f'OUTPUT_PREFIX="{output_prefix}"'
    )
    
    # Enable NumPy fix for Audio Flamingo models (Numba compatibility)
    if "Flamingo" in model_name:
        script = script.replace('NUMPY_FIX_NEEDED="false"', 'NUMPY_FIX_NEEDED="true"')
    
    # Add variant parameter if provided (for Audio Flamingo 3)
    if variant:
        # Add --variant argument to the python command (after --model_path)
        import re
        # Find the python infer_jsonl.py line and add --variant after --model_path
        # Match the pattern: python infer_jsonl.py \n    --model_path "${MODEL_PATH}" \
        pattern = r'(python infer_jsonl.py\s+\\\s*\n\s+--model_path\s+"[^"]+"\s+\\\s*\n\s+)'
        replacement = f'\\1--variant {variant} \\\n    '
        script = re.sub(pattern, replacement, script)
        
        # If the above didn't match, try a simpler pattern (single line or different format)
        if f'--variant {variant}' not in script:
            # Try matching just after --model_path
            pattern2 = r'(--model_path\s+"[^"]+"\s+\\\s*\n\s+)'
            replacement2 = f'\\1--variant {variant} \\\n    '
            script = re.sub(pattern2, replacement2, script)
    
    # For Audio Flamingo models, ensure .venv path is correct (already in project_dir)
    # The template uses source .venv/bin/activate which should work since we cd to project_dir
    
    return script


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for all tasks")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., Kimi)")
    parser.add_argument("--model_path", type=str, required=True, help="Model path/ID")
    parser.add_argument("--project_dir", type=str, required=True, help="Model project directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for SLURM scripts")
    parser.add_argument("--variant", type=str, default=None, 
                       help="Model variant (e.g., 'base' or 'think' for Audio Flamingo 3)")
    parser.add_argument("--config", type=str, 
                       default="/orange/ufdatastudios/c.okocha/Afro_entailment/inference/task_config.json",
                       help="Task configuration file")
    parser.add_argument("--template", type=str,
                       default="/orange/ufdatastudios/c.okocha/Afro_entailment/inference/templates/run_infer.sh",
                       help="SLURM script template")
    
    args = parser.parse_args()
    
    # Load config
    config = load_task_config(args.config)
    tasks = config['tasks']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate scripts
    print(f"Generating SLURM scripts for {len(tasks)} tasks...")
    for task_name, task_config in tasks.items():
        script_content = generate_slurm_script(
            task_name, task_config, args.model_name, args.model_path,
            args.project_dir, args.template, variant=args.variant
        )
        
        script_path = output_dir / f"run_{task_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        print(f"  Generated: {script_path}")
    
    print(f"\nDone! Generated {len(tasks)} SLURM scripts in {output_dir}")
    print("\nTo submit all jobs:")
    print(f"  cd {output_dir}")
    print(f"  for script in run_*.sh; do sbatch $script; done")


if __name__ == "__main__":
    main()

