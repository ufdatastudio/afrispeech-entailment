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
import re
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
    variant: str = None,
    venv_path: str = None
) -> str:
    """Generate SLURM script content for a task."""
    
    # Read template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders (order matters - do specific replacements first)
    script = template.replace("MODEL-PROJECT", project_dir)
    script = script.replace("MODEL-TASK-DATASET", f"{model_name.lower()}-{task_name}")
    
    # Replace MODEL_NAME_PLACEHOLDER first (before any other MODEL replacements)
    # Create a bash-safe variable name (replace dots with underscores for variable names)
    model_name_safe = model_name.replace(".", "_")
    
    # Replace MODEL_PATH_HERE BEFORE replacing MODEL (to avoid MODEL_PATH_HERE -> Kimi_PATH_HERE)
    script = script.replace("MODEL_PATH_HERE", model_path)
    
    # Replace MODEL_NAME_VAR with safe name for bash variable
    script = script.replace("MODEL_NAME_VAR", model_name_safe)
    
    # Replace MODEL_NAME_PLACEHOLDER in the variable value with actual model name (for paths)
    # This happens after MODEL_NAME_VAR replacement to avoid double replacement
    script = script.replace(f'{model_name_safe}="MODEL_NAME_PLACEHOLDER"', f'{model_name_safe}="{model_name}"')
    
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
    
    # Handle model-specific arguments
    import re
    
    # SALMONN uses --cfg_path instead of --model_path
    if model_name == "SALMONN":
        # Replace --model_path with --cfg_path
        cfg_path = f"{project_dir}/configs/decode_config.yaml"
        # Replace in the python command (handle both MODEL_PATH and SALMONN_PATH variable names)
        script = script.replace(f'--model_path "${{MODEL_PATH}}"', f'--cfg_path "{cfg_path}"')
        script = script.replace(f'--model_path "${{{model_name}_PATH}}"', f'--cfg_path "{cfg_path}"')
        # Replace variable definition
        script = script.replace('MODEL_PATH="MODEL_PATH_HERE"', f'CFG_PATH="{cfg_path}"')
        script = script.replace(f'{model_name}_PATH="{project_dir}"', f'CFG_PATH="{cfg_path}"')
        # Replace any remaining MODEL_PATH references
        script = script.replace('${MODEL_PATH}', f'"{cfg_path}"')
        script = script.replace(f'${{{model_name}_PATH}}', f'"{cfg_path}"')
        script = script.replace('echo "Model: ${MODEL_PATH}"', f'echo "Model: SALMONN (config: {cfg_path})"')
    
    # GAMA uses --base_model_path and --checkpoint_path instead of --model_path
    elif model_name == "GAMA":
        # Replace --model_path with --base_model_path and --checkpoint_path
        base_model_path = f"{project_dir}/models/Llama-2-7b-chat-hf-qformer/Llama-2-7b-chat-hf-qformer"
        checkpoint_path = f"{project_dir}/checkpoints/stage5_epoch1/checkpoint-2500/pytorch_model.bin"
        
        # Replace the python command arguments (handle both MODEL_PATH and GAMA_PATH variable names)
        script = script.replace(
            '--model_path "${MODEL_PATH}" \\',
            f'--base_model_path "{base_model_path}" \\\n    --checkpoint_path "{checkpoint_path}" \\'
        )
        script = script.replace(
            f'--model_path "${{{model_name}_PATH}}" \\',
            f'--base_model_path "{base_model_path}" \\\n    --checkpoint_path "{checkpoint_path}" \\'
        )
        
        # Update variable definitions
        script = script.replace('MODEL_PATH="MODEL_PATH_HERE"', f'BASE_MODEL_PATH="{base_model_path}"\nCHECKPOINT_PATH="{checkpoint_path}"')
        script = script.replace(f'{model_name}_PATH="{project_dir}"', f'BASE_MODEL_PATH="{base_model_path}"\nCHECKPOINT_PATH="{checkpoint_path}"')
        # Replace any remaining MODEL_PATH references
        script = script.replace('${MODEL_PATH}', f'"{base_model_path}"')
        script = script.replace(f'${{{model_name}_PATH}}', f'"{base_model_path}"')
        script = script.replace('echo "Model: ${MODEL_PATH}"', f'echo "Model: GAMA (base: {base_model_path}, checkpoint: {checkpoint_path})"')
    
    # CLAP doesn't require --model_path (uses default checkpoint), but needs --cuda flag
    elif model_name == "CLAP":
        # Remove --model_path argument (CLAP uses default checkpoint if not provided)
        script = script.replace('--model_path "${MODEL_PATH}" \\', '')
        script = script.replace(f'--model_path "${{{model_name}_PATH}}" \\', '')
        # Update variable definitions (model_path is optional for CLAP)
        script = script.replace('MODEL_PATH="MODEL_PATH_HERE"', 'MODEL_PATH=""  # Optional: path to CLAP checkpoint')
        script = script.replace(f'{model_name}_PATH="{project_dir}"', 'MODEL_PATH=""  # Optional: path to CLAP checkpoint')
        script = script.replace('echo "Model: ${MODEL_PATH}"', 'echo "Model: CLAP (using default checkpoint)"')
        # Remove --max_new_tokens (not applicable to CLAP)
        script = script.replace('    ARG_OUTPUT_JSONL \\\n    --max_new_tokens 512', '    ARG_OUTPUT_JSONL')
    
    # Add variant parameter if provided (for Audio Flamingo 3)
    elif variant:
        # Add --variant argument to the python command (after --model_path)
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
    
    # Handle custom venv path if provided (for shared venv like Qwen models)
    if venv_path:
        # Replace the venv activation line
        script = script.replace(
            'source .venv/bin/activate',
            f'source {venv_path}/bin/activate'
        )
    # For Audio Flamingo models, ensure .venv path is correct (already in project_dir)
    # The template uses source .venv/bin/activate which should work since we cd to project_dir
    
    # Handle different argument names for different models
    # Qwen models use --input_jsonl and --output_jsonl
    # Audio Flamingo and others use --jsonl_path and --out_jsonl
    if "Qwen" in model_name:
        script = script.replace(
            'ARG_JSONL_PATH',
            f'--input_jsonl "${{JSONL_PATH}}"'
        )
        script = script.replace(
            'ARG_OUTPUT_JSONL',
            f'--output_jsonl "${{OUT_JSONL}}"'
        )
    else:
        script = script.replace(
            'ARG_JSONL_PATH',
            f'--jsonl_path "${{JSONL_PATH}}"'
        )
        script = script.replace(
            'ARG_OUTPUT_JSONL',
            f'--out_jsonl "${{OUT_JSONL}}"'
        )
    
    # Add --cuda flag for CLAP (after argument replacements)
    if model_name == "CLAP":
        # Add --cuda flag before --jsonl_path
        script = script.replace(
            '--jsonl_path "${JSONL_PATH}"',
            '--cuda \\\n    --jsonl_path "${JSONL_PATH}"'
        )
    
    # Fix any remaining MODEL_PATH variable names that have dots (for Qwen2.5-Omni)
    # Replace MODEL_PATH variable assignments with safe variable names
    model_path_var_safe = model_name.replace(".", "_") + "_PATH"
    model_name_escaped = re.escape(model_name)
    # Replace variable assignments like "Qwen2.5Omni_PATH=" with safe name
    script = re.sub(
        model_name_escaped + r'_PATH=',
        model_path_var_safe + '=',
        script
    )
    # Replace variable references like "${Qwen2.5Omni_PATH}" with safe name
    # Use string concatenation to avoid f-string issues with braces
    pattern = r'\$\{' + model_name_escaped + r'_PATH\}'
    replacement = '${' + model_path_var_safe + '}'
    script = re.sub(pattern, replacement, script)
    
    # Remove any trailing backslashes before newlines that might cause issues
    script = re.sub(r' \\\s*\n\s*--', r' \\\n    --', script)
    script = re.sub(r'([^\\])\s*\\\s*\n\s*--', r'\1 \\\n    --', script)
    
    # Validate: Check for common bash syntax errors
    if re.search(r'[A-Za-z0-9_]*\.[A-Za-z0-9_]*=', script):
        # Found variable assignment with dot - this is invalid in bash
        print(f"Warning: Found potential bash syntax error (dot in variable name) for {model_name}")
    
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
    parser.add_argument("--venv_path", type=str, default=None,
                       help="Custom virtual environment path (if different from project_dir/.venv)")
    
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
            args.project_dir, args.template, variant=args.variant,
            venv_path=args.venv_path
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

