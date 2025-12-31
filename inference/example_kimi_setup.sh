#!/bin/bash
# Example setup script for Kimi model
# Run this from the Kimi-Audio project directory

set -e

MODEL_NAME="Kimi"
MODEL_PATH="moonshotai/Kimi-Audio-7B-Instruct"
PROJECT_DIR="/orange/ufdatastudios/c.okocha/Kimi-Audio"
INFERENCE_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/inference"

echo "Setting up Kimi inference..."

# Copy inference script
cp "${INFERENCE_DIR}/templates/infer_jsonl.py" "${PROJECT_DIR}/infer_jsonl.py"

echo ""
echo "Next step: Customize infer_jsonl.py"
echo "  See ${INFERENCE_DIR}/templates/infer_jsonl_kimi_example.py for Kimi-specific functions"
echo "  Replace init_model() and generate_with_model() functions in infer_jsonl.py"

# Generate SLURM scripts
python3 "${INFERENCE_DIR}/generate_slurm_scripts.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --project_dir "${PROJECT_DIR}" \
    --output_dir "${PROJECT_DIR}/slurm_scripts"

echo "Done! Next steps:"
echo "  1. Review and customize infer_jsonl.py if needed"
echo "  2. Review generated SLURM scripts in slurm_scripts/"
echo "  3. Submit jobs: cd slurm_scripts && for script in run_*.sh; do sbatch \$script; done"

