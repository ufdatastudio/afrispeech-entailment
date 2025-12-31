#!/bin/bash
# Setup script for Kimi model inference
# This sets up the Kimi project folder and generates SLURM scripts

set -e

MODEL_NAME="Kimi"
MODEL_PATH="moonshotai/Kimi-Audio-7B-Instruct"
PROJECT_DIR="/orange/ufdatastudios/c.okocha/Kimi-Audio"
INFERENCE_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/inference"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/Afro_entailment/outputs"

echo "===== Setting up Kimi inference ====="
echo "Model: ${MODEL_NAME}"
echo "Project: ${PROJECT_DIR}"
echo "Outputs: ${OUTPUT_BASE}/${MODEL_NAME}/"

# Ensure project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Error: Project directory ${PROJECT_DIR} does not exist"
    exit 1
fi

# Copy inference script if it doesn't exist
if [ ! -f "${PROJECT_DIR}/infer_jsonl.py" ]; then
    echo "Copying infer_jsonl.py to ${PROJECT_DIR}..."
    cp "${INFERENCE_DIR}/templates/infer_jsonl.py" "${PROJECT_DIR}/infer_jsonl.py"
    echo "  Note: You need to customize init_model() and generate_with_model() functions"
    echo "  See: ${INFERENCE_DIR}/templates/infer_jsonl_kimi_example.py"
else
    echo "infer_jsonl.py already exists in ${PROJECT_DIR}"
fi

# Create output directories structure
echo "Creating output directories..."
mkdir -p "${OUTPUT_BASE}/${MODEL_NAME}"

# Generate SLURM scripts
echo "Generating SLURM scripts..."
python3 "${INFERENCE_DIR}/generate_slurm_scripts.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --project_dir "${PROJECT_DIR}" \
    --output_dir "${PROJECT_DIR}/slurm_scripts"

echo ""
echo "===== Setup complete! ====="
echo ""
echo "Next steps:"
echo "  1. Customize ${PROJECT_DIR}/infer_jsonl.py for Kimi (if not done)"
echo "  2. Review SLURM scripts in ${PROJECT_DIR}/slurm_scripts/"
echo "  3. Submit jobs:"
echo "     cd ${PROJECT_DIR}/slurm_scripts"
echo "     sbatch run_medical_consistency.sh  # Example"
echo "     # Or submit all:"
echo "     for script in run_*.sh; do sbatch \$script; done"
echo ""
echo "Results will be saved to: ${OUTPUT_BASE}/${MODEL_NAME}/TASK_NAME/results/"

