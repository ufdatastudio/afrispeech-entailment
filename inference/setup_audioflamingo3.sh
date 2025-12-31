#!/bin/bash
# Setup script for Audio Flamingo 3 (base) inference
# This sets up the Audio Flamingo 3 project folder and generates SLURM scripts

set -e

MODEL_NAME="AudioFlamingo3"
MODEL_PATH="nvidia/audio-flamingo-3-hf"
VARIANT="base"
PROJECT_DIR="/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3"
INFERENCE_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/inference"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/Afro_entailment/outputs"

echo "===== Setting up Audio Flamingo 3 (base) inference ====="
echo "Model: ${MODEL_NAME}"
echo "Variant: ${VARIANT}"
echo "Project: ${PROJECT_DIR}"
echo "Outputs: ${OUTPUT_BASE}/${MODEL_NAME}/"

# Ensure project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Error: Project directory ${PROJECT_DIR} does not exist"
    exit 1
fi

# Check if infer_jsonl.py exists (should already be there)
if [ ! -f "${PROJECT_DIR}/infer_jsonl.py" ]; then
    echo "Warning: infer_jsonl.py not found in ${PROJECT_DIR}"
    echo "  Please ensure the script is in place"
else
    echo "✓ infer_jsonl.py found in ${PROJECT_DIR}"
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
    --output_dir "${PROJECT_DIR}/slurm_scripts" \
    --variant "${VARIANT}"

echo ""
echo "===== Setup complete! ====="
echo ""
echo "Next steps:"
echo "  1. Review SLURM scripts in ${PROJECT_DIR}/slurm_scripts/"
echo "  2. Submit jobs:"
echo "     cd ${PROJECT_DIR}/slurm_scripts"
echo "     sbatch run_medical_consistency.sh  # Example"
echo "     # Or submit all:"
echo "     for script in run_*.sh; do sbatch \$script; done"
echo ""
echo "Results will be saved to: ${OUTPUT_BASE}/${MODEL_NAME}/TASK_NAME/results/"

