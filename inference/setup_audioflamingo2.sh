#!/bin/bash
# Setup script for Audio Flamingo 2 inference
# This sets up the Audio Flamingo 2 project folder and generates SLURM scripts

set -e

MODEL_NAME="AudioFlamingo2"
MODEL_PATH="nvidia/audio-flamingo-2"
VARIANT="base"
PROJECT_DIR="/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_2"
INFERENCE_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/inference"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/Afro_entailment/outputs"

echo "===== Setting up Audio Flamingo 2 inference ====="
echo "Model: ${MODEL_NAME}"
echo "Model Path: ${MODEL_PATH}"
echo "Project: ${PROJECT_DIR}"
echo "Outputs: ${OUTPUT_BASE}/${MODEL_NAME}/"

# Create project directory if it doesn't exist
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Creating project directory: ${PROJECT_DIR}"
    mkdir -p "${PROJECT_DIR}"
fi

# Copy Audio Flamingo 2 specific inference template
echo "Copying Audio Flamingo 2 inference template..."
if [ -f "${INFERENCE_DIR}/templates/infer_jsonl_audioflamingo2.py" ]; then
    cp "${INFERENCE_DIR}/templates/infer_jsonl_audioflamingo2.py" "${PROJECT_DIR}/infer_jsonl.py"
    echo "✓ Using Audio Flamingo 2 specific template"
else
    echo "Warning: Audio Flamingo 2 template not found, using generic template"
    cp "${INFERENCE_DIR}/templates/infer_jsonl.py" "${PROJECT_DIR}/infer_jsonl.py"
    echo "  You will need to customize it for Audio Flamingo 2"
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
echo "  1. Review and customize ${PROJECT_DIR}/infer_jsonl.py for Audio Flamingo 2"
echo "  2. Review SLURM scripts in ${PROJECT_DIR}/slurm_scripts/"
echo "  3. Submit jobs:"
echo "     cd ${PROJECT_DIR}/slurm_scripts"
echo "     sbatch run_medical_consistency.sh  # Example"
echo "     # Or submit all:"
echo "     for script in run_*.sh; do sbatch \$script; done"
echo ""
echo "Results will be saved to: ${OUTPUT_BASE}/${MODEL_NAME}/TASK_NAME/results/"

