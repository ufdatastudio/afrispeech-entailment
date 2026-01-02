#!/bin/bash
# Setup script for CLAP model inference
# This sets up the CLAP project folder and generates SLURM scripts

set -e

MODEL_NAME="CLAP"
MODEL_PATH=""  # CLAP uses default checkpoint, can be overridden
PROJECT_DIR="/orange/ufdatastudios/c.okocha/CLAP"
INFERENCE_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/inference"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/Afro_entailment/outputs"

echo "===== Setting up CLAP inference ====="
echo "Model: ${MODEL_NAME}"
echo "Project: ${PROJECT_DIR}"
echo "Outputs: ${OUTPUT_BASE}/${MODEL_NAME}/"

# Ensure project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Creating project directory ${PROJECT_DIR}..."
    mkdir -p "${PROJECT_DIR}"
fi

# Copy CLAP-specific inference script
echo "Copying CLAP inference script..."
cp "${INFERENCE_DIR}/templates/infer_jsonl_clap.py" "${PROJECT_DIR}/infer_jsonl.py"
chmod +x "${PROJECT_DIR}/infer_jsonl.py"
echo "✓ Script copied"

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
echo "  1. Set up virtual environment in ${PROJECT_DIR}/.venv (if not done)"
echo "  2. Install dependencies:"
echo "     source ${PROJECT_DIR}/.venv/bin/activate"
echo "     pip install laion-clap librosa numpy"
echo "  3. Review SLURM scripts in ${PROJECT_DIR}/slurm_scripts/"
echo "  4. Submit jobs:"
echo "     cd ${PROJECT_DIR}/slurm_scripts"
echo "     sbatch run_medical_nli.sh  # Example"
echo "     # Or submit all:"
echo "     for script in run_*.sh; do sbatch \$script; done"
echo ""
echo "Results will be saved to: ${OUTPUT_BASE}/${MODEL_NAME}/TASK_NAME/results/"

