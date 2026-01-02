#!/bin/bash
# Setup script for GAMA inference
# This sets up the GAMA project folder and generates SLURM scripts

set -e

MODEL_NAME="GAMA"
PROJECT_DIR="/orange/ufdatastudios/c.okocha/GAMA"
INFERENCE_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/inference"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/Afro_entailment/outputs"

echo "===== Setting up GAMA inference ====="
echo "Model: ${MODEL_NAME}"
echo "Project: ${PROJECT_DIR}"
echo "Outputs: ${OUTPUT_BASE}/${MODEL_NAME}/"

# Ensure project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Error: Project directory ${PROJECT_DIR} does not exist"
    exit 1
fi

# Copy GAMA-specific inference template
echo "Copying GAMA inference template..."
cp "${INFERENCE_DIR}/templates/infer_jsonl_gama.py" "${PROJECT_DIR}/infer_jsonl.py"
echo "✓ Template copied"

# Create output directories structure
echo "Creating output directories..."
mkdir -p "${OUTPUT_BASE}/${MODEL_NAME}"

# Generate SLURM scripts
echo "Generating SLURM scripts..."
python3 "${INFERENCE_DIR}/generate_slurm_scripts.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${PROJECT_DIR}" \
    --project_dir "${PROJECT_DIR}" \
    --output_dir "${PROJECT_DIR}/slurm_scripts"

echo ""
echo "===== Setup complete! ====="
echo ""
echo "Next steps:"
echo "  1. Review ${PROJECT_DIR}/infer_jsonl.py"
echo "  2. Review SLURM scripts in ${PROJECT_DIR}/slurm_scripts/"
echo "  3. Update SLURM scripts to set BASE_MODEL_PATH and CHECKPOINT_PATH"
echo "  4. Submit jobs:"
echo "     cd ${PROJECT_DIR}/slurm_scripts"
echo "     sbatch run_medical_consistency.sh  # Example"
echo ""
echo "Results will be saved to: ${OUTPUT_BASE}/${MODEL_NAME}/TASK_NAME/results/"

