#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name ltu-interview_nli
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/logs/infer_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/logs/infer_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200

set -euo pipefail

echo "===== GPU Info ====="
nvidia-smi || true

# Go to LTU directory
cd /orange/ufdatastudios/c.okocha/ltu/src/ltu_as

# Load CUDA
module load cuda/12.8.1 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Activate LTU venv
source /orange/ufdatastudios/c.okocha/ltu/.venv/bin/activate

# Paths
JSONL_PATH="/orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl"
FEATURE_DIR="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/whisper_features"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs"
MODEL_NAME="LTU"
OUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/interview_nli"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
OUT_JSONL="${RESULTS_DIR}/${MODEL_NAME}_interview_nli.jsonl"

# Ensure output dirs exist
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# Model paths
BASE_MODEL="/orange/ufdatastudios/c.okocha/ltu/pretrained_mdls/vicuna_ltuas/"
MODEL_CHECKPOINT="/orange/ufdatastudios/c.okocha/ltu/pretrained_mdls/ltuas_long_noqa_a6.bin"

echo "===== Running LTU-AS inference ====="
echo "Input JSONL: ${JSONL_PATH}"
echo "Features: ${FEATURE_DIR}"
echo "Output: ${OUT_JSONL}"
echo "Base model: ${BASE_MODEL}"
echo "Checkpoint: ${MODEL_CHECKPOINT}"

python infer_interview_nli.py \
    --input_jsonl "${JSONL_PATH}" \
    --feature_dir "${FEATURE_DIR}" \
    --output_jsonl "${OUT_JSONL}" \
    --base_model "${BASE_MODEL}" \
    --model_checkpoint "${MODEL_CHECKPOINT}" \
    --max_new_tokens 512 \
    --temperature 0.1 \
    --resume

echo ""
echo "===== Done ====="
echo "Results in ${RESULTS_DIR}"
echo "Logs in ${LOGS_DIR}"
echo ""
echo "📝 Next Steps:"
echo "   1. Convert to CSV:"
echo "      cd /orange/ufdatastudios/c.okocha/afrispeech-entailment"
echo "      python3 convert_to_difficulty_csv.py \\"
echo "          --jsonl ${OUT_JSONL} \\"
echo "          --output ${OUT_DIR}/interview_nli_results.csv \\"
echo "          --task interview_nli --dataset interview --alm ${MODEL_NAME}"
echo ""
echo "   2. Evaluate by difficulty:"
echo "      python3 evaluation_by_difficulty.py \\"
echo "          --csv ${OUT_DIR}/interview_nli_results.csv \\"
echo "          --output_csv ${OUT_DIR}/interview_nli_metrics.csv"
