#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name qwen2audioinstruct-interview_nli
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Qwen2AudioInstruct/interview_nli/logs/infer_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Qwen2AudioInstruct/interview_nli/logs/infer_%j.err
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

# Go to model project root
cd /orange/ufdatastudios/c.okocha/AfroBust/models/Qwen/Qwen2-Audio-7B-Instruct

# Load CUDA toolkit
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Activate project venv
source /orange/ufdatastudios/c.okocha/AfroBust/.venv/bin/activate

# Paths
JSONL_PATH="/orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl"
AUDIO_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview"
TASK="interview_nli"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs"
MODEL_NAME="Qwen2AudioInstruct"
OUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/interview_nli"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
OUT_JSONL="${RESULTS_DIR}/${MODEL_NAME}_interview_nli.jsonl"
MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"

# Ensure output dirs exist
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# Use /orange for model caches
BASE_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face token
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]] && [[ -f "/orange/ufdatastudios/c.okocha/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN=$(cat /orange/ufdatastudios/c.okocha/.cache/huggingface/token)
fi

echo "===== Running inference ====="
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Input: ${JSONL_PATH}"
echo "Audio: ${AUDIO_DIR}"
echo "Output: ${OUT_JSONL}"

srun -N 1 --gpus 1 --cpus-per-task 8 \
  python infer_jsonl.py \
    --model_path "${MODEL_PATH}" \
    --input_jsonl "${JSONL_PATH}" \
    --audio_dir "${AUDIO_DIR}" \
    --task "${TASK}" \
    --output_jsonl "${OUT_JSONL}" \
    --max_new_tokens 512

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
