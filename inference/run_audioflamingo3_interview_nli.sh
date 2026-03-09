#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name af3-interview_nli
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo3/interview_nli/logs/infer_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo3/interview_nli/logs/infer_%j.err
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
echo ""
echo "===== CUDA Devices Available ====="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# Go to model project root
cd /orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3

# Set PYTHONPATH for llava
export PYTHONPATH=$(pwd)

# Load CUDA toolkit
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Load FFmpeg
module load intel/2020.0.166 ffmpeg/n7.2 || true

# Force PyTorch to see CUDA devices properly
export CUDA_LAUNCH_BLOCKING=1

# Activate project venv
if [[ -f ".venv_af3/bin/activate" ]]; then
  source .venv_af3/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "ERROR: No venv found at .venv_af3 or .venv"
  exit 1
fi

# Paths
JSONL_PATH="/orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl"
AUDIO_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview"
TASK_NAME="interview_nli"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs"
MODEL="AudioFlamingo3"
OUT_DIR="${OUTPUT_BASE}/${MODEL}/${TASK_NAME}"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
OUTPUT_PREFIX="interview_nli"
OUT_JSONL="${RESULTS_DIR}/${MODEL}_${OUTPUT_PREFIX}.jsonl"
MODEL_PATH="/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3/audio-flamingo-3"
TASK="interview_nli"

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

echo "===== Running FULL Interview NLI inference (AudioFlamingo 3 Local) ====="
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Input: ${JSONL_PATH}"
echo "Audio: ${AUDIO_DIR}"
echo "Output: ${OUT_JSONL}"
echo ""
echo "📊 FULL DATASET: 24 audio samples × 9 hypotheses = 216 predictions"
echo ""

python infer_jsonl_interview_nli.py \
  --model_path "${MODEL_PATH}" \
  --prompt_variant "caption_reasoning" \
  --jsonl_path "${JSONL_PATH}" \
  --audio_dir "${AUDIO_DIR}" \
  --task "${TASK}" \
  --out_jsonl "${OUT_JSONL}"

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
echo "          --task interview_nli --dataset interview --alm AudioFlamingo3"
echo ""
echo "   2. Evaluate by difficulty:"
echo "      python3 evaluation_by_difficulty.py \\"
echo "          --csv ${OUT_DIR}/interview_nli_results.csv \\"
echo "          --output_csv ${OUT_DIR}/interview_nli_metrics.csv"
