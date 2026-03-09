#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name af2-interview_nli
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo2/interview_nli/logs/infer_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo2/interview_nli/logs/infer_%j.err
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

# Go to AF2 inference directory (required for imports to work)
cd /orange/ufdatastudios/c.okocha/audio-flamingo-2/inference_HF_pretrained

# Create symlinks to the downloaded model if not present
if [ ! -d "safe_ckpt" ] && [ -d "/home/c.okocha/.cache/huggingface/models--nvidia--audio-flamingo-2/snapshots" ]; then
    echo "Creating symlinks to downloaded model..."
    SNAPSHOT_DIR=$(ls -d /home/c.okocha/.cache/huggingface/models--nvidia--audio-flamingo-2/snapshots/* | head -1)
    [ -e "safe_ckpt" ] || ln -s "${SNAPSHOT_DIR}/safe_ckpt" safe_ckpt
    [ -e "clap_ckpt" ] || ln -s "${SNAPSHOT_DIR}/clap_ckpt" clap_ckpt
    echo "Symlinks created"
fi

# Load CUDA toolkit
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Activate AF2 venv (from new audio-flamingo-2 directory)
source /orange/ufdatastudios/c.okocha/audio-flamingo-2/.venv_af2/bin/activate

# Fix NumPy/Numba compatibility (Numba requires NumPy <= 2.0)
NUMPY_FIX_NEEDED="true"
if [[ "$NUMPY_FIX_NEEDED" == "true" ]]; then
    echo "Checking NumPy version for Numba compatibility..."
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0.0.0")
    NUMPY_MAJOR=$(echo $NUMPY_VERSION | cut -d. -f1)
    NUMPY_MINOR=$(echo $NUMPY_VERSION | cut -d. -f2)
    if [ "$NUMPY_MAJOR" -gt 2 ] || ([ "$NUMPY_MAJOR" -eq 2 ] && [ "$NUMPY_MINOR" -gt 0 ]); then
        echo "Downgrading NumPy from $NUMPY_VERSION to <2.1 for Numba compatibility..."
        uv pip install "numpy<2.1" --quiet || pip install "numpy<2.1" --quiet
        echo "NumPy downgraded successfully"
    else
        echo "NumPy version $NUMPY_VERSION is compatible with Numba"
    fi
fi

# Paths
JSONL_PATH="/orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl"
AUDIO_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview"
TASK_NAME="interview_nli"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs"
MODEL="AudioFlamingo2"
OUT_DIR="${OUTPUT_BASE}/${MODEL}/${TASK_NAME}"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
OUTPUT_PREFIX="interview_nli"
OUT_JSONL="${RESULTS_DIR}/${MODEL}_${OUTPUT_PREFIX}.jsonl"
AudioFlamingo2_PATH="nvidia/audio-flamingo-2"
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

echo "===== Running FULL Interview NLI inference (AudioFlamingo 2) ====="
echo "Model: ${AudioFlamingo2_PATH}"
echo "Task: ${TASK}"
echo "Input: ${JSONL_PATH}"
echo "Audio: ${AUDIO_DIR}"
echo "Output: ${OUT_JSONL}"
echo ""
echo "📊 FULL DATASET: 24 audio samples × 9 hypotheses = 216 predictions"
echo ""

# Model will be downloaded to current directory by snapshot_download
MODEL_CHECKPOINT_DIR="."

python3 infer_interview_nli.py \
  --jsonl_path "${JSONL_PATH}" \
  --audio_dir "${AUDIO_DIR}" \
  --task "interview_nli" \
  --out_jsonl "${OUT_JSONL}" \
  --model_checkpoint_dir "${MODEL_CHECKPOINT_DIR}" \
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
echo "          --task interview_nli --dataset interview --alm AudioFlamingo2"
echo ""
echo "   2. Evaluate by difficulty:"
echo "      python3 evaluation_by_difficulty.py \\"
echo "          --csv ${OUT_DIR}/interview_nli_results.csv \\"
echo "          --output_csv ${OUT_DIR}/interview_nli_metrics.csv"
