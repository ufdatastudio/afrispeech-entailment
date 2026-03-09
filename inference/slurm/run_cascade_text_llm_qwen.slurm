#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name cascade-qwen
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/cascade_asr_text_llm/slurm_logs/cascade_qwen_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/cascade_asr_text_llm/slurm_logs/cascade_qwen_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200

set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
CHILD_ROOT="/orange/ufdatastudios/c.okocha/child__speech_analysis"
LOG_DIR="${ROOT}/outputs/cascade_asr_text_llm/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "===== GPU Info ====="
nvidia-smi || true

module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export HF_HOME="${CHILD_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${CHILD_ROOT}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]] && [[ -f "/orange/ufdatastudios/c.okocha/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN=$(cat /orange/ufdatastudios/c.okocha/.cache/huggingface/token)
fi

export PYTHON_BIN="${PYTHON_BIN:-${CHILD_ROOT}/.venv/bin/python}"
export ASR_MODELS="${ASR_MODELS:-whisper granite}"
export TEXT_MODELS="qwen"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export RESUME="${RESUME:-1}"
export DRY_RUN=0

echo "===== Running cascade baseline ====="
echo "TEXT_MODELS=${TEXT_MODELS}"
echo "ASR_MODELS=${ASR_MODELS}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

cd "${ROOT}"
srun -N 1 --gpus 1 --cpus-per-task 8 bash "${ROOT}/inference/run_cascade_asr_text_llm.sh"

echo "===== Done: cascade-qwen ====="
