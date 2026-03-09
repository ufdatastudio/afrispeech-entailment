#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name interview-cascade
#SBATCH --output=/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/cascade_entailment/slurm_logs/%x_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/cascade_entailment/slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=8
#SBATCH --partition hpg-b200

set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
CHILD_ROOT="/orange/ufdatastudios/c.okocha/child__speech_analysis"
OUT_ROOT="${CHILD_ROOT}/results/Interview/cascade_entailment"
LOG_DIR="${OUT_ROOT}/slurm_logs"

TASK_JSONL="${OUT_ROOT}/interview_nli_for_cascade.jsonl"
WHISPER_ASR="${OUT_ROOT}/interview_whisper_simple.jsonl"
GRANITE_ASR="${OUT_ROOT}/interview_granite_simple.jsonl"

PYTHON_BIN="${PYTHON_BIN:-${CHILD_ROOT}/.venv/bin/python}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RESUME="${RESUME:-1}"

TEXT_MODEL="${TEXT_MODEL:-llama}"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"

echo "===== Interview cascade start ====="
echo "TEXT_MODEL=${TEXT_MODEL}"
echo "MODEL_ID=${MODEL_ID}"

module load cuda/12.8.1 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export HF_HOME="${CHILD_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${CHILD_ROOT}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]] && [[ -f "/orange/ufdatastudios/c.okocha/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN=$(cat /orange/ufdatastudios/c.okocha/.cache/huggingface/token)
fi

resume_flag=""
if [[ "${RESUME}" == "1" ]]; then
  resume_flag="--resume"
fi

for asr in whisper granite; do
  if [[ "${asr}" == "whisper" ]]; then
    asr_jsonl="${WHISPER_ASR}"
  else
    asr_jsonl="${GRANITE_ASR}"
  fi

  out_jsonl="${OUT_ROOT}/${asr}/${TEXT_MODEL}/interview_nli.jsonl"
  mkdir -p "$(dirname "${out_jsonl}")"

  echo "=== Running ASR=${asr} MODEL=${TEXT_MODEL} ==="
  srun -N 1 --gpus 1 --cpus-per-task 8 "${PYTHON_BIN}" "${ROOT}/inference/run_text_llm_cascade.py" \
    --model_id "${MODEL_ID}" \
    --task "nli" \
    --task_jsonl "${TASK_JSONL}" \
    --asr_jsonl "${asr_jsonl}" \
    --output_jsonl "${out_jsonl}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    ${resume_flag}
done

echo "===== Interview cascade done ====="
