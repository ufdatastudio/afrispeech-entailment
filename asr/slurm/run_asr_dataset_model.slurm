#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=asr-transcribe
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/asr/slurm_logs/%x_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/asr/slurm_logs/%x_%j.err
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
cd "$ROOT"
mkdir -p "$ROOT/outputs/asr/slurm_logs" "$ROOT/outputs/asr"

echo "===== GPU Info ====="
nvidia-smi || true
echo ""
echo "===== CUDA Devices Available ====="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

module load cuda/12.8.1 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face token fallback
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]] && [[ -f "/orange/ufdatastudios/c.okocha/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN=$(cat /orange/ufdatastudios/c.okocha/.cache/huggingface/token)
fi

VENV_CANDIDATE=""
if [[ -n "${VENV_PATH:-}" && -d "${VENV_PATH}" ]]; then
  VENV_CANDIDATE="${VENV_PATH}"
elif [[ -d "$ROOT/.venvASR" ]]; then
  VENV_CANDIDATE="$ROOT/.venvASR"
elif [[ -d "$ROOT/.venv" ]]; then
  VENV_CANDIDATE="$ROOT/.venv"
fi

if [[ -z "$VENV_CANDIDATE" ]]; then
  echo "ERROR: No virtual environment found."
  echo "Create one at $ROOT/.venvASR (recommended) or set VENV_PATH when submitting."
  exit 1
fi

echo "Using virtualenv: $VENV_CANDIDATE"
source "$VENV_CANDIDATE/bin/activate"

echo "===== Python/CUDA Check ====="
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

if ! python3 - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "ERROR: CUDA not available inside job."
  echo "Check partition/GPU allocation, CUDA module, and virtual environment torch build."
  exit 1
fi

if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "ERROR: MODEL and DATASET must be provided via sbatch --export"
  echo "Example: sbatch --export=ALL,MODEL=whisper,DATASET=afrispeech200 asr/slurm/run_asr_dataset_model.slurm"
  exit 1
fi

case "$DATASET" in
  afrispeech200)
    INPUT_CSV="$ROOT/Entailment/metadata_afrispeech-200.csv"
    AUDIO_COL="file_name"
    AUDIO_ROOT="$ROOT/Audio/Afrispeech200"
    ;;
  medical)
    INPUT_CSV="$ROOT/Entailment/metadata_medical.csv"
    AUDIO_COL="file_name"
    AUDIO_ROOT="$ROOT/Audio/medical"
    ;;
  general)
    INPUT_CSV="$ROOT/Entailment/metadata_afrispeech-general.csv"
    AUDIO_COL="file_name"
    AUDIO_ROOT="$ROOT/Audio/general"
    ;;
  afrinames)
    INPUT_CSV="$ROOT/Entailment/metadata_afri-names.csv"
    AUDIO_COL="file_name"
    AUDIO_ROOT="$ROOT/Audio/Afrinames"
    ;;
  *)
    echo "ERROR: Unsupported DATASET=$DATASET"
    exit 1
    ;;
esac

LIMIT_ARG=""
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

COMMON_ARGS=(
  --input_csv "$INPUT_CSV"
  --audio_col "$AUDIO_COL"
  --audio_root "$AUDIO_ROOT"
  --output_csv "$ROOT/outputs/asr/${DATASET}_${MODEL}.csv"
  --output_jsonl "$ROOT/outputs/asr/${DATASET}_${MODEL}.jsonl"
)

echo "===== ASR Job ====="
echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "INPUT_CSV=$INPUT_CSV"
echo "AUDIO_ROOT=$AUDIO_ROOT"

case "$MODEL" in
  whisper)
    MODEL_ID="${WHISPER_MODEL_ID:-openai/whisper-large-v3}"
    srun -N 1 --gpus 1 --cpus-per-task 8 \
      python3 asr/run_whisper_asr.py \
      "${COMMON_ARGS[@]}" \
      --model_id "$MODEL_ID" \
      --language english \
      --chunk_length_s "${CHUNK_LENGTH_S:-30}" \
      $LIMIT_ARG
    ;;
  parakeet)
    MODEL_ID="${PARAKEET_MODEL_ID:-nvidia/parakeet-tdt-0.6b-v2}"
    srun -N 1 --gpus 1 --cpus-per-task 8 \
      python3 asr/run_parakeet_asr_template.py \
      "${COMMON_ARGS[@]}" \
      --model_id "$MODEL_ID" \
      --language en \
      $LIMIT_ARG
    ;;
  granite)
    MODEL_ID="${GRANITE_MODEL_ID:-ibm-granite/granite-speech-3.3-2b}"
    srun -N 1 --gpus 1 --cpus-per-task 8 \
      python3 asr/run_granite_asr_template.py \
      "${COMMON_ARGS[@]}" \
      --model_id "$MODEL_ID" \
      --language english \
      --chunk_length_s "${CHUNK_LENGTH_S:-30}" \
      --max_new_tokens "${MAX_NEW_TOKENS:-1000}" \
      $LIMIT_ARG
    ;;
  *)
    echo "ERROR: Unsupported MODEL=$MODEL. Use whisper|parakeet|granite"
    exit 1
    ;;
esac

echo "Done: outputs/asr/${DATASET}_${MODEL}.csv"
