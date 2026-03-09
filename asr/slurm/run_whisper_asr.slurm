#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=whisper-asr
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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
module load intel/2020.0.166 ffmpeg/n7.2 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]] && [[ -f "/orange/ufdatastudios/c.okocha/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN=$(cat /orange/ufdatastudios/c.okocha/.cache/huggingface/token)
fi

VENV_CANDIDATE=""
if [[ -n "${VENV_PATH:-}" && -d "${VENV_PATH}" ]]; then
  VENV_CANDIDATE="${VENV_PATH}"
elif [[ -d "$ROOT/.venvASR" ]]; then
  VENV_CANDIDATE="$ROOT/.venvASR"
fi

if [[ -z "$VENV_CANDIDATE" ]]; then
  echo "ERROR: No ASR virtual environment found (.venvASR or VENV_PATH)."
  echo "Create: asr/slurm/setup_asr_env.sh"
  exit 1
fi

echo "Using virtualenv: $VENV_CANDIDATE"
source "$VENV_CANDIDATE/bin/activate"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not available in PATH."
  echo "Load ffmpeg module or install ffmpeg in the environment."
  exit 1
fi

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in this environment")
print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

DATASET="${DATASET:-afrispeech200}"
case "$DATASET" in
  afrispeech200)
    INPUT_JSONL="$ROOT/result/Entailment/AfriSpeech200/Llama/nli_single/afrispeech200_nli_top100.jsonl"
    AUDIO_ROOT="$ROOT/Audio/Afrispeech200"
    ;;
  medical)
    INPUT_JSONL="$ROOT/result/Entailment/Medical/Llama/nli/medical_nli.jsonl"
    AUDIO_ROOT="$ROOT/Audio/medical"
    ;;
  general)
    INPUT_JSONL="$ROOT/result/Entailment/AfriSpeechGeneral/Llama/consistency/afrispeech_general_consistency.jsonl"
    AUDIO_ROOT="$ROOT/Audio/general"
    ;;
  afrinames)
    INPUT_JSONL="$ROOT/result/Entailment/AfriNames/Llama/restraint/afri_names_restraint_top100.jsonl"
    AUDIO_ROOT="$ROOT/Audio/Afrinames"
    ;;
  *)
    echo "ERROR: Unsupported DATASET=$DATASET"
    exit 1
    ;;
esac

MODEL_ID="${WHISPER_MODEL_ID:-openai/whisper-large-v3}"
LIMIT_ARG=""
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARG="--limit ${LIMIT}"; fi

echo "Running Whisper ASR | dataset=${DATASET} | model=${MODEL_ID}"

srun -N 1 --gpus 1 --cpus-per-task 8 \
  python3 asr/run_whisper_asr.py \
    --input_jsonl "$INPUT_JSONL" \
    --audio_col file_name \
    --audio_root "$AUDIO_ROOT" \
    --output_jsonl "$ROOT/outputs/asr/simple/${DATASET}_whisper_asr.jsonl" \
    --simple_jsonl \
    --model_id "$MODEL_ID" \
    --language english \
    --chunk_length_s "${CHUNK_LENGTH_S:-30}" \
    $LIMIT_ARG

echo "Done: outputs/asr/simple/${DATASET}_whisper_asr.jsonl"