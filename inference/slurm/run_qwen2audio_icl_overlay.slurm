#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=qwen2audio-icl
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/ICL_overlay_nli/slurm_logs/%x_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/ICL_overlay_nli/slurm_logs/%x_%j.err
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=24:00:00

set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
QWEN_VENV_PRIMARY="/orange/ufdatastudios/c.okocha/AfroBust/.venv"
QWEN_VENV_FALLBACK="/orange/ufdatastudios/c.okocha/child__speech_analysis/.venv"
QWEN_VENV="${QWEN_VENV:-}"
LOG_DIR="${ROOT}/outputs/ICL_overlay_nli/slurm_logs"

mkdir -p "${LOG_DIR}"

module load cuda/12.8.1 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

if [[ -z "${QWEN_VENV}" ]]; then
    if "${QWEN_VENV_PRIMARY}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
assert hasattr(torch, 'cuda')
PY
    then
        QWEN_VENV="${QWEN_VENV_PRIMARY}"
    else
        QWEN_VENV="${QWEN_VENV_FALLBACK}"
    fi
fi

export PATH=$CUDA_HOME/bin:${QWEN_VENV}/bin:$PATH

echo "===== Qwen2Audio ICL SLURM START ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date -Iseconds)"
echo "QWEN_VENV: ${QWEN_VENV}"
echo "WHICH_PYTHON: $(which python)"
python --version

echo "===== GPU CHECK (nvidia-smi) ====="
nvidia-smi || true

echo "===== GPU CHECK (torch) ====="
python - <<'PY'
import torch
print('torch_version=', getattr(torch, '__version__', 'unknown'))
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
PY

cd "${ROOT}"

SHOTS="${SHOTS:-1 3 5 7 10}"
ICL_VARIANTS="${ICL_VARIANTS:-audio_only audio_plus_transcript audio_exemplar_only}"

for variant in ${ICL_VARIANTS}; do
  echo "===== RUNNING Qwen2Audio ICL: ${variant} ====="
  MODEL=qwen2audio ICL_VARIANT="${variant}" SHOTS="${SHOTS}" DRY_RUN=0 bash inference/run_nli_icl_sweep_overlay.sh
done

echo "===== Qwen2Audio ICL SLURM DONE ====="
echo "DATE: $(date -Iseconds)"
