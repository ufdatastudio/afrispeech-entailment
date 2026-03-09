#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=af3-icl
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
AF3_ROOT="/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3"
AF3_VENV="${AF3_ROOT}/.venv_af3"
LOG_DIR="${ROOT}/outputs/ICL_overlay_nli/slurm_logs"

mkdir -p "${LOG_DIR}"

module load cuda/12.8.1 || true
module load intel/2020.0.166 ffmpeg/n7.2 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:${AF3_VENV}/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "===== AF3 ICL SLURM START ====="
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date -Iseconds)"
echo "WHICH_PYTHON: $(which python)"
python --version

echo "===== GPU CHECK (nvidia-smi) ====="
nvidia-smi || true

echo "===== GPU CHECK (torch) ====="
python - <<'PY'
import torch
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
PY

cd "${ROOT}"

SHOTS="${SHOTS:-1 3 5 7 10}"
ICL_VARIANTS="${ICL_VARIANTS:-audio_only audio_plus_transcript audio_exemplar_only}"

for variant in ${ICL_VARIANTS}; do
  echo "===== RUNNING AF3 ICL: ${variant} ====="
  MODEL=af3 ICL_VARIANT="${variant}" SHOTS="${SHOTS}" DRY_RUN=0 bash inference/run_nli_icl_sweep_overlay.sh
done

echo "===== AF3 ICL SLURM DONE ====="
echo "DATE: $(date -Iseconds)"
