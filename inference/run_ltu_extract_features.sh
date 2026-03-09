#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name ltu_extract_features
#SBATCH --output=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/logs/extract_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/logs/extract_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
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
module load ffmpeg || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Activate LTU venv
source /orange/ufdatastudios/c.okocha/ltu/.venv/bin/activate

# Paths
AUDIO_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview"
FEATURE_DIR="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/whisper_features"
WHISPER_CHECKPOINT="/orange/ufdatastudios/c.okocha/ltu/pretrained_mdls/large-v1.pt"
LOG_DIR="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/LTU/interview_nli/logs"

# Create directories
mkdir -p "${FEATURE_DIR}" "${LOG_DIR}"

echo "===== Extracting Whisper Features ====="
echo "Audio dir: ${AUDIO_DIR}"
echo "Output dir: ${FEATURE_DIR}"
echo "Whisper checkpoint: ${WHISPER_CHECKPOINT}"

python extract_interview_features.py \
    --audio_dir "${AUDIO_DIR}" \
    --output_dir "${FEATURE_DIR}" \
    --whisper_checkpoint "${WHISPER_CHECKPOINT}" \
    --gpu 0

echo ""
echo "===== Feature extraction complete ====="
echo "Features saved to: ${FEATURE_DIR}"
