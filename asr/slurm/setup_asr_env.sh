#!/bin/bash
set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
VENV_PATH_DEFAULT="$ROOT/.venvASR"
VENV_PATH="${1:-$VENV_PATH_DEFAULT}"

cd "$ROOT"

python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip wheel "setuptools<81"

# CUDA-enabled torch stack for CUDA 12.8 (B200-compatible)
# Use explicit +cu128 wheels to avoid accidentally pulling CPU-only builds.
python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128"

# Core ASR dependencies (Whisper + Granite)
python -m pip install transformers datasets accelerate pandas tqdm soundfile librosa sentencepiece huggingface_hub safetensors pydub

# NVIDIA Parakeet dependency
python -m pip install Cython
python -m pip install --no-build-isolation youtokentome
python -m pip install --no-build-isolation "nemo_toolkit[asr]"

echo "ASR virtualenv ready at: $VENV_PATH"
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda runtime:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    idx = torch.cuda.current_device()
    print('device name:', torch.cuda.get_device_name(idx))
    print('device capability:', torch.cuda.get_device_capability(idx))
PY
