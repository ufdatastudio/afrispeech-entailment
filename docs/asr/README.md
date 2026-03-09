# ASR Ground-Truth Workflow for ACL Rebuttal

This folder provides a consistent ASR pipeline to generate text transcripts that can be used as a control condition for ALM error analysis.

## Goal

Use ASR outputs as a diagnostic pivot:

- If ASR transcripts are accurate but ALM NLI is wrong, errors are likely from ALM reasoning/language modeling.
- If ASR transcripts are poor and ALM NLI is wrong, errors may be propagated from audio encoding/recognition.

## Scripts

- `run_whisper_asr.py` — production batch runner for Whisper (`openai/whisper-large-v3` by default).
- `run_parakeet_asr_template.py` — template with the same I/O contract for NVIDIA Parakeet integration.
- `run_granite_asr_template.py` — template with the same I/O contract for IBM Granite Speech integration.

All scripts read a CSV containing audio paths and write:

- CSV output with ASR text and status fields.
- Optional JSONL output for downstream tools.

## Input expectation

Your CSV should include one audio path column (default: `audio_file`).

Supported path styles:

- Absolute path.
- Relative path resolved from `--audio_root`.

## Quick start (Whisper)

```bash
python asr/run_whisper_asr.py \
  --input_csv Entailment/metadata_afrispeech-200.csv \
  --audio_col file_name \
  --audio_root Audio/Afrispeech200 \
  --output_csv outputs/asr/afrispeech200_whisper_large_v3.csv \
  --output_jsonl outputs/asr/afrispeech200_whisper_large_v3.jsonl \
  --model_id openai/whisper-large-v3 \
  --language english
```

Medical example:

```bash
python asr/run_whisper_asr.py \
  --input_csv Entailment/metadata_medical.csv \
  --audio_col file_name \
  --audio_root Audio/medical \
  --output_csv outputs/asr/medical_whisper_large_v3.csv \
  --output_jsonl outputs/asr/medical_whisper_large_v3.jsonl \
  --model_id openai/whisper-large-v3 \
  --language english
```

## Common output columns

- `asr_model`
- `asr_text`
- `asr_status`
- `asr_error`

This consistent schema enables direct comparison across Whisper, Parakeet, and Granite.

## SLURM batch submission (4 datasets)

Scripts:

- `asr/slurm/run_asr_dataset_model.slurm` — runs one `(dataset, model)` job.
- `asr/slurm/submit_all_asr_jobs.sh` — submits all combinations for:
  - Datasets: `afrispeech200`, `medical`, `general`, `afrinames`
  - Models: `whisper`, `parakeet`, `granite`
- `asr/slurm/submit_whisper_asr_jobs.sh` — submits Whisper jobs for all 4 datasets.
- `asr/slurm/submit_parakeet_asr_jobs.sh` — submits Parakeet jobs for all 4 datasets.
- `asr/slurm/submit_granite_asr_jobs.sh` — submits Granite jobs for all 4 datasets.

### Preferred debugging workflow

Submit one model at a time so failures are isolated:

```bash
bash asr/slurm/submit_whisper_asr_jobs.sh
bash asr/slurm/submit_parakeet_asr_jobs.sh
bash asr/slurm/submit_granite_asr_jobs.sh
```

Submit all jobs:

```bash
bash asr/slurm/submit_all_asr_jobs.sh
```

Submit a single job:

```bash
sbatch --export=ALL,MODEL=whisper,DATASET=afrispeech200 asr/slurm/run_asr_dataset_model.slurm
```

Optional runtime overrides:

```bash
export WHISPER_MODEL_ID=openai/whisper-large-v3
export PARAKEET_MODEL_ID=nvidia/parakeet-tdt-0.6b-v2
export GRANITE_MODEL_ID=ibm-granite/granite-speech-3.3-2b
export LIMIT=50
export CHUNK_LENGTH_S=30
export MAX_NEW_TOKENS=1000
bash asr/slurm/submit_all_asr_jobs.sh
```

### Virtual environment strategy

The SLURM runner resolves environments in this order:

1. `VENV_PATH` (if passed via `--export`)
2. `./.venvASR` (recommended dedicated ASR env)

`./.venv` is intentionally not used by ASR SLURM scripts to avoid dependency drift.

Recommended for reliability:

```bash
cd /orange/ufdatastudios/c.okocha/afrispeech-entailment
bash asr/slurm/setup_asr_env.sh
```

### B200 PyTorch compatibility

The setup script installs explicit CUDA 12.8 wheels:

- `torch==2.8.0+cu128`
- `torchaudio==2.8.0+cu128`
- `torchvision==0.23.0+cu128`

Quick verification:

```bash
source .venvASR/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
  i = torch.cuda.current_device()
  print(torch.cuda.get_device_name(i))
  print(torch.cuda.get_device_capability(i))
PY
```

To force a specific env at submit time:

```bash
export VENV_PATH=/orange/ufdatastudios/c.okocha/afrispeech-entailment/.venvASR
bash asr/slurm/submit_whisper_asr_jobs.sh
```

Monitor jobs:

```bash
squeue -u c.okocha
```

Logs:

- `outputs/asr/slurm_logs/*.out`
- `outputs/asr/slurm_logs/*.err`