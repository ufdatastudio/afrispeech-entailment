#!/bin/bash
set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
cd "$ROOT"

SCRIPT="asr/slurm/run_granite_asr.slurm"
MODEL="granite"
DATASETS=(afrispeech200 medical general afrinames)

echo "Submitting ${MODEL} ASR jobs for datasets: ${DATASETS[*]}"

for dataset in "${DATASETS[@]}"; do
  JOB_NAME="asr-${dataset}-${MODEL}"
  echo "Submitting $JOB_NAME"

  EXPORTS="ALL,DATASET=${dataset}"
  if [[ -n "${GRANITE_MODEL_ID:-}" ]]; then EXPORTS+="\,GRANITE_MODEL_ID=${GRANITE_MODEL_ID}"; fi
  if [[ -n "${LIMIT:-}" ]]; then EXPORTS+="\,LIMIT=${LIMIT}"; fi
  if [[ -n "${CHUNK_LENGTH_S:-}" ]]; then EXPORTS+="\,CHUNK_LENGTH_S=${CHUNK_LENGTH_S}"; fi
  if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then EXPORTS+="\,MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"; fi
  if [[ -n "${VENV_PATH:-}" ]]; then EXPORTS+="\,VENV_PATH=${VENV_PATH}"; fi

  sbatch --job-name="$JOB_NAME" --export="$EXPORTS" "$SCRIPT"
done

echo "Submitted Granite ASR jobs."
echo "Check queue: squeue -u c.okocha"