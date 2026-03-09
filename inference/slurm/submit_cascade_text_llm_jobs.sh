#!/bin/bash
set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
cd "$ROOT"

MODELS=(llama qwen mistral)

echo "Submitting cascade text-LLM jobs for models: ${MODELS[*]}"

auto_export="ALL"
if [[ -n "${ASR_MODELS:-}" ]]; then auto_export+=",ASR_MODELS=${ASR_MODELS}"; fi
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then auto_export+=",MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"; fi
if [[ -n "${TEMPERATURE:-}" ]]; then auto_export+=",TEMPERATURE=${TEMPERATURE}"; fi
if [[ -n "${RESUME:-}" ]]; then auto_export+=",RESUME=${RESUME}"; fi
if [[ -n "${PYTHON_BIN:-}" ]]; then auto_export+=",PYTHON_BIN=${PYTHON_BIN}"; fi

for model in "${MODELS[@]}"; do
  script="inference/slurm/run_cascade_text_llm_${model}.slurm"
  job_name="cascade-${model}"
  echo "Submitting ${job_name} via ${script}"
  sbatch --job-name="$job_name" --export="$auto_export" "$script"
done

echo "Submitted all cascade jobs."
echo "Check queue: squeue -u c.okocha"
