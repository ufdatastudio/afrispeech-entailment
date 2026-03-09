#!/bin/bash
set -euo pipefail

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
SCRIPT="${ROOT}/inference/slurm/run_interview_cascade_text_llm.slurm"

submit_one() {
  local model_key="$1"
  local model_id="$2"

  sbatch \
    --job-name "iv-cascade-${model_key}" \
    --export=ALL,TEXT_MODEL="${model_key}",MODEL_ID="${model_id}" \
    "${SCRIPT}"
}

submit_one "llama" "meta-llama/Meta-Llama-3.1-8B-Instruct"
submit_one "qwen" "Qwen/Qwen2-7B-Instruct"
submit_one "mistral" "mistralai/Mistral-7B-Instruct-v0.3"

echo "Submitted interview cascade jobs for llama, qwen, mistral."
