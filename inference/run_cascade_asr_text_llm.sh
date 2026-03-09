#!/bin/bash
set -euo pipefail

# Cascade baseline launcher:
# ASR transcript (Whisper/Granite) + text LLM (Llama/Qwen/Mistral)
# Tasks: nli, consistency, plausibility, restraint, accent_drift

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
CHILD_ROOT="/orange/ufdatastudios/c.okocha/child__speech_analysis"
PYTHON_BIN="${PYTHON_BIN:-${CHILD_ROOT}/.venv/bin/python}"

OUT_ROOT="${ROOT}/outputs/cascade_asr_text_llm"
ASR_ROOT="${ROOT}/outputs/asr/simple"

ASR_MODELS="${ASR_MODELS:-whisper granite}"
TEXT_MODELS="${TEXT_MODELS:-llama qwen mistral}"

# Single-label classification; keep generation short.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DRY_RUN="${DRY_RUN:-1}"
RESUME="${RESUME:-1}"

# Task config: task_key|task_name|task_jsonl|asr_dataset_key
TASK_SPECS=(
  "afri200_nli|nli|${ROOT}/result/Entailment/AfriSpeech200/Llama/nli_single/afrispeech200_nli_top100.jsonl|afrispeech200"
  "medical_nli|nli|${ROOT}/result/Entailment/Medical/Llama/nli/medical_nli.jsonl|medical"
  "afri200_consistency|consistency|${ROOT}/result/Entailment/AfriSpeech200/Llama/consistency_single/afrispeech200_consistency_top100.jsonl|afrispeech200"
  "medical_consistency|consistency|${ROOT}/result/Entailment/Medical/Llama/consistency/medical_consistency.jsonl|medical"
  "general_consistency|consistency|${ROOT}/result/Entailment/AfriSpeechGeneral/Llama/consistency/afrispeech_general_consistency.jsonl|general"
  "afri200_plausibility|plausibility|${ROOT}/result/Entailment/AfriSpeech200/Llama/plausibility_single/afrispeech200_plausibility_top100.jsonl|afrispeech200"
  "medical_plausibility|plausibility|${ROOT}/result/Entailment/Medical/Llama/plausibility/medical_plausibility.jsonl|medical"
  "general_plausibility|plausibility|${ROOT}/result/Entailment/AfriSpeechGeneral/Llama/plausibility/afrispeech_general_plausibility.jsonl|general"
  "afrinames_restraint|restraint|${ROOT}/result/Entailment/AfriNames/Llama/restraint/afri_names_restraint_top100.jsonl|afrinames"
  "afrinames_accent_drift|accent_drift|${ROOT}/result/Entailment/AfriNames/Llama/accent_drift/afri_names_accent_top100.jsonl|afrinames"
)

resolve_model_id() {
  local model_key="$1"
  case "$model_key" in
    llama) echo "meta-llama/Meta-Llama-3.1-8B-Instruct" ;;
    qwen) echo "Qwen/Qwen2-7B-Instruct" ;;
    mistral) echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
    *) echo "" ;;
  esac
}

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $cmd"
  else
    eval "$cmd"
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found/executable: $PYTHON_BIN"
  exit 1
fi

mkdir -p "$OUT_ROOT"

for asr in $ASR_MODELS; do
  for text_model in $TEXT_MODELS; do
    model_id="$(resolve_model_id "$text_model")"
    if [[ -z "$model_id" ]]; then
      echo "Unsupported TEXT_MODELS entry: $text_model"
      exit 1
    fi

    for spec in "${TASK_SPECS[@]}"; do
      IFS='|' read -r task_key task_name task_jsonl asr_dataset <<< "$spec"

      asr_jsonl="${ASR_ROOT}/${asr_dataset}_${asr}_asr.jsonl"
      out_jsonl="${OUT_ROOT}/${asr}/${text_model}/${task_key}.jsonl"
      mkdir -p "$(dirname "$out_jsonl")"

      if [[ ! -f "$task_jsonl" ]]; then
        echo "Missing task jsonl: $task_jsonl"
        exit 1
      fi
      if [[ ! -f "$asr_jsonl" ]]; then
        echo "Missing ASR jsonl: $asr_jsonl"
        exit 1
      fi

      resume_flag=""
      if [[ "$RESUME" == "1" ]]; then
        resume_flag="--resume"
      fi

      cmd="$PYTHON_BIN ${ROOT}/inference/run_text_llm_cascade.py \
        --model_id '${model_id}' \
        --task '${task_name}' \
        --task_jsonl '${task_jsonl}' \
        --asr_jsonl '${asr_jsonl}' \
        --output_jsonl '${out_jsonl}' \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE} \
        ${resume_flag}"

      echo "=== ASR=${asr} TEXT_MODEL=${text_model} TASK=${task_key} ==="
      run_cmd "$cmd"
    done
  done
done

echo "Cascade sweep complete (DRY_RUN=${DRY_RUN})"
