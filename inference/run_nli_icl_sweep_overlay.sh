#!/bin/bash
set -euo pipefail

# ICL overlay sweep launcher (N=1..10) for:
# - Qwen2-Audio-7B-Instruct
# - Kimi-Audio-7B-Instruct
# - AudioFlamingo2
# - AudioFlamingo3
# Datasets:
# - Afri-200
# - Medical

ROOT="/orange/ufdatastudios/c.okocha/afrispeech-entailment"
OVERLAY_ROOT="${ROOT}/result/Entailment/ICL_overlay_nli"
OUT_ROOT="${ROOT}/outputs/ICL_overlay_nli"
ICL_VARIANT="${ICL_VARIANT:-audio_only}"  # audio_only | audio_plus_transcript | audio_exemplar_only

MODEL="${MODEL:-qwen2audio}"  # qwen2audio | kimi | af2 | af3
# Sparse default sweep for quick, high-signal comparison.
SHOTS="${SHOTS:-1 3 5 7 10}"
DATASETS="${DATASETS:-afri200 medical}"
DRY_RUN="${DRY_RUN:-1}"        # 1 = print commands only, 0 = execute

AFRI200_AUDIO_DIR="${AFRI200_AUDIO_DIR:-/orange/ufdatastudios/c.okocha/afrispeech-entailment/Audio/Afrispeech200}"
MEDICAL_AUDIO_DIR="${MEDICAL_AUDIO_DIR:-/orange/ufdatastudios/c.okocha/afrispeech-entailment/Audio/medical}"

QWEN2AUDIO_PROJECT_DIR="${QWEN2AUDIO_PROJECT_DIR:-/orange/ufdatastudios/c.okocha/AfroBust/models/Qwen/Qwen2-Audio-7B-Instruct}"
QWEN2AUDIO_PY="${QWEN2AUDIO_PY:-infer_jsonl.py}"
QWEN2AUDIO_MODEL_PATH="${QWEN2AUDIO_MODEL_PATH:-Qwen/Qwen2-Audio-7B-Instruct}"

KIMI_PROJECT_DIR="${KIMI_PROJECT_DIR:-/orange/ufdatastudios/c.okocha/Kimi-Audio}"
KIMI_PY="${KIMI_PY:-infer_jsonl.py}"
KIMI_MODEL_PATH="${KIMI_MODEL_PATH:-moonshotai/Kimi-Audio-7B-Instruct}"

AF2_PROJECT_DIR="${AF2_PROJECT_DIR:-/orange/ufdatastudios/c.okocha/audio-flamingo-2/inference_HF_pretrained}"
AF2_PY="${AF2_PY:-/orange/ufdatastudios/c.okocha/afrispeech-entailment/inference/infer_jsonl_audioflamingo2.py}"
AF2_CKPT_DIR="${AF2_CKPT_DIR:-.}"

AF3_PROJECT_DIR="${AF3_PROJECT_DIR:-/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3}"
AF3_PY="${AF3_PY:-infer_jsonl_interview_nli.py}"
AF3_MODEL_PATH="${AF3_MODEL_PATH:-/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3/audio-flamingo-3}"

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $cmd"
  else
    eval "$cmd"
  fi
}

build_input_path() {
  local dataset="$1"
  local shot="$2"
  local variant_dir="${OVERLAY_ROOT}/${ICL_VARIANT}"
  if [[ "$dataset" == "afri200" ]]; then
    echo "${variant_dir}/afrispeech200_nli_icl_shot${shot}.jsonl"
  else
    echo "${variant_dir}/medical_nli_icl_shot${shot}.jsonl"
  fi
}

build_audio_dir() {
  local dataset="$1"
  if [[ "$dataset" == "afri200" ]]; then
    echo "$AFRI200_AUDIO_DIR"
  else
    echo "$MEDICAL_AUDIO_DIR"
  fi
}

for shot in $SHOTS; do
  for dataset in $DATASETS; do
    INPUT_JSONL="$(build_input_path "$dataset" "$shot")"
    AUDIO_DIR="$(build_audio_dir "$dataset")"
    OUT_DIR="${OUT_ROOT}/${ICL_VARIANT}/${MODEL}/${dataset}/shot${shot}"
    mkdir -p "$OUT_DIR"

    if [[ ! -f "$INPUT_JSONL" ]]; then
      echo "Missing overlay file: $INPUT_JSONL"
      echo "Run build_nli_icl_overlay.py first with --variant ${ICL_VARIANT} (or --variant both)."
      exit 1
    fi

    case "$MODEL" in
      qwen2audio)
        CMD="cd ${QWEN2AUDIO_PROJECT_DIR} && python ${QWEN2AUDIO_PY} --model_path ${QWEN2AUDIO_MODEL_PATH} --input_jsonl ${INPUT_JSONL} --audio_dir ${AUDIO_DIR} --task nli --output_jsonl ${OUT_DIR}/predictions.jsonl --max_new_tokens 512"
        ;;
      kimi)
        CMD="cd ${KIMI_PROJECT_DIR} && python ${KIMI_PY} --model_path ${KIMI_MODEL_PATH} --jsonl_path ${INPUT_JSONL} --audio_dir ${AUDIO_DIR} --task nli --out_jsonl ${OUT_DIR}/predictions.jsonl --max_new_tokens 32"
        ;;
      af2)
        CMD="cd ${AF2_PROJECT_DIR} && PYTHONPATH=${AF2_PROJECT_DIR}:${PYTHONPATH:-} python ${AF2_PY} --jsonl_path ${INPUT_JSONL} --audio_dir ${AUDIO_DIR} --task interview_nli --out_jsonl ${OUT_DIR}/predictions.jsonl --model_checkpoint_dir ${AF2_CKPT_DIR} --resume"
        ;;
      af3)
        CMD="cd ${AF3_PROJECT_DIR} && python ${AF3_PY} --model_path ${AF3_MODEL_PATH} --prompt_variant interview_nli --jsonl_path ${INPUT_JSONL} --audio_dir ${AUDIO_DIR} --task nli --out_jsonl ${OUT_DIR}/predictions.jsonl"
        ;;
      *)
        echo "Unsupported MODEL=${MODEL}. Use qwen2audio|kimi|af2|af3"
        exit 1
        ;;
    esac

    echo "=== MODEL=${MODEL} DATASET=${dataset} SHOT=${shot} ==="
    run_cmd "$CMD"
  done
done

echo "Finished sweep launcher for MODEL=${MODEL}, ICL_VARIANT=${ICL_VARIANT} (DRY_RUN=${DRY_RUN})"
