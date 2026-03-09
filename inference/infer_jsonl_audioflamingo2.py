#!/usr/bin/env python3
"""
AudioFlamingo2 inference for interview_nli task.
Based on NVIDIA's official AudioFlamingo2 inference code.
"""

import os
import sys
import yaml
import json
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# The script must be run from inference_HF_pretrained directory for imports to work
from src.factory import create_model_and_transforms
from utils import Dict2Class, get_autocast, get_cast_dtype

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ========== TASK PROMPT TEMPLATES ==========
PROMPTS = {
    "interview_nli": """You are listening to a child discussing their experience with stuttering.
Listen carefully to the audio and then evaluate the hypothesis below.

Determine if the hypothesis is:
- ENTAILMENT: The audio clearly supports or implies the hypothesis
- CONTRADICTION: The audio clearly contradicts the hypothesis  
- NEUTRAL: The audio neither supports nor contradicts the hypothesis

Output ONLY one of these three labels:
ENTAILMENT
CONTRADICTION
NEUTRAL

Hypothesis: "{hypothesis}"
""",
}


def normalize_label(text: str, task: str) -> str:
    """Normalize model output to canonical labels."""
    if not text:
        return "UNPARSEABLE"
    
    text_lower = text.strip().lower()
    
    if task == "interview_nli":
        if "entail" in text_lower or text_lower.startswith("entail"):
            return "ENTAILMENT"
        elif "contradict" in text_lower or text_lower.startswith("contradict"):
            return "CONTRADICTION"
        elif "neutral" in text_lower or "neither" in text_lower:
            return "NEUTRAL"
    
    return "UNPARSEABLE"


def extract_hypotheses(record: Dict, task: str) -> List[Tuple[str, Optional[str]]]:
    """Extract hypotheses and gold labels from JSONL record."""
    hypotheses = []

    # Overlay/NLI format: hypotheses are stored in record["output"] by label key.
    if "output" in record and isinstance(record["output"], dict):
        output = record["output"]
        if task == "interview_nli":
            for key in ["entailment", "contradiction", "neutral"]:
                if key not in output:
                    continue
                raw_items = output[key]
                items = raw_items if isinstance(raw_items, list) else [raw_items]
                for item in items:
                    if isinstance(item, dict):
                        text = item.get("hypothesis", item.get("text", ""))
                        label = item.get("label", key.upper())
                    else:
                        text = str(item)
                        label = key.upper()
                    if text:
                        hypotheses.append((text, label))

    # Handle interview_nli format with hypotheses list
    if task == "interview_nli" and "hypotheses" in record:
        raw_hyps = record.get("hypotheses", [])
        if isinstance(raw_hyps, list):
            for hyp in raw_hyps:
                if isinstance(hyp, dict):
                    hypotheses.append((hyp.get("text", ""), hyp.get("label", None)))
                else:
                    hypotheses.append((str(hyp), None))
    
    if not hypotheses and "hypothesis" in record:
        gold = record.get("gold", record.get("label", None))
        hypotheses.append((record["hypothesis"], gold))
    
    return hypotheses


def find_audio_path(audio_dir: str, file_name: str) -> Optional[str]:
    """Find audio file path."""
    if not file_name:
        return None
    
    basename = file_name
    if "/" in basename:
        basename = basename.split("/")[-1]
    
    candidate = os.path.join(audio_dir, basename)
    if os.path.isfile(candidate):
        return candidate
    
    base, ext = os.path.splitext(basename)
    if not ext:
        for e in [".wav", ".flac", ".mp3"]:
            cand = os.path.join(audio_dir, base + e)
            if os.path.isfile(cand):
                return cand
    
    return None


# ========== AUDIO PROCESSING (from official AF2 code) ==========
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def get_num_windows(T, sr, clap_config):
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    
    return num_windows, full_length


def read_audio(file_path, target_sr, duration, start, clap_config):
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))
    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    
    assert len(data.shape) == 1, data.shape
    return data


def load_audio(audio_path, clap_config):
    sr = 16000
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]
    
    audio_data = read_audio(audio_path, sr, duration, 0.0, clap_config)
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr, clap_config)

    # pads to the nearest multiple of window_length
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
    
    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) > max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)
    
    return audio_clips, audio_embed_mask


def predict(model, tokenizer, device_id, cast_dtype, filepath, question, clap_config, inference_kwargs):
    """Generate prediction for a single audio + question."""
    audio_clips, audio_embed_mask = load_audio(filepath, clap_config)
    audio_clips = audio_clips.to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=cast_dtype, non_blocking=True)
    
    text_prompt = str(question).lower()
    sample = f"<audio>{text_prompt.strip()}{tokenizer.sep_token}"

    text = tokenizer(
        sample,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device_id, non_blocking=True)
    prompt = input_ids

    with torch.no_grad():
        output = model.generate(
            audio_x=audio_clips.unsqueeze(0),
            audio_x_mask=audio_embed_mask.unsqueeze(0),
            lang_x=prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            **inference_kwargs,
        )[0]
    
    output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(
        tokenizer.eos_token, ''
    ).replace(
        tokenizer.pad_token, ''
    ).replace(
        '<|endofchunk|>', ''
    ).strip()

    return output_decoded


def main():
    parser = argparse.ArgumentParser(description="Run AudioFlamingo2 inference on JSONL files")
    
    # Input
    parser.add_argument("--jsonl_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--task", type=str, required=True, choices=["interview_nli"], help="Task type")
    
    # Output
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL file")
    
    # Model
    parser.add_argument("--model_checkpoint_dir", type=str, required=True, help="Path to downloaded AF2 checkpoint dir")
    parser.add_argument("--config_path", type=str, default=None, help="Path to inference config YAML")
    parser.add_argument("--resume", action="store_true", help="Skip items already in out_jsonl")
    
    args = parser.parse_args()
    
    # Load prompt template
    if args.task not in PROMPTS:
        raise ValueError(f"Unknown task: {args.task}")
    prompt_template = PROMPTS[args.task]
    
    # Load config
    if args.config_path is None:
        candidates = [
            os.path.join(os.getcwd(), "configs", "inference.yaml"),
            os.path.join(os.environ.get("AF2_PROJECT_DIR", ""), "configs", "inference.yaml"),
            os.path.join(os.environ.get("AF2_ROOT", ""), "inference_HF_pretrained", "configs", "inference.yaml"),
        ]
        args.config_path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if args.config_path is None:
            raise FileNotFoundError(
                "Could not resolve default AF2 config path. Set --config_path or AF2_PROJECT_DIR/AF2_ROOT."
            )
    
    print(f"Loading config from {args.config_path}...")
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    train_args = Dict2Class(config['train_config'])

    # Initialize model
    print(f"\nInitializing AudioFlamingo2 model...")
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=train_args.offline,
        gradient_checkpointing=train_args.gradient_checkpointing,
        freeze_lm_embeddings=train_args.freeze_lm_embeddings,
    )

    device_id = 0
    model = model.to(device_id)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint from {args.model_checkpoint_dir}...")
    metadata_path = os.path.join(args.model_checkpoint_dir, "safe_ckpt/metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct the full state_dict
    state_dict = {}
    for chunk_name in metadata:
        chunk_path = os.path.join(args.model_checkpoint_dir, f"safe_ckpt/{chunk_name}.safetensors")
        chunk_tensors = load_file(chunk_path)
        state_dict.update(chunk_tensors)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    print(f"Loaded checkpoint: {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")

    autocast = get_autocast(train_args.precision, cache_enabled=(not train_args.fsdp))
    cast_dtype = get_cast_dtype(train_args.precision)

    # Load input JSONL
    print(f"\nLoading JSONL from {args.jsonl_path}...")
    records = []
    with open(args.jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    
    # Resume support
    seen_item_ids = set()
    if args.resume and os.path.isfile(args.out_jsonl):
        print(f"Resuming from {args.out_jsonl}...")
        with open(args.out_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    seen_item_ids.add(obj.get("item_id"))
        print(f"Found {len(seen_item_ids)} existing items")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    
    # Inference kwargs (deterministic for benchmarking)
    inference_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
    }
    
    # Inference loop
    jsonl_mode = "a" if args.resume else "w"
    n_items = 0
    n_skipped = 0
    n_errors = 0
    
    with open(args.out_jsonl, jsonl_mode) as out_f:
        for record_idx, record in enumerate(records):
            # Handle both file_name and audio_id fields
            file_name = record.get("file_name", "")
            if args.task == "interview_nli" and not file_name and "audio_id" in record:
                file_name = f"{record['audio_id']}.wav"
            
            audio_path = find_audio_path(args.audio_dir, file_name)
            base_id = os.path.splitext(os.path.basename(file_name))[0] if file_name else f"record_{record_idx}"
            
            # Extract hypotheses
            hypotheses = extract_hypotheses(record, args.task)
            
            for hyp_idx, (hypothesis, gold_label) in enumerate(hypotheses):
                item_id = f"{base_id}__hyp_{hyp_idx}"
                
                if item_id in seen_item_ids:
                    n_skipped += 1
                    continue
                
                if audio_path is None:
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": None,
                        "hypothesis": hypothesis,
                        "gold": gold_label,
                        "pred_raw": None,
                        "pred": None,
                        "error": f"Audio not found for {file_name}",
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    n_errors += 1
                    continue
                
                # The ICL overlay stores a fully formed prompt string in record["output"].
                # If that is present, use it directly; otherwise apply the default template.
                if (
                    isinstance(hypothesis, str)
                    and "Target Hypothesis:" in hypothesis
                    and "Answer:" in hypothesis
                ):
                    prompt = hypothesis
                else:
                    prompt = prompt_template.format(hypothesis=hypothesis)
                
                try:
                    # Generate
                    text = predict(model, tokenizer, device_id, cast_dtype, audio_path, prompt, clap_config, inference_kwargs)
                    pred = normalize_label(text, args.task)
                    
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": audio_path,
                        "hypothesis": hypothesis,
                        "gold": gold_label,
                        "pred_raw": text,
                        "pred": pred,
                        "error": None,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    n_items += 1
                    
                    if n_items % 10 == 0:
                        print(f"Processed {n_items} items (skipped {n_skipped}, errors {n_errors})")
                
                except Exception as e:
                    err = str(e) or e.__class__.__name__
                    tb = traceback.format_exc()
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": audio_path,
                        "hypothesis": hypothesis,
                        "gold": gold_label,
                        "pred_raw": None,
                        "pred": None,
                        "error": err,
                        "traceback": tb,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    n_errors += 1
                    print(f"Failed {item_id}: {err}")
    
    print(
        f"\nDone.\n"
        f"- Saved JSONL: {args.out_jsonl}\n"
        f"- Items processed: {n_items}\n"
        f"- Items skipped (resume): {n_skipped}\n"
        f"- Errors: {n_errors}\n"
    )


if __name__ == "__main__":
    main()
