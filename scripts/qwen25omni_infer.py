#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B inference script for JSONL-based audio entailment tasks.
This is a multimodal model that can handle audio, video, images, and text.
"""
import json
import os
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import librosa
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor


# ========== TASK PROMPT TEMPLATES ==========
# These match PROMPTS.md - Audio Language Model Inference Prompts

PROMPTS = {
    "nli_caption": """You are given an audio recording and a text hypothesis.

First, write a short caption of ONLY what is explicitly said in the audio (1–2 sentences, no guessing).
Then decide whether the hypothesis is ENTAILMENT, CONTRADICTION, or NEUTRAL based ONLY on the caption.

IMPORTANT: Output only the final label on a single line:
ENTAILMENT
or
CONTRADICTION
or
NEUTRAL

Hypothesis: "{hypothesis}"
""",
"nli": """You are given an audio recording of spoken speech and a text statement.

Based only on the content of the audio, determine whether the statement is:
- Entailed by the audio
- Contradicted by the audio
- Neither entailed nor contradicted by the audio

Do not assume any information not present in the audio.

Respond with one of the following labels only:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}""",

    "consistency": """You are given an audio recording and a text statement.

Determine whether the text is consistent with the meaning conveyed in the audio.

Respond with one of the following labels only:
CONSISTENT
INCONSISTENT

STATEMENT:
{hypothesis}""",

    "plausibility": """You are given an audio recording and a text statement.

Is the statement plausible given what is heard in the audio?

Respond with:
PLAUSIBLE
IMPLAUSIBLE

STATEMENT:
{hypothesis}""",

    "intent": """You are given an audio recording of spoken speech.

Which of the following best describes the speaker's communicative intent?

Choose the single best answer from the provided options.

OPTIONS:
{hypothesis}""",

    "restraint": """You are given an audio recording and a text statement.

Does the audio support or provide evidence for the statement?

Respond with one of the following labels only:
SUPPORTED
UNSUPPORTED

STATEMENT:
{hypothesis}""",

    "accent_drift": """You are given an audio recording and a text statement.

Based only on the content of the audio (ignore any accent or speaker characteristics), determine whether the statement is TRUE or FALSE.

If the statement is about the content of the audio and is factually supported, respond TRUE.
If the statement is not supported by the audio content, or if it makes inferences based on accent/speaker characteristics, respond FALSE.

Respond with one of the following labels only:
TRUE
FALSE

STATEMENT:
{hypothesis}""",

    "commonsense": """You are given an audio recording and a text statement.

Does the statement align with commonsense reasoning based on the audio content?

Respond with one of the following labels only:
ALIGNED
NOT_ALIGNED

STATEMENT:
{hypothesis}""",

    "interview_nli": """You are given an audio recording of spoken speech and a text statement.

Based only on the content of the audio, determine whether the statement is:
- Entailed by the audio
- Contradicted by the audio
- Neither entailed nor contradicted by the audio

Do not assume any information not present in the audio.

Respond with one of the following labels only:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}""",
}


def normalize_label(text: str, task: str) -> str:
    """
    Normalize model output to expected label format.
    """
    if not text:
        return "UNPARSEABLE"
    
    text_upper = text.upper().strip()
    
    # Remove common prefixes/suffixes
    text_upper = text_upper.replace("LABEL:", "").replace("ANSWER:", "").strip()
    
    # Task-specific normalization
    if task in ["nli", "nli_caption", "interview_nli"]:
        if "ENTAIL" in text_upper:
            return "ENTAILMENT"
        elif "CONTRADICT" in text_upper:
            return "CONTRADICTION"
        elif "NEUTRAL" in text_upper:
            return "NEUTRAL"
    
    elif task == "consistency":
        if "CONSISTENT" in text_upper:
            return "CONSISTENT"
        elif "INCONSISTENT" in text_upper:
            return "INCONSISTENT"
    
    elif task == "plausibility":
        if "PLAUSIBLE" in text_upper:
            return "PLAUSIBLE"
        elif "IMPLAUSIBLE" in text_upper:
            return "IMPLAUSIBLE"
    
    elif task == "restraint":
        if "SUPPORTED" in text_upper:
            return "SUPPORTED"
        elif "UNSUPPORTED" in text_upper:
            return "UNSUPPORTED"
    
    elif task == "accent_drift":
        if "TRUE" in text_upper or "YES" in text_upper:
            return "TRUE"
        elif "FALSE" in text_upper or "NO" in text_upper:
            return "FALSE"
    
    elif task == "commonsense":
        if "ALIGNED" in text_upper or "ALIGN" in text_upper:
            return "ALIGNED"
        elif "NOT_ALIGNED" in text_upper or "NOT ALIGN" in text_upper:
            return "NOT_ALIGNED"
    
    # Fallback: return first word or UNPARSEABLE
    first_word = text_upper.split()[0] if text_upper.split() else "UNPARSEABLE"
    return first_word if len(first_word) > 2 else "UNPARSEABLE"


def extract_hypotheses(record: dict, task: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract hypotheses from JSONL record.
    Returns list of (hypothesis_text, gold_label) tuples.
    """
    hypotheses = []
    
    # Check for output field with structured hypotheses
    if "output" in record:
        output = record["output"]
        
        if task in ["nli", "nli_caption"]:
            for key in ["entailment", "contradiction", "neutral"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hypotheses.append((hyp.get("hypothesis", ""), hyp.get("label", None)))
                        else:
                            hypotheses.append((str(hyp), key.upper()))
        
        elif task == "consistency":
            for key in ["consistent", "inconsistent"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hypotheses.append((hyp.get("hypothesis", ""), hyp.get("label", None)))
                        else:
                            hypotheses.append((str(hyp), key.upper()))
        
        elif task == "plausibility":
            for key in ["plausible", "implausible"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hypotheses.append((hyp.get("hypothesis", ""), hyp.get("label", None)))
                        else:
                            hypotheses.append((str(hyp), key.upper()))
        
        elif task == "restraint":
            for key in ["supported", "unsupported"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hypotheses.append((hyp.get("hypothesis", ""), hyp.get("label", None)))
                        else:
                            hypotheses.append((str(hyp), key.upper()))
        
        elif task == "accent_drift":
            # accent_invariant should be ACCENT_INVARIANT, accent_sensitive_lures should be ACCENT_TEST
            accent_invariant_hyps = output.get("accent_invariant", [])
            if isinstance(accent_invariant_hyps, list):
                for hyp in accent_invariant_hyps:
                    if isinstance(hyp, dict):
                        hypotheses.append((hyp.get("hypothesis", ""), "ACCENT_INVARIANT"))
                    else:
                        hypotheses.append((str(hyp), "ACCENT_INVARIANT"))
            
            accent_sensitive_hyps = output.get("accent_sensitive_lures", [])
            if isinstance(accent_sensitive_hyps, list):
                for hyp in accent_sensitive_hyps:
                    if isinstance(hyp, dict):
                        hypotheses.append((hyp.get("hypothesis", ""), "ACCENT_TEST"))
                    else:
                        hypotheses.append((str(hyp), "ACCENT_TEST"))
    
    # interview_nli format: direct hypotheses list
    if task == "interview_nli" and "hypotheses" in record:
        hyps_list = record["hypotheses"]
        if isinstance(hyps_list, list):
            for hyp_obj in hyps_list:
                if isinstance(hyp_obj, dict):
                    text = hyp_obj.get("text", "")
                    label = hyp_obj.get("label", None)
                    hypotheses.append((text, label))
    
    # Fallback: direct hypothesis field
    if not hypotheses and "hypothesis" in record:
        gold = record.get("gold", record.get("label", None))
        hypotheses.append((record["hypothesis"], gold))
    
    return hypotheses


def find_audio_path(audio_dir: str, file_name: str) -> Optional[str]:
    """
    Find audio file path, handling various filename formats.
    """
    if not file_name:
        return None
    
    # Remove common prefixes
    basename = file_name
    for prefix in ["data/", "/data/", "Audio/", "/Audio/"]:
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
    
    # Remove UUID prefixes
    if "/" in basename:
        basename = basename.split("/")[-1]
    
    # Try direct match
    candidate = os.path.join(audio_dir, basename)
    if os.path.isfile(candidate):
        return candidate
    
    # Try with common extensions if no extension
    base, ext = os.path.splitext(basename)
    if not ext:
        for e in [".wav", ".flac", ".mp3"]:
            cand = os.path.join(audio_dir, base + e)
            if os.path.isfile(cand):
                return cand
    
    return None


# ========== MODEL INTERFACE ==========

def init_model(model_path: str, dtype: str = "auto", disable_talker: bool = True):
    """
    Initialize Qwen2.5-Omni-7B model.
    
    Args:
        model_path: HuggingFace model ID
        dtype: "auto", "float16", "bfloat16", or "float32"
        disable_talker: If True, disable audio output to save memory (~2GB)
    
    Returns:
        Dict with "model" and "processor" keys
    """
    # Set torch dtype
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    print(f"Loading Qwen2.5-Omni-7B from {model_path}...")
    print(f"Using dtype: {torch_dtype}")
    
    # Get HuggingFace token if available
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        token=token,
        trust_remote_code=True
    )
    
    # Load model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=token,
        trust_remote_code=True
    )
    
    # Disable talker (audio output) to save memory if requested
    if disable_talker:
        model.disable_talker()
        print("Audio output disabled (talker disabled) to save memory")
    
    model.eval()
    
    print("Model loaded successfully!")
    
    return {
        "model": model,
        "processor": processor
    }


def process_mm_info(conversations, use_audio_in_video=False):
    """
    Process multimodal information from conversations.
    For Qwen2.5-Omni, the processor can handle file paths directly.
    """
    audios = []
    images = []
    videos = []
    
    for conv in conversations:
        conv_audios = []
        conv_images = []
        conv_videos = []
        
        for msg in conv:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "audio":
                        path = item.get("audio") or item.get("audio_path")
                        if path:
                            # Qwen2.5-Omni processor accepts file paths directly
                            conv_audios.append(path)
                    elif item.get("type") == "image":
                        path = item.get("image") or item.get("image_path")
                        if path:
                            conv_images.append(path)
                    elif item.get("type") == "video":
                        path = item.get("video") or item.get("video_path")
                        if path:
                            conv_videos.append(path)
        
        # Return as lists (processor expects lists per conversation)
        audios.append(conv_audios if conv_audios else None)
        images.append(conv_images if conv_images else None)
        videos.append(conv_videos if conv_videos else None)
    
    return audios, images, videos


def generate_with_model(model_dict: Dict, audio_path: str, prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate text using Qwen2.5-Omni-7B.
    
    Args:
        model_dict: Dict with "model" and "processor" keys
        audio_path: Path to audio file
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    """
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    # Create conversation format for Qwen2.5-Omni
    # Required system prompt for audio output (but we disable talker, so it's optional)
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process multimodal info
    conversations = [conversation]
    USE_AUDIO_IN_VIDEO = False  # We only have audio, no video
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    # Apply chat template
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    
    # Prepare inputs - Qwen2.5-Omni processor needs audio loaded, not just paths
    # Load audio file if we have one
    audio_data = None
    if audios[0] and len(audios[0]) > 0:
        import librosa
        audio_path = audios[0][0]
        # Load audio at 16kHz (standard for speech models)
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_data = [audio_data]  # Wrap in list for batch processing
    
    # Prepare inputs
    inputs = processor(
        text=text,
        audio=audio_data if audio_data else None,
        images=images[0] if images[0] else None,
        videos=videos[0] if videos[0] else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = inputs.to(model.device).to(model.dtype)
    
    # Generate (text only, no audio output)
    with torch.inference_mode():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=False,
            max_new_tokens=max_new_tokens
        )
        
        text = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    
    return text


# ========== MAIN INFERENCE LOOP ==========

def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Omni-7B inference on JSONL files")
    
    # Model
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model path")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype")
    parser.add_argument("--enable_talker", action="store_true", help="Enable audio output (uses ~2GB more memory)")
    
    # Data
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--audio_dir", type=str, required=True, help="Audio directory")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--task", type=str, required=True, choices=list(PROMPTS.keys()), help="Task type")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for generation")
    
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    
    args = parser.parse_args()
    
    # Load model
    model_dict = init_model(args.model_path, args.dtype, disable_talker=not args.enable_talker)
    
    # Load task prompt template
    prompt_template = PROMPTS[args.task]
    
    # Resume logic
    seen_item_ids = set()
    if args.resume and os.path.exists(args.output_jsonl):
        print(f"Resuming from {args.output_jsonl}...")
        with open(args.output_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    seen_item_ids.add(entry.get("item_id"))
        print(f"Found {len(seen_item_ids)} existing entries")
    
    # Process JSONL
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    n_items = 0
    n_skipped = 0
    n_errors = 0
    
    with open(args.input_jsonl, 'r') as in_f, \
         open(args.output_jsonl, 'a' if args.resume else 'w') as out_f:
        
        for line_num, line in enumerate(in_f, 1):
            try:
                record = json.loads(line)
                
                # Extract base ID and hypotheses
                base_id = record.get("item_id", record.get("id", f"line_{line_num}"))
                file_name = record.get("file_name", record.get("audio_file", ""))
                
                # Handle interview_nli format with audio_id
                if args.task == "interview_nli" and "audio_id" in record:
                    audio_id = record["audio_id"]
                    # Convert audio_id to filename (e.g., "08f" -> "08f.wav")
                    file_name = f"{audio_id}.wav" if not audio_id.endswith(".wav") else audio_id
                
                hypotheses = extract_hypotheses(record, args.task)
                
                for hyp_idx, (hypothesis, gold_label) in enumerate(hypotheses):
                    item_id = f"{base_id}__hyp_{hyp_idx}"
                    
                    if item_id in seen_item_ids:
                        n_skipped += 1
                        continue
                    
                    # Find audio file
                    audio_path = find_audio_path(args.audio_dir, file_name)
                    if audio_path is None:
                        obj = {
                            "item_id": item_id,
                            "file_name": file_name,
                            "audio_path": None,
                            "hypothesis": hypothesis,
                            "gold": gold_label,
                            "pred_raw": None,
                            "retry_count": 0,
                            "error": "Audio file not found",
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                        out_f.write(json.dumps(obj) + "\n")
                        out_f.flush()
                        n_errors += 1
                        continue
                    
                    # Format prompt
                    prompt = prompt_template.format(hypothesis=hypothesis)
                    
                    # Generate
                    try:
                        text = generate_with_model(model_dict, audio_path, prompt, args.max_new_tokens)
                        
                        # Retry if empty
                        retry_count = 0
                        if not text or not text.strip():
                            retry_count = 1
                            text = generate_with_model(model_dict, audio_path, prompt, args.max_new_tokens)
                        
                        # Extract only the assistant's response (after "assistant\n" or "assistant:")
                        # The full text includes the entire conversation, we need just the assistant's part
                        assistant_markers = ["assistant\n", "assistant:", "assistant "]
                        assistant_text = text
                        for marker in assistant_markers:
                            if marker.lower() in text.lower():
                                # Find the last occurrence (most recent assistant response)
                                parts = text.lower().rsplit(marker.lower(), 1)
                                if len(parts) > 1:
                                    assistant_text = parts[-1].strip()
                                    break
                        
                        # Normalize the label
                        pred = normalize_label(assistant_text, args.task)
                        
                        obj = {
                            "item_id": item_id,
                            "file_name": file_name,
                            "audio_path": audio_path,
                            "hypothesis": hypothesis,
                            "gold": gold_label,
                            "pred_raw": assistant_text,
                            "pred": pred,
                            "retry_count": retry_count,
                            "error": None,
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                        out_f.write(json.dumps(obj) + "\n")
                        out_f.flush()
                        n_items += 1
                        
                    except Exception as e:
                        error_msg = str(e)
                        tb = traceback.format_exc()
                        obj = {
                            "item_id": item_id,
                            "file_name": file_name,
                            "audio_path": audio_path,
                            "hypothesis": hypothesis,
                            "gold": gold_label,
                            "pred_raw": None,
                            "retry_count": 0,
                            "error": error_msg,
                            "traceback": tb,
                            "ts": datetime.utcnow().isoformat() + "Z",
                        }
                        out_f.write(json.dumps(obj) + "\n")
                        out_f.flush()
                        n_errors += 1
                        print(f"Error processing {item_id}: {error_msg}")
                
                if (line_num) % 50 == 0:
                    print(f"Processed {line_num} input records (items: {n_items}, skipped: {n_skipped}, errors: {n_errors})")
            
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                n_errors += 1
                continue
    
    print(f"\nDone. Processed {n_items} items (skipped {n_skipped}, errors {n_errors})")
    print(f"Results saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()

