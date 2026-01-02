#!/usr/bin/env python3
"""
Audio Flamingo 2 inference script for JSONL-based audio entailment tasks.

This is a template adapted from Audio Flamingo 3, customized for Audio Flamingo 2.
"""
import json
import os
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Try to import Audio Flamingo 2 classes
# Audio Flamingo 2 may use different class names - try multiple options
try:
    from transformers import AudioFlamingo2ForConditionalGeneration, AutoProcessor
    AF2_CLASS = AudioFlamingo2ForConditionalGeneration
    print("Using AudioFlamingo2ForConditionalGeneration")
except ImportError:
    try:
        # Fallback: try AutoModel approach
        from transformers import AutoModelForConditionalGeneration, AutoProcessor
        AF2_CLASS = AutoModelForConditionalGeneration
        print("Using AutoModelForConditionalGeneration (fallback)")
    except ImportError:
        raise ImportError("Could not import Audio Flamingo 2 classes. Please check transformers version.")


# ========== TASK PROMPT TEMPLATES ==========
# These match PROMPTS.md - Audio Language Model Inference Prompts

PROMPTS = {
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
{options}

Respond with the exact option text only.""",

    "commonsense": """You are given an audio recording of spoken speech.

Based on common knowledge and the content of the audio, answer the following question.

Respond with:
YES
NO

QUESTION:
{hypothesis}""",

    "restraint": """You are given an audio recording of spoken speech and a text statement.

Based only on the content of the audio, determine whether the statement is:
- Supported by the audio
- Not supported by the audio

Do not assume any information not present in the audio.

Respond with one of the following labels only:
SUPPORTED
UNSUPPORTED

STATEMENT:
{hypothesis}""",

    "accent_drift": """You are given an audio recording of spoken speech and a text statement.

Based only on the content of the audio, determine whether the statement is:
- True (supported by the audio)
- False (not supported by the audio)

Do not assume any information not present in the audio.

Respond with one of the following labels only:
TRUE
FALSE

STATEMENT:
{hypothesis}""",
}


def normalize_label(text: str, task: str) -> str:
    """
    Normalize model output to canonical labels.
    """
    if not text:
        return "UNPARSEABLE"
    
    text_lower = text.strip().lower()
    
    if task == "nli":
        if "entail" in text_lower or text_lower.startswith("entail"):
            return "ENTAILMENT"
        elif "contradict" in text_lower or text_lower.startswith("contradict"):
            return "CONTRADICTION"
        elif "neutral" in text_lower or "neither" in text_lower:
            return "NEUTRAL"
    
    elif task == "consistency":
        if "consistent" in text_lower:
            return "CONSISTENT"
        elif "inconsistent" in text_lower:
            return "INCONSISTENT"
    
    elif task == "plausibility":
        if "plausible" in text_lower:
            return "PLAUSIBLE"
        elif "implausible" in text_lower:
            return "IMPLAUSIBLE"
    
    elif task == "intent":
        # For intent, return the raw text (should match one of the options)
        return text.strip()
    
    elif task == "commonsense":
        if text_lower.startswith("yes") or text_lower == "y":
            return "YES"
        elif text_lower.startswith("no") or text_lower == "n":
            return "NO"
    
    elif task == "restraint":
        if "support" in text_lower and "not" not in text_lower:
            return "SUPPORTED"
        elif "not support" in text_lower or "unsupport" in text_lower:
            return "UNSUPPORTED"
    
    elif task == "accent_drift":
        if text_lower.startswith("true") or text_lower == "t":
            return "TRUE"
        elif text_lower.startswith("false") or text_lower == "f":
            return "FALSE"
    
    return "UNPARSEABLE"


def extract_hypotheses(record: Dict, task: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract hypotheses and gold labels from JSONL record.
    """
    hypotheses = []
    
    # Check if record has 'output' field with structured hypotheses
    if "output" in record and isinstance(record["output"], dict):
        output = record["output"]
        
        if task == "nli":
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
            # accent_invariant should be TRUE, accent_sensitive_lures should be FALSE
            accent_invariant_hyps = output.get("accent_invariant", [])
            if isinstance(accent_invariant_hyps, list):
                for hyp in accent_invariant_hyps:
                    if isinstance(hyp, dict):
                        hypotheses.append((hyp.get("hypothesis", ""), "TRUE"))
                    else:
                        hypotheses.append((str(hyp), "TRUE"))
            
            accent_sensitive_hyps = output.get("accent_sensitive_lures", [])
            if isinstance(accent_sensitive_hyps, list):
                for hyp in accent_sensitive_hyps:
                    if isinstance(hyp, dict):
                        hypotheses.append((hyp.get("hypothesis", ""), "FALSE"))
                    else:
                        hypotheses.append((str(hyp), "FALSE"))
    
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
    
    # Remove UUID prefixes (e.g., "uuid/filename.wav" -> "filename.wav")
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

def init_model(model_path: str, dtype: str = "auto"):
    """
    Initialize Audio Flamingo 2 model.
    
    Args:
        model_path: HuggingFace model ID or path (e.g., "nvidia/audio-flamingo-2")
        dtype: "auto", "float16", "bfloat16", or "float32"
    
    Returns:
        Dict with "model" and "processor" keys
    """
    # Set torch dtype
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    
    print(f"Loading Audio Flamingo 2 from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load model - try the specific class first, fallback to AutoModel
    if AF2_CLASS == AutoModelForConditionalGeneration:
        model = AF2_CLASS.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    else:
        model = AF2_CLASS.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    
    return {"model": model, "processor": processor, "device": model.device}


def generate_with_model(model_dict: Dict, messages: List[Dict], sampling_params: Dict, max_new_tokens: int) -> str:
    """
    Generate text using Audio Flamingo 2.
    
    Args:
        model_dict: Dict with "model", "processor", and "device" keys
        messages: List of message dicts (will be converted to AF2 conversation format)
        sampling_params: Model-specific sampling parameters
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    """
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = model_dict["device"]
    
    # Convert messages to AF2 conversation format
    # AF2 likely expects similar format to AF3: [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "audio", "path": "..."}]}]
    conversation = []
    for msg in messages:
        if msg.get("message_type") == "text":
            if not conversation or conversation[-1]["role"] != "user":
                conversation.append({"role": "user", "content": []})
            conversation[-1]["content"].append({"type": "text", "text": msg["content"]})
        elif msg.get("message_type") == "audio":
            if not conversation or conversation[-1]["role"] != "user":
                conversation.append({"role": "user", "content": []})
            conversation[-1]["content"].append({"type": "audio", "path": msg["content"]})
    
    # Apply chat template
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(device)
    
    # Convert audio input features to match model dtype (if needed)
    if hasattr(model, 'dtype'):
        model_dtype = model.dtype
    elif hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
        model_dtype = model.config.torch_dtype
    else:
        model_dtype = torch.bfloat16  # Default
    
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)
    if "audio_features" in inputs:
        inputs["audio_features"] = inputs["audio_features"].to(dtype=model_dtype)
    
    # Generation kwargs
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if sampling_params.get("temperature", 0.0) > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": sampling_params.get("temperature", 0.7),
        })
    else:
        gen_kwargs.update({"do_sample": False})
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode
    decoded = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]
    
    return decoded


# ========== MAIN INFERENCE LOOP ==========

def main():
    parser = argparse.ArgumentParser(description="Run Audio Flamingo 2 inference on JSONL files")
    
    # Model
    parser.add_argument("--model_path", type=str, default="nvidia/audio-flamingo-2", help="HF model ID or path")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16|float32")
    
    # Input
    parser.add_argument("--jsonl_path", type=str, required=True, help="Input JSONL file with hypotheses")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["nli", "consistency", "plausibility", "intent", "commonsense", "restraint", "accent_drift"],
                       help="Task type")
    
    # Output
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL file")
    
    # Limits
    parser.add_argument("--max_rows", type=int, default=-1, help="Limit number of rows (-1 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip items already in out_jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for generation")
    
    # Model-specific sampling params
    parser.add_argument("--temperature", type=float, default=0.0, help="Text generation temperature")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k sampling (not used if temperature=0)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {args.jsonl_path}")
    if not os.path.isdir(args.audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {args.audio_dir}")
    
    # Initialize model
    print(f"Initializing Audio Flamingo 2 from {args.model_path}...")
    model_dict = init_model(args.model_path, dtype=args.dtype)
    print("Model loaded successfully!")
    
    # Load prompt template
    if args.task not in PROMPTS:
        raise ValueError(f"Unknown task: {args.task}")
    prompt_template = PROMPTS[args.task]
    
    # Sampling params
    sampling_params = {
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    
    # Load input JSONL
    print(f"Loading JSONL from {args.jsonl_path}...")
    records = []
    with open(args.jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if args.max_rows > 0:
        records = records[:args.max_rows]
    
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
    
    # Inference loop
    jsonl_mode = "a" if args.resume else "w"
    all_results = {}
    n_items = 0
    n_skipped = 0
    n_errors = 0
    
    with open(args.out_jsonl, jsonl_mode) as out_f:
        for record_idx, record in enumerate(records):
            file_name = record.get("file_name", "")
            audio_path = find_audio_path(args.audio_dir, file_name)
            
            base_id = os.path.splitext(os.path.basename(file_name))[0]
            
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
                    all_results[item_id] = obj
                    n_errors += 1
                    continue
                
                # Build prompt
                if args.task == "intent":
                    options = "\n".join([f"- {h}" for h, _ in hypotheses])
                    prompt = prompt_template.format(options=options, hypothesis=hypothesis)
                else:
                    prompt = prompt_template.format(hypothesis=hypothesis)
                
                # Build messages (AF2 format - similar to AF3)
                messages = [
                    {"role": "user", "message_type": "text", "content": prompt},
                    {"role": "user", "message_type": "audio", "content": audio_path},
                ]
                
                try:
                    # Generate
                    text = generate_with_model(model_dict, messages, sampling_params, args.max_new_tokens)
                    
                    # Retry if empty
                    retry_count = 0
                    if not text or not text.strip():
                        retry_count = 1
                        retry_params = sampling_params.copy()
                        retry_params["temperature"] = 0.0
                        retry_params["top_k"] = 1
                        text = generate_with_model(model_dict, messages, retry_params, args.max_new_tokens)
                    
                    pred = normalize_label(text, args.task)
                    
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": audio_path,
                        "hypothesis": hypothesis,
                        "gold": gold_label,
                        "pred_raw": text,
                        "pred": pred,
                        "retry_count": retry_count,
                        "error": None,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    all_results[item_id] = obj
                    n_items += 1
                    
                    if n_items % 50 == 0:
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
                        "retry_count": 0,
                        "error": err,
                        "traceback": tb,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    all_results[item_id] = obj
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

