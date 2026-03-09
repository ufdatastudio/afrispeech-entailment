#!/usr/bin/env python3
"""
SALMONN inference script for JSONL-based audio entailment tasks.
Adapted from SALMONN's cli_inference.py to work with our JSONL format.
"""
import json
import os
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import WhisperFeatureExtractor

# SALMONN imports
from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


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

    # interview_nli uses the same prompt structure as nli
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
    """Normalize model output to canonical labels."""
    if not text:
        return "UNPARSEABLE"
    
    text_lower = text.strip().lower()
    
    if task in ("nli", "interview_nli"):
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
    """Extract hypotheses and gold labels from JSONL record."""
    hypotheses = []

    if task == "interview_nli" and "hypotheses" in record:
        raw_hyps = record.get("hypotheses", [])
        if isinstance(raw_hyps, list):
            for hyp in raw_hyps:
                if isinstance(hyp, dict):
                    hypotheses.append((hyp.get("text", ""), hyp.get("label", None)))
                else:
                    hypotheses.append((str(hyp), None))

    if "output" in record and isinstance(record["output"], dict):
        output = record["output"]

        if task in ("nli", "interview_nli"):
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

    if not hypotheses and "hypothesis" in record:
        gold = record.get("gold", record.get("label", None))
        hypotheses.append((record["hypothesis"], gold))

    return hypotheses


def find_audio_path(audio_dir: str, file_name: str) -> Optional[str]:
    """Find audio file path, handling various filename formats."""
    if not file_name:
        return None
    
    basename = file_name
    for prefix in ["data/", "/data/", "Audio/", "/Audio/"]:
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
    
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


# ========== MODEL INTERFACE ==========

def init_model(cfg_path: str, device: str = "cuda:0"):
    """
    Initialize SALMONN model.
    
    Args:
        cfg_path: Path to SALMONN config file
        device: Device to load model on
    
    Returns:
        Dict with "model", "processor", "cfg", and "device" keys
    """
    # Create a minimal args object for Config
    class Args:
        def __init__(self):
            self.cfg_path = cfg_path
            self.options = None
    
    args = Args()
    cfg = Config(args)
    
    model = SALMONN.from_config(cfg.config.model)
    model.to(device)
    model.eval()
    
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    return {
        "model": model,
        "processor": wav_processor,
        "cfg": cfg,
        "device": device
    }


def generate_with_model(model_dict: Dict, audio_path: str, prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate text using SALMONN.
    
    Args:
        model_dict: Dict with "model", "processor", "cfg", and "device" keys
        audio_path: Path to audio file
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    """
    model = model_dict["model"]
    wav_processor = model_dict["processor"]
    cfg = model_dict["cfg"]
    device = model_dict["device"]
    
    # Prepare audio sample
    samples = prepare_one_sample(audio_path, wav_processor)
    
    # Format prompt using SALMONN's template
    formatted_prompt = [
        cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
    ]
    
    # Generate
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output_list = model.generate(samples, cfg.config.generate, prompts=formatted_prompt)
            output = output_list[0] if output_list else ""
    
    # Strip control characters and whitespace
    if output:
        # Remove control characters (like \u0003 ETX)
        output = ''.join(char for char in output if ord(char) >= 32 or char in '\n\t')
        # Strip leading/trailing whitespace
        output = output.strip()
        # Remove the prompt template if it appears in the output
        # The model might echo back the prompt, so extract only the response
        if "ASSISTANT:" in output:
            output = output.split("ASSISTANT:")[-1].strip()
    
    return output


# ========== MAIN INFERENCE LOOP ==========

def main():
    parser = argparse.ArgumentParser(description="Run SALMONN inference on JSONL files")
    
    # Model
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to SALMONN config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0, cuda:1, etc.)")
    
    # Input
    parser.add_argument("--jsonl_path", type=str, required=True, help="Input JSONL file with hypotheses")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["nli", "interview_nli", "consistency", "plausibility", "intent", "commonsense", "restraint", "accent_drift"],
                       help="Task type")
    
    # Output
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL file")
    
    # Limits
    parser.add_argument("--max_rows", type=int, default=-1, help="Limit number of rows (-1 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip items already in out_jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for generation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {args.jsonl_path}")
    if not os.path.isdir(args.audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {args.audio_dir}")
    if not os.path.isfile(args.cfg_path):
        raise FileNotFoundError(f"Config file not found: {args.cfg_path}")
    
    # Initialize model
    print(f"Initializing SALMONN from {args.cfg_path}...")
    model_dict = init_model(args.cfg_path, device=args.device)
    print("Model loaded successfully!")
    
    # Load prompt template
    if args.task not in PROMPTS:
        raise ValueError(f"Unknown task: {args.task}")
    prompt_template = PROMPTS[args.task]
    
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
    n_items = 0
    n_skipped = 0
    n_errors = 0
    
    with open(args.out_jsonl, jsonl_mode) as out_f:
        for record_idx, record in enumerate(records):
            file_name = record.get("file_name", "")
            if args.task == "interview_nli" and not file_name and "audio_id" in record:
                file_name = f"{record['audio_id']}.wav"
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
                    n_errors += 1
                    continue
                
                # Build prompt
                if args.task == "intent":
                    options = "\n".join([f"- {h}" for h, _ in hypotheses])
                    prompt = prompt_template.format(options=options, hypothesis=hypothesis)
                else:
                    prompt = prompt_template.format(hypothesis=hypothesis)
                
                try:
                    # Generate
                    text = generate_with_model(model_dict, audio_path, prompt, args.max_new_tokens)
                    
                    # Retry if empty
                    retry_count = 0
                    if not text or not text.strip():
                        retry_count = 1
                        # SALMONN doesn't have temperature control in the same way, so just retry
                        text = generate_with_model(model_dict, audio_path, prompt, args.max_new_tokens)
                    
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

