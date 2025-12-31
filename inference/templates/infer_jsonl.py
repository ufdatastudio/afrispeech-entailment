#!/usr/bin/env python3
"""
Generic JSONL-based audio entailment inference script.

Reads JSONL files with hypotheses and runs audio-language model inference.
Can be copied to each model's folder and customized.
"""
import json
import os
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import your model here - customize per model
# Example: from kimia_infer.api.kimia import KimiAudio
# Or: from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


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

Determine whether the statement is true based on the semantic content of the audio, regardless of accent or pronunciation.

Respond with one of the following labels only:
TRUE
FALSE

STATEMENT:
{hypothesis}""",
}


# ========== LABEL NORMALIZATION ==========

def normalize_label(text: str, task: str) -> str:
    """Extract canonical label from model output."""
    if not isinstance(text, str):
        return "UNPARSEABLE"
    
    t = text.strip().upper()
    
    if task == "nli":
        for lab in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
            if lab in t:
                return lab
    elif task == "consistency":
        for lab in ("CONSISTENT", "INCONSISTENT"):
            if lab in t:
                return lab
    elif task == "plausibility":
        for lab in ("PLAUSIBLE", "IMPLAUSIBLE"):
            if lab in t:
                return lab
    elif task == "intent":
        # Return as-is, or extract from options
        return t
    elif task == "commonsense":
        for lab in ("YES", "NO"):
            if lab in t:
                return lab
    elif task == "restraint":
        for lab in ("SUPPORTED", "UNSUPPORTED"):
            if lab in t:
                return lab
    elif task == "accent_drift":
        for lab in ("TRUE", "FALSE"):
            if lab in t:
                return lab
    
    return "UNPARSEABLE"


# ========== HYPOTHESIS EXTRACTION ==========

def extract_hypotheses(record: Dict, task: str) -> List[Tuple[str, str]]:
    """
    Extract (hypothesis, gold_label) pairs from JSONL record.
    
    Returns: List of (hypothesis_text, gold_label) tuples.
    """
    output = record.get("output", {})
    if not isinstance(output, dict):
        return []
    
    hypotheses = []
    
    if task == "nli":
        # Extract from entailment, neutral, contradiction lists
        for label in ["entailment", "neutral", "contradiction"]:
            hyps = output.get(label, [])
            if isinstance(hyps, list):
                for hyp in hyps:
                    if isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper()))
    
    elif task == "consistency":
        # Extract from consistent, inconsistent lists
        for label in ["consistent", "inconsistent"]:
            hyps = output.get(label, [])
            if isinstance(hyps, list):
                for hyp in hyps:
                    if isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper()))
    
    elif task == "plausibility":
        # Extract from plausible, implausible lists
        for label in ["plausible", "implausible"]:
            hyps = output.get(label, [])
            if isinstance(hyps, list):
                for hyp in hyps:
                    if isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper()))
    
    elif task == "intent":
        # Extract from intent list
        intents = output.get("intent", [])
        if isinstance(intents, list):
            for intent in intents:
                if isinstance(intent, str) and intent.strip():
                    hypotheses.append((intent.strip(), "INTENT"))
    
    elif task == "commonsense":
        # Extract from commonsense_inference list
        inferences = output.get("commonsense_inference", [])
        if isinstance(inferences, list):
            for inf in inferences:
                if isinstance(inf, str) and inf.strip():
                    # For commonsense, we might need to formulate as question
                    hypotheses.append((inf.strip(), "COMMONSENSE"))
    
    elif task == "restraint":
        # Extract from supported, unsupported lists
        for label in ["supported", "unsupported"]:
            hyps = output.get(label, [])
            if isinstance(hyps, list):
                for hyp in hyps:
                    if isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper()))
    
    elif task == "accent_drift":
        # Extract from accent_invariant, accent_sensitive_lures lists
        for label_type in ["accent_invariant", "accent_sensitive_lures"]:
            hyps = output.get(label_type, [])
            if isinstance(hyps, list):
                for hyp in hyps:
                    if isinstance(hyp, str) and hyp.strip():
                        # Both types test semantic understanding
                        hypotheses.append((hyp.strip(), "ACCENT_TEST"))
    
    return hypotheses


# ========== AUDIO FILE FINDING ==========

def find_audio_path(audio_dir: str, file_name: str) -> Optional[str]:
    """Find audio file, handling various path formats."""
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    
    # Remove common prefixes
    basename = file_name
    for prefix in ["data/", "/data/", "Audio/", "/Audio/"]:
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
    
    basename = os.path.basename(basename)
    
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
# Customize this section for your specific model

def init_model(model_path: str):
    """
    Initialize your model here.
    Return model object that has a generate() method.
    """
    # Example for Kimi:
    # from kimia_infer.api.kimia import KimiAudio
    # return KimiAudio(model_path=model_path, load_detokenizer=True)
    
    # Example for other models:
    # from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    # processor = AutoProcessor.from_pretrained(model_path)
    # return {"model": model, "processor": processor}
    
    raise NotImplementedError("Please implement init_model() for your model")


def generate_with_model(model, messages: List[Dict], sampling_params: Dict, max_new_tokens: int) -> str:
    """
    Generate text using your model.
    
    Args:
        model: Your model object
        messages: List of message dicts with role/content
        sampling_params: Model-specific sampling parameters
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text string
    """
    # Example for Kimi:
    # _, text = model.generate(messages, **sampling_params, output_type="text", max_new_tokens=max_new_tokens)
    # return text
    
    # Example for other models:
    # inputs = processor(messages, return_tensors="pt")
    # outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # return processor.decode(outputs[0], skip_special_tokens=True)
    
    raise NotImplementedError("Please implement generate_with_model() for your model")


# ========== MAIN INFERENCE LOOP ==========

def main():
    parser = argparse.ArgumentParser(description="Run audio entailment inference on JSONL files")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    
    # Input
    parser.add_argument("--jsonl_path", type=str, required=True, help="Input JSONL file with hypotheses")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["nli", "consistency", "plausibility", "intent", "commonsense", "restraint", "accent_drift"],
                       help="Task type")
    
    # Output
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--out_json", type=str, default=None, help="Optional: output JSON file")
    
    # Limits
    parser.add_argument("--max_rows", type=int, default=-1, help="Limit number of rows (-1 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip items already in out_jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for generation")
    
    # Model-specific sampling params (customize as needed)
    parser.add_argument("--temperature", type=float, default=0.0, help="Text generation temperature")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k sampling")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {args.jsonl_path}")
    if not os.path.isdir(args.audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {args.audio_dir}")
    
    # Initialize model
    print(f"Initializing model from {args.model_path}...")
    model = init_model(args.model_path)
    
    # Sampling params (customize per model)
    sampling_params = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        # Add model-specific params here
    }
    
    # Load JSONL
    print(f"Loading JSONL from {args.jsonl_path}...")
    records = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    
    if args.max_rows > 0:
        records = records[:args.max_rows]
    
    print(f"Loaded {len(records)} records")
    
    # Resume support
    seen_item_ids = set()
    if args.resume and os.path.isfile(args.out_jsonl):
        print(f"Resuming from {args.out_jsonl}...")
        with open(args.out_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "item_id" in obj:
                            seen_item_ids.add(obj["item_id"])
                    except:
                        pass
    
    # Get prompt template
    prompt_template = PROMPTS.get(args.task)
    if not prompt_template:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Open output file
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    jsonl_mode = "a" if args.resume and os.path.isfile(args.out_jsonl) else "w"
    
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
                    # Intent needs options - extract from hypotheses
                    options = "\n".join([f"- {h}" for h, _ in hypotheses])
                    prompt = prompt_template.format(options=options, hypothesis=hypothesis)
                else:
                    prompt = prompt_template.format(hypothesis=hypothesis)
                
                # Build messages (customize per model API)
                messages = [
                    {"role": "user", "message_type": "text", "content": prompt},
                    {"role": "user", "message_type": "audio", "content": audio_path},
                ]
                
                try:
                    # Generate
                    text = generate_with_model(model, messages, sampling_params, args.max_new_tokens)
                    
                    # Retry if empty
                    retry_count = 0
                    if not text or not text.strip():
                        retry_count = 1
                        retry_params = sampling_params.copy()
                        retry_params["temperature"] = 0.0
                        retry_params["top_k"] = 1
                        text = generate_with_model(model, messages, retry_params, args.max_new_tokens)
                    
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
    
    # Write JSON snapshot
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
    
    print(
        f"\nDone.\n"
        f"- Saved JSONL: {args.out_jsonl}\n"
        f"- Saved JSON: {args.out_json or 'N/A'}\n"
        f"- Items processed: {n_items}\n"
        f"- Items skipped (resume): {n_skipped}\n"
        f"- Errors: {n_errors}\n"
    )


if __name__ == "__main__":
    main()

