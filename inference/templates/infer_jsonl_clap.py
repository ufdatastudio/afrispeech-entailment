#!/usr/bin/env python3
"""
CLAP (LAION-AI/CLAP) inference script for JSONL-based audio entailment tasks.

CLAP is a contrastive model that provides audio and text embeddings.
For NLI-style tasks, we compute cosine similarity between audio embedding
and each hypothesis text embedding, then pick the highest similarity
as the predicted label.
"""
import json
import os
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import laion_clap


# ========== LABEL MAPPING ==========

LABELS = {
    "nli": ["entailment", "neutral", "contradiction"],
    "consistency": ["consistent", "inconsistent"],
    "plausibility": ["plausible", "implausible"],
    "restraint": ["supported", "unsupported"],
    "accent_drift": ["true", "false"],
}

# For tasks that don't fit the 3-label structure, we'll handle them separately
TASK_LABEL_MAP = {
    "intent": None,  # Variable number of options
    "commonsense": ["yes", "no"],
}


# ========== PROMPT TEMPLATES ==========

PROMPT_TEMPLATES = {
    "nli": "Given the audio, the following statement is true: {hyp}",
    "consistency": "Given the audio, the following statement is consistent: {hyp}",
    "plausibility": "Given the audio, the following statement is plausible: {hyp}",
    "restraint": "Given the audio, the following statement is supported: {hyp}",
    "accent_drift": "Given the audio, the following statement is true: {hyp}",
    "commonsense": "Given the audio, answer this question: {hyp}",
    "intent": "{hyp}",  # Intent usually doesn't need a template
}


# ========== UTILITY FUNCTIONS ==========

def int16_to_float32(x):
    """Convert int16 audio to float32."""
    return (x / 32767.0).astype("float32")


def load_audio_48k(path: str, target_sr: int = 48000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


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


# ========== HYPOTHESIS EXTRACTION ==========

def extract_hypotheses(record: Dict, task: str) -> List[Tuple[str, str]]:
    """
    Extract (hypothesis, gold_label) pairs from JSONL record.
    
    Returns: List of (hypothesis_text, gold_label) tuples.
    """
    hypotheses = []
    
    # Check if record has 'output' field with structured hypotheses (existing format)
    if "output" in record and isinstance(record["output"], dict):
        output = record["output"]
        
        if task == "nli":
            for key in ["entailment", "contradiction", "neutral"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "consistency":
            for key in ["consistent", "inconsistent"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "plausibility":
            for key in ["plausible", "implausible"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "restraint":
            for key in ["supported", "unsupported"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "accent_drift":
            # accent_invariant should be TRUE, accent_sensitive_lures should be FALSE
            accent_invariant_hyps = output.get("accent_invariant", [])
            if isinstance(accent_invariant_hyps, list):
                for hyp in accent_invariant_hyps:
                    if isinstance(hyp, dict):
                        hyp_text = hyp.get("hypothesis", "")
                    else:
                        hyp_text = str(hyp)
                    if hyp_text and hyp_text.strip():
                        hypotheses.append((hyp_text.strip(), "TRUE"))
            
            accent_sensitive_hyps = output.get("accent_sensitive_lures", [])
            if isinstance(accent_sensitive_hyps, list):
                for hyp in accent_sensitive_hyps:
                    if isinstance(hyp, dict):
                        hyp_text = hyp.get("hypothesis", "")
                    else:
                        hyp_text = str(hyp)
                    if hyp_text and hyp_text.strip():
                        hypotheses.append((hyp_text.strip(), "FALSE"))
        
        elif task == "commonsense":
            inferences = output.get("commonsense_inference", [])
            if isinstance(inferences, list):
                for inf in inferences:
                    if isinstance(inf, dict):
                        hyp_text = inf.get("hypothesis", "")
                    else:
                        hyp_text = str(inf)
                    if hyp_text and hyp_text.strip():
                        hypotheses.append((hyp_text.strip(), "COMMONSENSE"))
        
        elif task == "intent":
            intents = output.get("intent", [])
            if isinstance(intents, list):
                for intent in intents:
                    if isinstance(intent, dict):
                        hyp_text = intent.get("hypothesis", "")
                    else:
                        hyp_text = str(intent)
                    if hyp_text and hyp_text.strip():
                        hypotheses.append((hyp_text.strip(), "INTENT"))
    
    # Also support direct format (user's template format)
    elif "hypotheses" in record and isinstance(record["hypotheses"], dict):
        hyps_dict = record["hypotheses"]
        for label, hyp_text in hyps_dict.items():
            if isinstance(hyp_text, str) and hyp_text.strip():
                hypotheses.append((hyp_text.strip(), label.upper()))
    
    return hypotheses


def build_text_candidates(prompt_template: str, hypotheses: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Build text candidates with prompt template applied.
    
    Returns: List of (formatted_text, label) tuples.
    """
    candidates = []
    for hyp_text, label in hypotheses:
        formatted = prompt_template.format(hyp=hyp_text)
        candidates.append((formatted, label))
    return candidates


# ========== MAIN INFERENCE ==========

def main():
    parser = argparse.ArgumentParser(description="Run CLAP inference on JSONL files")
    
    # Model
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to CLAP checkpoint (optional, uses default if not provided)")
    
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
    
    # CLAP-specific
    parser.add_argument("--prompt_template", type=str, default=None,
                       help="Custom prompt template (overrides default for task)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--enable_fusion", action="store_true", help="Enable CLAP fusion mode")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {args.jsonl_path}")
    if not os.path.isdir(args.audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {args.audio_dir}")
    
    # Initialize CLAP model
    device = "cuda" if args.cuda else "cpu"
    print(f"Initializing CLAP model on {device}...")
    model = laion_clap.CLAP_Module(enable_fusion=args.enable_fusion, device=device)
    
    if args.model_path:
        model.load_ckpt(args.model_path)
        print(f"Loaded checkpoint from {args.model_path}")
    else:
        model.load_ckpt()  # Downloads default checkpoint
        print("Loaded default CLAP checkpoint")
    
    # Get prompt template
    if args.prompt_template:
        prompt_template = args.prompt_template
    else:
        prompt_template = PROMPT_TEMPLATES.get(args.task, "{hyp}")
    
    print(f"Using prompt template: {prompt_template}")
    
    # Get task labels
    task_labels = LABELS.get(args.task)
    if not task_labels and args.task in TASK_LABEL_MAP:
        task_labels = TASK_LABEL_MAP[args.task]
    
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
        print(f"Found {len(seen_item_ids)} existing items")
    
    # Create output directory
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
            
            # Also support direct audio_path in record
            if not audio_path and "audio_path" in record:
                audio_path = record["audio_path"]
                if audio_path and not os.path.isfile(audio_path):
                    audio_path = None
            
            base_id = os.path.splitext(os.path.basename(file_name))[0] if file_name else f"record_{record_idx}"
            
            # Extract hypotheses
            hypotheses = extract_hypotheses(record, args.task)
            
            if not hypotheses:
                print(f"Warning: No hypotheses found for record {record_idx}")
                continue
            
            if not audio_path:
                # No audio - write error for all hypotheses
                for hyp_idx, (hyp_text, gold_label) in enumerate(hypotheses):
                    item_id = f"{base_id}__hyp_{hyp_idx}"
                    if item_id in seen_item_ids:
                        n_skipped += 1
                        continue
                    
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": None,
                        "hypothesis": hyp_text,
                        "gold": gold_label,
                        "pred": None,
                        "scores": {},
                        "error": f"Audio not found for {file_name}",
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    all_results[item_id] = obj
                    n_errors += 1
                continue
            
            try:
                # Load audio and get embedding (once per audio)
                wav = load_audio_48k(audio_path)
                audio_emb = model.get_audio_embedding_from_data(x=[wav], use_tensor=False)
                audio_emb = np.asarray(audio_emb).reshape(-1)
                
                # Process each hypothesis individually
                for hyp_idx, (hyp_text, gold_label) in enumerate(hypotheses):
                    item_id = f"{base_id}__hyp_{hyp_idx}"
                    
                    if item_id in seen_item_ids:
                        n_skipped += 1
                        continue
                    
                    # For tasks with label templates (nli, consistency, plausibility, restraint, accent_drift)
                    # Compute similarity against all possible label templates
                    if args.task in LABELS:
                        possible_labels = LABELS[args.task]
                        
                        # Create label-specific templates for this hypothesis
                        label_templates = []
                        for label in possible_labels:
                            if args.task == "consistency":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            elif args.task == "plausibility":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            elif args.task == "restraint":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            elif args.task == "accent_drift":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            elif args.task == "nli":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            else:
                                # Fallback to using prompt template
                                label_template = prompt_template.format(hyp=hyp_text)
                            label_templates.append(label_template)
                        
                        # Get text embeddings for all label templates
                        text_embs = model.get_text_embedding(label_templates, use_tensor=False)
                        text_embs = [np.asarray(t).reshape(-1) for t in text_embs]
                        
                        # Compute similarities
                        sims = [cosine_sim(audio_emb, t) for t in text_embs]
                        
                        # Create scores dict mapping label names (uppercase) to scores
                        score_dict = {label.upper(): float(sims[i]) for i, label in enumerate(possible_labels)}
                        
                        # Find best label (highest similarity)
                        best_label_idx = int(np.argmax(sims))
                        pred_label = possible_labels[best_label_idx].upper()
                    
                    else:
                        # For tasks without label templates (intent, commonsense), use original approach
                        candidates = build_text_candidates(prompt_template, [(hyp_text, gold_label)])
                        candidate_texts = [cand[0] for cand in candidates]
                        text_embs = model.get_text_embedding(candidate_texts, use_tensor=False)
                        text_embs = [np.asarray(t).reshape(-1) for t in text_embs]
                        sims = [cosine_sim(audio_emb, t) for t in text_embs]
                        score_dict = {candidates[i][1]: float(sims[i]) for i in range(len(candidates))}
                        best_label_idx = int(np.argmax(sims))
                        pred_label = candidates[best_label_idx][1]
                    
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": audio_path,
                        "hypothesis": hyp_text,
                        "gold": gold_label,
                        "pred": pred_label,
                        "pred_raw": None,  # CLAP doesn't generate text
                        "scores": score_dict,
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
                print(f"Error processing {base_id}: {err}")
                
                # Write error for all hypotheses
                for hyp_idx, (hyp_text, gold_label) in enumerate(hypotheses):
                    item_id = f"{base_id}__hyp_{hyp_idx}"
                    if item_id in seen_item_ids:
                        n_skipped += 1
                        continue
                    
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "audio_path": audio_path,
                        "hypothesis": hyp_text,
                        "gold": gold_label,
                        "pred": None,
                        "scores": {},
                        "error": err,
                        "traceback": tb,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj) + "\n")
                    out_f.flush()
                    all_results[item_id] = obj
                    n_errors += 1
    
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

