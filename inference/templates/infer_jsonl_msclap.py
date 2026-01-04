#!/usr/bin/env python3
"""
MSCLAP (Microsoft CLAP) inference script for JSONL-based audio entailment tasks.

MSCLAP is a contrastive model that provides audio and text embeddings.
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
from msclap import CLAP


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
            for key in ["true", "false"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            hypotheses.append((hyp_text.strip(), key.upper()))
        
        elif task == "commonsense":
            question = output.get("question", "")
            for key in ["yes", "no"]:
                if key in output:
                    hyps = output[key] if isinstance(output[key], list) else [output[key]]
                    for hyp in hyps:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            full_hyp = f"{question} {hyp_text.strip()}" if question else hyp_text.strip()
                            hypotheses.append((full_hyp, key.upper()))
        
        elif task == "intent":
            for key, val in output.items():
                if isinstance(val, list):
                    for hyp in val:
                        if isinstance(hyp, dict):
                            hyp_text = hyp.get("hypothesis", "")
                        else:
                            hyp_text = str(hyp)
                        if hyp_text and hyp_text.strip():
                            hypotheses.append((hyp_text.strip(), key.upper()))
    
    # Check for direct hypothesis field (processed format)
    elif "hypothesis" in record:
        hyp = record["hypothesis"].strip()
        gold = record.get("gold", "UNKNOWN").strip()
        if hyp:
            hypotheses.append((hyp, gold))
    
    return hypotheses


# ========== MODEL INTERFACE ==========

def init_model(version: str = "2023", use_cuda: bool = True):
    """
    Initialize MSCLAP model.
    
    Args:
        version: MSCLAP version ('2022' or '2023')
        use_cuda: Whether to use CUDA
    
    Returns:
        CLAP model instance
    """
    print(f"Initializing MSCLAP model (version: {version}, cuda: {use_cuda})...")
    model = CLAP(version=version, use_cuda=use_cuda)
    print("Model loaded successfully!")
    return model


def predict(
    model,
    audio_path: str,
    hypotheses: List[str],
    task: str,
    template: str = None
) -> Tuple[str, List[float]]:
    """
    Predict the most likely hypothesis for given audio using MSCLAP.
    
    Args:
        model: MSCLAP model instance
        audio_path: Path to audio file
        hypotheses: List of hypothesis texts
        task: Task name
        template: Optional prompt template
    
    Returns:
        (predicted_label, similarity_scores)
    """
    # Format hypotheses with template if provided
    if template:
        formatted_hyps = [template.format(hyp=h) for h in hypotheses]
    else:
        formatted_hyps = hypotheses
    
    # Get embeddings
    text_embeddings = model.get_text_embeddings(formatted_hyps)
    audio_embeddings = model.get_audio_embeddings([audio_path])
    
    # Compute similarities
    similarities = model.compute_similarity(audio_embeddings, text_embeddings)
    
    # Get scores for the single audio (first row)
    scores = similarities[0].detach().cpu().numpy() if hasattr(similarities[0], 'cpu') else similarities[0]
    
    # Find best match
    best_idx = int(np.argmax(scores))
    predicted_label = hypotheses[best_idx]
    
    return predicted_label, scores.tolist()


# ========== MAIN INFERENCE ==========

def main():
    parser = argparse.ArgumentParser(description="MSCLAP inference for audio entailment tasks")
    parser.add_argument("--version", type=str, default="2023", choices=["2022", "2023"],
                        help="MSCLAP version")
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--out_jsonl", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--task", type=str, required=True,
                        choices=["nli", "consistency", "plausibility", "intent", 
                                "commonsense", "restraint", "accent_drift"],
                        help="Task type")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MSCLAP Inference")
    print("="*60)
    print(f"Version: {args.version}")
    print(f"Task: {args.task}")
    print(f"Input: {args.jsonl_path}")
    print(f"Audio: {args.audio_dir}")
    print(f"Output: {args.out_jsonl}")
    print(f"Device: {'CPU' if args.use_cpu else 'GPU'}")
    print("="*60)
    print()
    
    # Initialize model
    model = init_model(version=args.version, use_cuda=not args.use_cpu)
    
    # Load input JSONL
    print(f"Loading input from {args.jsonl_path}...")
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} entries")
    
    # Get prompt template
    template = PROMPT_TEMPLATES.get(args.task)
    
    # Process each record
    results = []
    errors = []
    processed_count = 0
    
    for record in records:
        try:
            file_name = record.get("file_name", "")
            audio_path = find_audio_path(args.audio_dir, file_name)
            
            if not audio_path:
                errors.append({
                    "record": record,
                    "error": f"Audio file not found: {file_name}"
                })
                continue
            
            # Extract hypotheses
            hypotheses = extract_hypotheses(record, args.task)
            
            if not hypotheses:
                errors.append({
                    "record": record,
                    "error": "No hypotheses found"
                })
                continue
            
            # Process each hypothesis
            for hyp_text, gold_label in hypotheses:
                processed_count += 1
                
                # For each hypothesis, compare audio against all possible label templates
                if args.task in LABELS:
                    # Get all possible labels for this task
                    possible_labels = LABELS[args.task]
                    
                    # Create formatted text for each label option
                    # E.g., for consistency task with hypothesis "X is true":
                    # - "Given the audio, the following statement is consistent: X is true"
                    # - "Given the audio, the following statement is inconsistent: X is true"
                    label_templates = []
                    for label in possible_labels:
                        if template:
                            # Use label-specific template
                            if args.task == "consistency":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            elif args.task == "plausibility":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            elif args.task == "nli":
                                label_template = f"Given the audio, the following statement is {label}: {hyp_text}"
                            else:
                                label_template = template.format(hyp=hyp_text)
                        else:
                            label_template = hyp_text
                        label_templates.append(label_template)
                    
                    # Get similarities for all label options
                    _, all_scores = predict(model, audio_path, label_templates, args.task, None)
                    
                    # Create dict mapping label names to scores
                    score_dict = {label.upper(): score for label, score in zip(possible_labels, all_scores)}
                    
                    # Determine prediction based on highest score
                    best_label = max(score_dict.items(), key=lambda x: x[1])[0]
                    
                    results.append({
                        "hypothesis": hyp_text,
                        "gold": gold_label,
                        "pred": best_label,
                        "pred_raw": None,
                        "scores": score_dict
                    })
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} hypotheses...")
        
        except Exception as e:
            errors.append({
                "record": record,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            print(f"Error processing record: {e}")
    
    print()
    
    # Save results
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    print(f"Writing output to {args.out_jsonl}...")
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print("="*60)
    print("Inference Complete!")
    print(f"Total: {processed_count}")
    print(f"Success: {len(results)}")
    print(f"Errors: {len(errors)}")
    print("="*60)
    
    # Save errors if any
    if errors:
        error_path = args.out_jsonl.replace(".jsonl", "_errors.jsonl")
        with open(error_path, "w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
        print(f"Errors saved to: {error_path}")


if __name__ == "__main__":
    main()



