#!/usr/bin/env python3
"""
CLAP (LAION-AI/CLAP) inference script for NLI-style tasks.

CLAP is not a generator; it gives audio embedding and text embedding.
For NLI-style setup, treat each hypothesis as a candidate text and pick
the highest similarity as the predicted label.

Usage:
    python run_clap_nli.py \
        --in_jsonl /path/to/input.jsonl \
        --out_jsonl /path/to/output.jsonl \
        --prompt_template "Given the audio, evaluate this statement: {hyp}" \
        --cuda
"""
import argparse
import json
from typing import Dict, Any, Iterable, List

import numpy as np
import librosa
import laion_clap


LABELS = ["entailment", "neutral", "contradiction"]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Read JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL file line by line."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def int16_to_float32(x):
    """Convert int16 audio to float32."""
    return (x / 32767.0).astype("float32")


def load_audio_48k(path: str, target_sr: int = 48000) -> np.ndarray:
    """Load audio file and resample to target sample rate (48k for CLAP)."""
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    return wav.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def build_text_candidates(prompt_template: str, hyps: Dict[str, str]) -> List[str]:
    """Build text candidates by applying prompt template to each hypothesis."""
    return [prompt_template.format(hyp=hyps[lab]) for lab in LABELS]


def main():
    ap = argparse.ArgumentParser(description="Run CLAP NLI inference on JSONL files")
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL file")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL file")
    ap.add_argument(
        "--prompt_template",
        default="Given the audio, evaluate this statement: {hyp}",
        help="Template applied to each hypothesis text.",
    )
    ap.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    ap.add_argument("--model_path", type=str, default=None, 
                   help="Path to CLAP checkpoint (optional, uses default if not provided)")
    ap.add_argument("--enable_fusion", action="store_true", 
                   help="Enable CLAP fusion mode")
    args = ap.parse_args()

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

    out_rows = []
    for ex in read_jsonl(args.in_jsonl):
        ex_id = ex.get("id")
        audio_path = ex.get("audio_path")
        hyps = ex.get("hypotheses", {})

        if not audio_path:
            print(f"Warning: No audio_path found for example {ex_id}, skipping")
            continue

        if not hyps or not isinstance(hyps, dict):
            print(f"Warning: No hypotheses found for example {ex_id}, skipping")
            continue

        # Ensure all labels are present
        missing_labels = [lab for lab in LABELS if lab not in hyps]
        if missing_labels:
            print(f"Warning: Missing hypotheses for labels {missing_labels} in example {ex_id}")

        # 1) Get audio embedding
        try:
            wav = load_audio_48k(audio_path)
            audio_emb = model.get_audio_embedding_from_data(x=wav, use_tensor=False)
            audio_emb = np.asarray(audio_emb).reshape(-1)
        except Exception as e:
            print(f"Error loading audio for {ex_id}: {e}")
            out_rows.append({
                "id": ex_id,
                "model_name": "clap_laion",
                "audio_path": audio_path,
                "prompt_template": args.prompt_template,
                "pred_label": None,
                "scores": {},
                "hypotheses": hyps,
                "error": str(e),
            })
            continue

        # 2) Get text embeddings for hypotheses (with prompt template)
        candidates = build_text_candidates(args.prompt_template, hyps)
        try:
            text_embs = model.get_text_embedding(candidates, use_tensor=False)
            text_embs = [np.asarray(t).reshape(-1) for t in text_embs]
        except Exception as e:
            print(f"Error getting text embeddings for {ex_id}: {e}")
            out_rows.append({
                "id": ex_id,
                "model_name": "clap_laion",
                "audio_path": audio_path,
                "prompt_template": args.prompt_template,
                "pred_label": None,
                "scores": {},
                "hypotheses": hyps,
                "error": str(e),
            })
            continue

        # 3) Compute similarities and pick best match
        sims = [cosine_sim(audio_emb, t) for t in text_embs]
        best_i = int(np.argmax(sims))
        pred_label = LABELS[best_i]

        out_rows.append({
            "id": ex_id,
            "model_name": "clap_laion",
            "audio_path": audio_path,
            "prompt_template": args.prompt_template,
            "pred_label": pred_label,
            "scores": {LABELS[i]: float(sims[i]) for i in range(len(LABELS))},
            "hypotheses": hyps,
        })

    # Write output
    write_jsonl(args.out_jsonl, out_rows)
    print(f"Processed {len(out_rows)} examples. Results written to {args.out_jsonl}")


if __name__ == "__main__":
    main()

