#!/usr/bin/env python3
"""
Run cascade baseline inference: ASR transcript + hypothesis -> text LLM label.

This script:
1) Reads task JSONL (contains file_name + hypotheses in `output`)
2) Looks up ASR transcript by file_name from simple ASR JSONL
3) Prompts a text-only LLM (Llama/Qwen/Mistral) with transcript + statement
4) Writes standardized JSONL predictions for downstream evaluation

Supported tasks:
- nli
- consistency
- plausibility
- restraint
- accent_drift
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = {
    "nli": """You are given a speech transcript and a statement.

Based only on the transcript, determine whether the statement is:
- ENTAILMENT
- CONTRADICTION
- NEUTRAL

Output exactly one label:
ENTAILMENT or CONTRADICTION or NEUTRAL

Transcript:
{transcript}

Statement:
{hypothesis}
""",
    "consistency": """You are given a speech transcript and a statement.

Determine whether the statement is consistent with the transcript meaning.
Output exactly one label:
CONSISTENT or INCONSISTENT

Transcript:
{transcript}

Statement:
{hypothesis}
""",
    "plausibility": """You are given a speech transcript and a statement.

Is the statement plausible given the transcript?
Output exactly one label:
PLAUSIBLE or IMPLAUSIBLE

Transcript:
{transcript}

Statement:
{hypothesis}
""",
    "restraint": """You are given a speech transcript and a statement.

Based only on transcript evidence, determine whether the statement is supported.
Output exactly one label:
SUPPORTED or UNSUPPORTED

Transcript:
{transcript}

Statement:
{hypothesis}
""",
    "accent_drift": """You are given a speech transcript and a statement.

Determine whether the statement is semantically true with respect to transcript content.
Output exactly one label:
TRUE or FALSE

Transcript:
{transcript}

Statement:
{hypothesis}
""",
}


def _match_label(text: str, labels: Tuple[str, ...]) -> Optional[str]:
    for lab in sorted(labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(lab)}\b", text):
            return lab
    return None


def normalize_label(text: str, task: str) -> str:
    if not isinstance(text, str):
        return "UNPARSEABLE"
    t = text.upper().strip()

    if task == "nli":
        m = _match_label(t, ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"))
    elif task == "consistency":
        m = _match_label(t, ("INCONSISTENT", "CONSISTENT"))
    elif task == "plausibility":
        m = _match_label(t, ("IMPLAUSIBLE", "PLAUSIBLE"))
    elif task == "restraint":
        m = _match_label(t, ("UNSUPPORTED", "SUPPORTED"))
    elif task == "accent_drift":
        m = _match_label(t, ("FALSE", "TRUE"))
    else:
        m = None

    return m if m else "UNPARSEABLE"


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def extract_hypotheses(record: Dict, task: str) -> List[Tuple[str, str, Optional[str]]]:
    output = record.get("output", {})
    if not isinstance(output, dict):
        return []

    hypotheses: List[Tuple[str, str, Optional[str]]] = []

    if task == "nli":
        for label in ["entailment", "contradiction", "neutral"]:
            items = output.get(label, [])
            if isinstance(items, list):
                for hyp in items:
                    if isinstance(hyp, dict):
                        text = str(hyp.get("hypothesis", "")).strip()
                        difficulty = str(hyp.get("difficulty", "")).strip() or None
                        if text:
                            hypotheses.append((text, label.upper(), difficulty))
                    elif isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper(), None))

    elif task == "consistency":
        for label in ["consistent", "inconsistent"]:
            items = output.get(label, [])
            if isinstance(items, list):
                for hyp in items:
                    if isinstance(hyp, dict):
                        text = str(hyp.get("hypothesis", "")).strip()
                        difficulty = str(hyp.get("difficulty", "")).strip() or None
                        if text:
                            hypotheses.append((text, label.upper(), difficulty))
                    elif isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper(), None))

    elif task == "plausibility":
        for label in ["plausible", "implausible"]:
            items = output.get(label, [])
            if isinstance(items, list):
                for hyp in items:
                    if isinstance(hyp, dict):
                        text = str(hyp.get("hypothesis", "")).strip()
                        difficulty = str(hyp.get("difficulty", "")).strip() or None
                        if text:
                            hypotheses.append((text, label.upper(), difficulty))
                    elif isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper(), None))

    elif task == "restraint":
        for label in ["supported", "unsupported"]:
            items = output.get(label, [])
            if isinstance(items, list):
                for hyp in items:
                    if isinstance(hyp, dict):
                        text = str(hyp.get("hypothesis", "")).strip()
                        difficulty = str(hyp.get("difficulty", "")).strip() or None
                        if text:
                            hypotheses.append((text, label.upper(), difficulty))
                    elif isinstance(hyp, str) and hyp.strip():
                        hypotheses.append((hyp.strip(), label.upper(), None))

    elif task == "accent_drift":
        for hyp in output.get("accent_invariant", []) if isinstance(output.get("accent_invariant", []), list) else []:
            if isinstance(hyp, dict):
                text = str(hyp.get("hypothesis", "")).strip()
                difficulty = str(hyp.get("difficulty", "")).strip() or None
                if text:
                    hypotheses.append((text, "TRUE", difficulty))
            elif isinstance(hyp, str) and hyp.strip():
                hypotheses.append((hyp.strip(), "TRUE", None))

        for hyp in output.get("accent_sensitive_lures", []) if isinstance(output.get("accent_sensitive_lures", []), list) else []:
            if isinstance(hyp, dict):
                text = str(hyp.get("hypothesis", "")).strip()
                difficulty = str(hyp.get("difficulty", "")).strip() or None
                if text:
                    hypotheses.append((text, "FALSE", difficulty))
            elif isinstance(hyp, str) and hyp.strip():
                hypotheses.append((hyp.strip(), "FALSE", None))

    return hypotheses


def load_asr_map(asr_jsonl: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(asr_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("file_name", "")).strip()
            text = str(row.get("asr_text", "")).strip()
            if key and text:
                mapping[key] = text
    return mapping


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful classifier. Return only the requested label."},
        {"role": "user", "content": prompt},
    ]


def generate_label(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        model_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        model_input = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    inputs = tokenizer(model_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-LLM cascade inference using ASR transcripts")
    parser.add_argument("--model_id", required=True, help="HF model id")
    parser.add_argument("--task", required=True, choices=list(PROMPTS.keys()))
    parser.add_argument("--task_jsonl", required=True, help="Task hypotheses JSONL")
    parser.add_argument("--asr_jsonl", required=True, help="ASR simple JSONL with file_name/asr_text")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_rows", type=int, default=-1)
    return parser.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    print(f"Loading ASR map: {args.asr_jsonl}")
    asr_map = load_asr_map(args.asr_jsonl)
    print(f"ASR rows loaded: {len(asr_map)}")

    print(f"Loading model: {args.model_id}")
    torch_dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    seen = set()
    if args.resume and os.path.isfile(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if "item_id" in obj:
                        seen.add(obj["item_id"])

    with open(args.task_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.max_rows > 0:
        records = records[: args.max_rows]

    mode = "a" if args.resume and os.path.isfile(args.output_jsonl) else "w"

    processed = 0
    skipped = 0
    errors = 0

    with open(args.output_jsonl, mode, encoding="utf-8") as out_f:
        for ridx, record in enumerate(records):
            file_name = str(record.get("file_name", "")).strip()
            transcript = asr_map.get(file_name, "")

            hypotheses = extract_hypotheses(record, args.task)
            for hidx, (hypothesis, gold, difficulty) in enumerate(hypotheses):
                item_id = f"{Path(file_name).stem or f'row{ridx}'}__hyp_{hidx}"

                if item_id in seen:
                    skipped += 1
                    continue

                if not transcript:
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "hypothesis": hypothesis,
                        "difficulty": difficulty,
                        "gold": gold,
                        "pred_raw": None,
                        "pred": None,
                        "error": "ASR transcript not found for file_name",
                        "task": args.task,
                        "model": args.model_id,
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    out_f.flush()
                    errors += 1
                    continue

                prompt = PROMPTS[args.task].format(transcript=transcript, hypothesis=hypothesis)
                messages = build_messages(prompt)

                try:
                    pred_raw = generate_label(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    pred = normalize_label(pred_raw, args.task)

                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "hypothesis": hypothesis,
                        "difficulty": difficulty,
                        "gold": gold,
                        "pred_raw": pred_raw,
                        "pred": pred,
                        "error": None,
                        "task": args.task,
                        "model": args.model_id,
                        "asr_source": os.path.basename(args.asr_jsonl),
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed += 1

                except Exception as exc:
                    obj = {
                        "item_id": item_id,
                        "file_name": file_name,
                        "hypothesis": hypothesis,
                        "difficulty": difficulty,
                        "gold": gold,
                        "pred_raw": None,
                        "pred": None,
                        "error": str(exc),
                        "task": args.task,
                        "model": args.model_id,
                        "asr_source": os.path.basename(args.asr_jsonl),
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    out_f.flush()
                    errors += 1

            if (ridx + 1) % 25 == 0:
                print(f"Processed records={ridx+1}, items={processed}, skipped={skipped}, errors={errors}")

    print(f"Done: items={processed}, skipped={skipped}, errors={errors}")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
