#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--audio_col", default="audio_file")
    parser.add_argument("--audio_root", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--simple_jsonl", action="store_true")
    parser.add_argument("--model_id", default="openai/whisper-large-v3")
    parser.add_argument("--language", default="english")
    parser.add_argument("--chunk_length_s", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_audio_path(raw_path: str, audio_root: Optional[Path]) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    if audio_root is not None:
        rooted = (audio_root / candidate).resolve()
        if rooted.exists():
            return rooted
        basename = candidate.name
        rooted_basename = (audio_root / basename).resolve()
        if rooted_basename.exists():
            return rooted_basename
    raise FileNotFoundError(f"Audio file not found: {raw_path}")


def build_pipe(model_id: str, language: str, max_new_tokens: int, num_beams: int, do_sample: bool):
    has_cuda = torch.cuda.is_available()
    torch_device = "cuda:0" if has_cuda else "cpu"
    device_id = 0 if has_cuda else -1
    torch_dtype = torch.float16 if has_cuda else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(torch_device)
    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device_id,
        generate_kwargs={
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "repetition_penalty": 1.0,
            "language": language,
        },
    )


def main() -> None:
    args = parse_args()

    if (args.input_csv is None) == (args.input_jsonl is None):
        raise ValueError("Provide exactly one of --input_csv or --input_jsonl")
    if args.output_csv is None and args.output_jsonl is None:
        raise ValueError("Provide at least one output: --output_csv or --output_jsonl")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install/load ffmpeg on this node before running Whisper ASR."
        )

    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    input_jsonl = Path(args.input_jsonl).resolve() if args.input_jsonl else None
    output_csv = Path(args.output_csv).resolve() if args.output_csv else None
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else None
    audio_root = Path(args.audio_root).resolve() if args.audio_root else None

    if input_csv is not None:
        df = pd.read_csv(input_csv)
        input_source = str(input_csv)
    else:
        records = []
        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        input_source = str(input_jsonl)

    if args.audio_col not in df.columns:
        raise ValueError(f"Column '{args.audio_col}' not found in {input_source}")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    asr_pipe = build_pipe(
        model_id=args.model_id,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
    )

    asr_texts = []
    statuses = []
    errors = []

    for raw_path in tqdm(df[args.audio_col].tolist(), desc="Whisper ASR"):
        try:
            resolved = resolve_audio_path(raw_path, audio_root)
            result = asr_pipe(
                str(resolved),
                return_timestamps=True,
                chunk_length_s=args.chunk_length_s,
                generate_kwargs={"language": args.language},
            )
            asr_texts.append((result.get("text") or "").strip())
            statuses.append("ok")
            errors.append("")
        except Exception as exc:
            asr_texts.append("")
            statuses.append("error")
            errors.append(str(exc))

    original_audio = df[args.audio_col].astype(str).tolist()
    df = df.copy()
    df["asr_model"] = args.model_id
    df["asr_text"] = asr_texts
    df["asr_status"] = statuses
    df["asr_error"] = errors

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            if args.simple_jsonl:
                for i, asr_text in enumerate(asr_texts):
                    record = {
                        args.audio_col: original_audio[i],
                        "asr_model": args.model_id,
                        "asr_text": asr_text,
                        "asr_status": statuses[i],
                        "asr_error": errors[i],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                for record in df.to_dict(orient="records"):
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok_count = int((df["asr_status"] == "ok").sum())
    total = len(df)
    if output_csv is not None:
        print(f"Saved: {output_csv}")
    if output_jsonl is not None:
        print(f"Saved: {output_jsonl}")
    print(f"Completed {ok_count}/{total} successfully")


if __name__ == "__main__":
    main()