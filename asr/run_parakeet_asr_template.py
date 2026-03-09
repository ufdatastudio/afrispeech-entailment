#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--audio_col", default="audio_file")
    parser.add_argument("--audio_root", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--simple_jsonl", action="store_true")
    parser.add_argument("--model_id", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--language", default="en")
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
        rooted_basename = (audio_root / candidate.name).resolve()
        if rooted_basename.exists():
            return rooted_basename
    raise FileNotFoundError(f"Audio file not found: {raw_path}")


def init_parakeet(model_id: str):
    try:
        import huggingface_hub as hfh
        if not hasattr(hfh, "ModelFilter"):
            class ModelFilter:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs
            hfh.ModelFilter = ModelFilter
    except Exception:
        pass

    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        raise ImportError("Install NeMo first: pip install nemo_toolkit[asr]") from exc

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    return model


def transcribe_parakeet(model, audio_path: Path, language: str) -> str:
    _ = language
    predictions = model.transcribe([str(audio_path)])
    if isinstance(predictions, list) and len(predictions) > 0:
        first = predictions[0]
        if hasattr(first, "text"):
            return str(first.text).strip()
        if isinstance(first, str):
            return first.strip()
        return str(first).strip()
    return ""


def main() -> None:
    args = parse_args()

    if (args.input_csv is None) == (args.input_jsonl is None):
        raise ValueError("Provide exactly one of --input_csv or --input_jsonl")
    if args.output_csv is None and args.output_jsonl is None:
        raise ValueError("Provide at least one output: --output_csv or --output_jsonl")

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

    model = init_parakeet(args.model_id)

    asr_texts = []
    statuses = []
    errors = []

    for raw_path in tqdm(df[args.audio_col].tolist(), desc="Parakeet ASR"):
        try:
            resolved = resolve_audio_path(raw_path, audio_root)
            text = transcribe_parakeet(model, resolved, args.language)
            asr_texts.append(text)
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