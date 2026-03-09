#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--audio_col", default="audio_file")
    parser.add_argument("--audio_root", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--simple_jsonl", action="store_true")
    parser.add_argument("--model_id", default="ibm-granite/granite-speech-3.3-2b")
    parser.add_argument("--language", default="english")
    parser.add_argument("--chunk_length_s", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
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
        rooted_basename = (audio_root / candidate.name).resolve()
        if rooted_basename.exists():
            return rooted_basename
    raise FileNotFoundError(f"Audio file not found: {raw_path}")


def init_granite(model_id: str):
    has_cuda = torch.cuda.is_available()
    torch_device = "cuda" if has_cuda else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True).to(torch_device)

    system_prompt = (
        "Knowledge Cutoff Date: April 2024.\n"
        "You are Granite, developed by IBM. Transcribe speech verbatim."
    )
    user_prompt_base = "<|audio|> Please transcribe the speech into written format."

    return model, processor, tokenizer, torch_device, system_prompt, user_prompt_base


def transcribe_granite(
    model,
    processor,
    tokenizer,
    device: str,
    system_prompt: str,
    user_prompt_base: str,
    audio_path: Path,
    chunk_length_s: int,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
) -> str:
    wav, sr = torchaudio.load(str(audio_path), normalize=True)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    chunk_size_samples = int(chunk_length_s) * 16000
    chunks = torch.split(wav, chunk_size_samples, dim=1)

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_base},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{system_prompt}\n{user_prompt_base}"

    generated_texts = []
    for chunk in chunks:
        model_inputs = processor(
            text,
            chunk,
            device=device,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            model_outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=1,
                top_p=1.0,
                repetition_penalty=1.0,
                length_penalty=1.0,
                temperature=1.0,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        output_text = tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )[0]
        generated_texts.append(output_text)

    return " ".join(generated_texts).strip()


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

    model, processor, tokenizer, device, system_prompt, user_prompt_base = init_granite(args.model_id)

    asr_texts = []
    statuses = []
    errors = []

    for raw_path in tqdm(df[args.audio_col].tolist(), desc="Granite ASR"):
        try:
            resolved = resolve_audio_path(raw_path, audio_root)
            asr_text = transcribe_granite(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                system_prompt=system_prompt,
                user_prompt_base=user_prompt_base,
                audio_path=resolved,
                chunk_length_s=args.chunk_length_s,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
            )
            asr_texts.append(asr_text)
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