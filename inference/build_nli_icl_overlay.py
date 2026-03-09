#!/usr/bin/env python3
"""
Build in-context learning (ICL) overlays for NLI/entailment JSONL datasets.

This script creates shot-specific JSONL files (N=1..K) by prepending a fixed
exemplar block to every hypothesis string, so existing zero-shot inference
scripts can be reused without code changes.

Two variants are supported:
- audio_only: examples use AudioFile + Hypothesis + Answer
- audio_plus_transcript: examples use AudioFile + Transcript + Hypothesis + Answer
- audio_exemplar_only: examples use AudioFile + Answer (no exemplar hypothesis/transcript)

Input format expected (per record):
- file_name
- transcript
- output: {"entailment": [...], "neutral": [...], "contradiction": [...]} 

Output:
- exemplars_10.jsonl (fixed exemplars used for all shots)
- {variant}/afrispeech200_nli_icl_shot{N}.jsonl for N=1..K
- {variant}/medical_nli_icl_shot{N}.jsonl for N=1..K
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABEL_MAP = {
    "entailment": "ENTAILMENT",
    "neutral": "NEUTRAL",
    "contradiction": "CONTRADICTION",
}


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def clip_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def format_audio_id(file_name: str, audio_id_mode: str) -> str:
    if audio_id_mode == "basename":
        return Path(file_name).name if file_name else ""
    return file_name


def extract_candidates(records: List[Dict]) -> List[Dict]:
    candidates: List[Dict] = []
    for rec in records:
        transcript = str(rec.get("transcript", "")).strip()
        file_name = str(rec.get("file_name", "")).strip()
        output = rec.get("output", {})
        if not isinstance(output, dict):
            continue

        for key in ("entailment", "neutral", "contradiction"):
            label = LABEL_MAP[key]
            for hyp in ensure_list(output.get(key, [])):
                candidates.append(
                    {
                        "file_name": file_name,
                        "transcript": transcript,
                        "hypothesis": hyp.strip(),
                        "answer": label,
                    }
                )
    return candidates


def select_fixed_exemplars(candidates: List[Dict], num_exemplars: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict]] = {"ENTAILMENT": [], "NEUTRAL": [], "CONTRADICTION": []}

    for c in candidates:
        answer = c.get("answer")
        if answer in by_label:
            by_label[answer].append(c)

    for label in by_label:
        rng.shuffle(by_label[label])

    selected: List[Dict] = []
    used_keys = set()
    label_cycle = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

    def unique_key(item: Dict) -> Tuple[str, str, str]:
        return (
            item.get("file_name", ""),
            item.get("hypothesis", ""),
            item.get("answer", ""),
        )

    cycle_idx = 0
    while len(selected) < num_exemplars:
        target_label = label_cycle[cycle_idx % len(label_cycle)]
        cycle_idx += 1

        picked = None
        while by_label[target_label]:
            candidate = by_label[target_label].pop()
            key = unique_key(candidate)
            if key not in used_keys:
                picked = candidate
                used_keys.add(key)
                break

        if picked is not None:
            selected.append(picked)
            continue

        fallback = None
        for alt_label in label_cycle:
            while by_label[alt_label]:
                candidate = by_label[alt_label].pop()
                key = unique_key(candidate)
                if key not in used_keys:
                    fallback = candidate
                    used_keys.add(key)
                    break
            if fallback is not None:
                break

        if fallback is None:
            break
        selected.append(fallback)

    if len(selected) < num_exemplars:
        raise ValueError(
            f"Could only select {len(selected)} unique exemplars, but {num_exemplars} were requested."
        )

    return selected


def build_icl_prefix(
    exemplars: List[Dict],
    n_shots: int,
    include_transcript: bool,
    include_exemplar_hypothesis: bool,
    audio_id_mode: str,
    max_transcript_chars: Optional[int],
) -> str:
    example_fields = "AudioFile and gold Answer label"
    if include_exemplar_hypothesis and include_transcript:
        example_fields = "AudioFile, Transcript, Hypothesis, and gold Answer label"
    elif include_exemplar_hypothesis:
        example_fields = "AudioFile, Hypothesis, and gold Answer label"
    elif include_transcript:
        example_fields = "AudioFile, Transcript, and gold Answer label"

    header = [
        "Use the solved examples below to classify the target pair.",
        f"Each example includes {example_fields}.",
        "Valid labels: ENTAILMENT, CONTRADICTION, NEUTRAL.",
        "",
    ]

    body: List[str] = []
    for idx, ex in enumerate(exemplars[:n_shots], start=1):
        exemplar_audio = format_audio_id(ex.get("file_name", ""), audio_id_mode)
        exemplar_transcript = clip_text(str(ex.get("transcript", "")).strip(), max_transcript_chars)
        body.extend(
            [
                f"Example {idx}:",
                f"AudioFile: {exemplar_audio}",
                *([f"Transcript: {exemplar_transcript}"] if include_transcript else []),
                *([f"Hypothesis: {ex['hypothesis']}"] if include_exemplar_hypothesis else []),
                f"Answer: {ex['answer']}",
                "",
            ]
        )

    trailer = [
        "Now classify the target pair with one label only.",
        "",
    ]

    return "\n".join(header + body + trailer)


def overlay_record(
    rec: Dict,
    exemplars: List[Dict],
    n_shots: int,
    include_transcript: bool,
    include_exemplar_hypothesis: bool,
    include_target_transcript: bool,
    variant: str,
    audio_id_mode: str,
    max_transcript_chars: Optional[int],
) -> Dict:
    new_rec = copy.deepcopy(rec)
    output = new_rec.get("output", {})
    if not isinstance(output, dict):
        return new_rec

    prefix = build_icl_prefix(
        exemplars,
        n_shots,
        include_transcript=include_transcript,
        include_exemplar_hypothesis=include_exemplar_hypothesis,
        audio_id_mode=audio_id_mode,
        max_transcript_chars=max_transcript_chars,
    )
    transcript = str(new_rec.get("transcript", "")).strip()
    file_name = str(new_rec.get("file_name", "")).strip()
    audio_id = format_audio_id(file_name, audio_id_mode)
    clipped_target_transcript = clip_text(transcript, max_transcript_chars)

    for key in ("entailment", "neutral", "contradiction"):
        hyps = ensure_list(output.get(key, []))
        transformed: List[str] = []
        for hyp in hyps:
            target_lines = [
                f"{prefix}"
                f"Target AudioFile: {audio_id}",
            ]
            if include_target_transcript:
                target_lines.append(f"Target Transcript: {clipped_target_transcript}")
            target_lines.extend(
                [
                    f"Target Hypothesis: {hyp}",
                    "Answer:",
                ]
            )
            transformed_hyp = (
                "\n".join(target_lines)
            )
            transformed.append(transformed_hyp)
        output[key] = transformed

    new_rec["output"] = output
    new_rec["icl_overlay"] = {
        "enabled": True,
        "variant": variant,
        "include_transcript": include_transcript,
        "include_exemplar_hypothesis": include_exemplar_hypothesis,
        "include_target_transcript": include_target_transcript,
        "audio_id_mode": audio_id_mode,
        "max_transcript_chars": max_transcript_chars,
        "num_shots": n_shots,
        "exemplar_count": len(exemplars),
    }
    return new_rec


def build_overlay_dataset(
    records: List[Dict],
    exemplars: List[Dict],
    n_shots: int,
    include_transcript: bool,
    include_exemplar_hypothesis: bool,
    include_target_transcript: bool,
    variant: str,
    audio_id_mode: str,
    max_transcript_chars: Optional[int],
) -> List[Dict]:
    return [
        overlay_record(
            rec,
            exemplars,
            n_shots,
            include_transcript=include_transcript,
            include_exemplar_hypothesis=include_exemplar_hypothesis,
            include_target_transcript=include_target_transcript,
            variant=variant,
            audio_id_mode=audio_id_mode,
            max_transcript_chars=max_transcript_chars,
        )
        for rec in records
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NLI ICL overlay JSONLs for N=1..K")

    parser.add_argument(
        "--afri200_jsonl",
        default="/orange/ufdatastudios/c.okocha/afrispeech-entailment/result/Entailment/AfriSpeech200/Llama/nli_single/afrispeech200_nli_top100.jsonl",
    )
    parser.add_argument(
        "--medical_jsonl",
        default="/orange/ufdatastudios/c.okocha/afrispeech-entailment/result/Entailment/Medical/Llama/nli/medical_nli.jsonl",
    )
    parser.add_argument(
        "--exemplar_source_jsonl",
        default=None,
        help="Source JSONL for selecting fixed exemplars. Default: afri200_jsonl",
    )
    parser.add_argument(
        "--output_root",
        default="/orange/ufdatastudios/c.okocha/afrispeech-entailment/result/Entailment/ICL_overlay_nli",
    )
    parser.add_argument("--num_exemplars", type=int, default=10)
    parser.add_argument("--max_shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_target_transcript",
        action="store_true",
        help="Include transcript in the target section. Off by default for fair A/B where examples differ.",
    )
    parser.add_argument(
        "--max_transcript_chars",
        type=int,
        default=None,
        help="Optional transcript character cap (applies to example/target transcript text when present).",
    )
    parser.add_argument(
        "--audio_id_mode",
        choices=["basename", "fullpath"],
        default="basename",
        help="How to represent audio file IDs in prompts. basename greatly reduces tokens.",
    )
    parser.add_argument(
        "--variant",
        choices=["audio_only", "audio_plus_transcript", "audio_exemplar_only", "both"],
        default="both",
        help="ICL variant to generate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    afri200_path = Path(args.afri200_jsonl).resolve()
    medical_path = Path(args.medical_jsonl).resolve()
    exemplar_source_path = Path(args.exemplar_source_jsonl).resolve() if args.exemplar_source_jsonl else afri200_path
    output_root = Path(args.output_root).resolve()

    afri200_records = read_jsonl(afri200_path)
    medical_records = read_jsonl(medical_path)
    exemplar_source_records = read_jsonl(exemplar_source_path)

    candidates = extract_candidates(exemplar_source_records)
    exemplars = select_fixed_exemplars(candidates, args.num_exemplars, args.seed)

    exemplars_path = output_root / "exemplars_10.jsonl"
    write_jsonl(exemplars_path, exemplars)

    variant_config = {
        "audio_only": {"include_transcript": False, "include_exemplar_hypothesis": True},
        "audio_plus_transcript": {"include_transcript": True, "include_exemplar_hypothesis": True},
        "audio_exemplar_only": {"include_transcript": False, "include_exemplar_hypothesis": False},
    }

    variants = [args.variant] if args.variant != "both" else ["audio_only", "audio_plus_transcript", "audio_exemplar_only"]
    generated_files = {}

    for variant in variants:
        include_transcript = variant_config[variant]["include_transcript"]
        include_exemplar_hypothesis = variant_config[variant]["include_exemplar_hypothesis"]
        variant_root = output_root / variant
        generated_files[variant] = {
            "afrispeech200": [],
            "medical": [],
        }

        for n in range(1, args.max_shots + 1):
            afri200_overlay = build_overlay_dataset(
                afri200_records,
                exemplars,
                n,
                include_transcript=include_transcript,
                include_exemplar_hypothesis=include_exemplar_hypothesis,
                include_target_transcript=args.include_target_transcript,
                variant=variant,
                audio_id_mode=args.audio_id_mode,
                max_transcript_chars=args.max_transcript_chars,
            )
            medical_overlay = build_overlay_dataset(
                medical_records,
                exemplars,
                n,
                include_transcript=include_transcript,
                include_exemplar_hypothesis=include_exemplar_hypothesis,
                include_target_transcript=args.include_target_transcript,
                variant=variant,
                audio_id_mode=args.audio_id_mode,
                max_transcript_chars=args.max_transcript_chars,
            )

            afri200_out = variant_root / f"afrispeech200_nli_icl_shot{n}.jsonl"
            medical_out = variant_root / f"medical_nli_icl_shot{n}.jsonl"

            write_jsonl(afri200_out, afri200_overlay)
            write_jsonl(medical_out, medical_overlay)

            generated_files[variant]["afrispeech200"].append(str(afri200_out))
            generated_files[variant]["medical"].append(str(medical_out))

    manifest = {
        "afri200_source": str(afri200_path),
        "medical_source": str(medical_path),
        "exemplar_source": str(exemplar_source_path),
        "output_root": str(output_root),
        "variants_generated": variants,
        "num_exemplars": args.num_exemplars,
        "max_shots": args.max_shots,
        "seed": args.seed,
        "include_target_transcript": args.include_target_transcript,
        "max_transcript_chars": args.max_transcript_chars,
        "audio_id_mode": args.audio_id_mode,
        "exemplars_file": str(exemplars_path),
        "generated_files": generated_files,
        "models_targeted": [
            "Qwen2-Audio-7B-Instruct",
            "Kimi-Audio-7B-Instruct",
            "AudioFlamingo2",
            "AudioFlamingo3",
        ],
        "notes": "Overlay files are compatible with existing zero-shot JSONL inference scripts because hypotheses are text-prefixed with ICL exemplars.",
    }

    manifest_path = output_root / "icl_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("✅ ICL overlay generation complete")
    print(f"Exemplars: {exemplars_path}")
    print(f"Manifest:  {manifest_path}")
    print(f"Variants:  {', '.join(variants)}")
    print(f"Shots:     1..{args.max_shots}")
    print(f"Afri-200 rows: {len(afri200_records)} | Medical rows: {len(medical_records)}")


if __name__ == "__main__":
    main()
