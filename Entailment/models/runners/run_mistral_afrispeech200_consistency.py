from __future__ import annotations

import argparse
from typing import Dict, List

from Entailment.models.llms.mistral_hf import MistralHF, extract_json
from Entailment.models.utils.io import read_csv_dicts, ensure_dir, write_jsonl


SYSTEM_MSG = "You assess semantic consistency from general spoken transcripts. Always return strictly valid JSON."

USER_PROMPT_TEMPLATE = """You are given a transcript of a spoken audio recording.

Task:
Generate four statements:
- Two semantically consistent with the audio.
- Two semantically inconsistent with the audio.

Constraints:
- Statements should be grounded in the general meaning of the speech.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "consistent": ["...", "..."],
  "inconsistent": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
"""


def build_messages(transcript: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{{TRANSCRIPT}}", transcript)},
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate AfriSpeech-200 semantic consistency annotations using Mistral.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to Entailment/metadata_afrispeech-200.csv")
    parser.add_argument("--output_dir", type=str, default="results/Entailment/AfriSpeech200/Mistral/consistency", help="Output directory")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_csv_dicts(args.csv_path)
    out_dir = ensure_dir(args.output_dir)
    out_jsonl = out_dir / "afrispeech200_consistency.jsonl"

    llm = MistralHF(model_id=args.model_id)

    outputs = []
    for idx, row in enumerate(rows):
        file_name = row.get("file_name", f"row_{idx+1}.wav")
        transcript = (row.get("transcript") or "").strip()
        if not transcript:
            continue

        messages = build_messages(transcript)
        text = llm.generate(messages, args.max_new_tokens, args.temperature)
        data = extract_json(text) or {}
        outputs.append({
            "file_name": file_name,
            "transcript": transcript,
            "model_id": args.model_id,
            "output": data
        })

    write_jsonl(out_jsonl, outputs)
    print(f"Saved {len(outputs)} rows to {out_jsonl}")


if __name__ == "__main__":
    main()









