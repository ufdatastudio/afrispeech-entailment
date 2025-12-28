from __future__ import annotations

import argparse
from typing import Dict, List

from Entailment.models.llms.llama_hf import LlamaHF, extract_json
from Entailment.models.utils.io import read_csv_dicts, ensure_dir, write_jsonl


SYSTEM_MSG = "You assess semantic consistency from general spoken transcripts. Always return strictly valid JSON."

USER_PROMPT_TEMPLATE_MULTI = """You are given a transcript of a spoken audio recording.

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

USER_PROMPT_TEMPLATE_SINGLE = """You are given a transcript of a spoken audio recording.

Task:
Generate two statements:
- one semantically consistent with the audio.
- one semantically inconsistent with the audio.

Constraints:
- Statements should be grounded in the general meaning of the speech.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "consistent": ["..."],
  "inconsistent": ["..."]
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
    parser = argparse.ArgumentParser(description="Generate AfriSpeech-200 semantic consistency annotations using Llama.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to Entailment/metadata_afrispeech-200.csv")
    parser.add_argument("--output_dir", type=str, default="results/Entailment/AfriSpeech200/Llama/consistency", help="Output directory")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit_rows", type=int, default=None, help="If set, process only the first N rows")
    parser.add_argument("--single", action="store_true", help="Generate a single consistent and a single inconsistent statement")
    args = parser.parse_args()

    rows = read_csv_dicts(args.csv_path)
    out_dir = ensure_dir(args.output_dir)
    out_jsonl = out_dir / "afrispeech200_consistency.jsonl"

    llm = LlamaHF(model_id=args.model_id)

    # Choose prompt variant
    global USER_PROMPT_TEMPLATE
    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE_SINGLE if args.single else USER_PROMPT_TEMPLATE_MULTI

    outputs = []
    iterable = rows[: args.limit_rows] if args.limit_rows else rows
    for idx, row in enumerate(iterable):
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











