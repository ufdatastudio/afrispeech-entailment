from __future__ import annotations

import argparse
from typing import Dict, List

from Entailment.models.llms.llama_hf import LlamaHF, extract_json
from Entailment.models.utils.io import read_csv_dicts, ensure_dir, write_jsonl


SYSTEM_MSG = "You infer commonsense implications in parliamentary contexts. Always return strictly valid JSON."

USER_PROMPT_TEMPLATE = """You are given a transcript of a spoken audio recording from a formal parliamentary context.

The transcript reflects the semantic content conveyed by the audio and is provided only for annotation.

Task:
Generate three commonsense inferences that a typical listener could reasonably make based on shared knowledge of parliamentary procedures and institutional norms, even if these facts are not explicitly stated.

Constraints:
- Inferences must rely on social or institutional knowledge.
- Do not quote or paraphrase the transcript.
- Do not use explicit negation.
- Each inference must be one sentence.

Respond only in JSON:
{{
  "commonsense_inference": ["...", "...", "..."]
}}

Transcript:
{{TRANSCRIPT}}
"""


def build_messages(transcript: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{{TRANSCRIPT}}", transcript)},
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate commonsense inferences for parliament transcripts using Llama.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata_afrispeech-parliament.csv")
    parser.add_argument("--output_dir", type=str, default="results/Entailment/Parliament/Llama/commonsense", help="Output directory")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_csv_dicts(args.csv_path)
    out_dir = ensure_dir(args.output_dir)
    out_jsonl = out_dir / "commonsense.jsonl"

    llm = LlamaHF(args.model_id)

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











