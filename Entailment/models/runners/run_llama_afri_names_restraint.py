from __future__ import annotations

import argparse
from typing import Dict, List

from Entailment.models.llms.llama_hf import LlamaHF, extract_json
from Entailment.models.utils.io import read_csv_dicts, ensure_dir, write_jsonl


SYSTEM_MSG = "You evaluate semantic restraint and hallucination risks in very short spoken transcripts. Always return strictly valid JSON."

USER_PROMPT_TEMPLATE = """Semantic Restraint / Over-Inference Hypothesis Generation
(Hallucination & Overgeneralization Evaluation)
You are a careful evaluator of spoken language meaning.

You are given a verbatim transcript of a very short spoken audio recording.
The transcript represents all meaningful semantic information available from the audio.
The transcript is provided only for annotation purposes.

This audio contains LIMITED semantic content.
Your task is to generate hypotheses that test whether a model can
correctly WITHHOLD inference when the audio does not provide enough evidence.

TASK:
Generate hypotheses for evaluating semantic restraint and hallucination.

Generate the following:

1) SUPPORTED hypotheses (2 items)
   – Statements that are clearly supported by the meaning of the audio.
   – These should be minimal and cautious.

2) UNSUPPORTED hypotheses (6 items)
   – Statements that may sound reasonable in general speech datasets,
     but are NOT supported by the meaning of this audio.
   – These hypotheses should tempt a model to over-infer.

CONSTRAINTS (must follow strictly):
- Do NOT quote or closely paraphrase the transcript.
- Do NOT describe what the speaker “says” or “mentions.”
- Do NOT add roles, topics, locations, intent, emotions, or domains
  unless they are explicitly supported.
- Each hypothesis must be one complete sentence.
- Avoid explicit negation terms such as “no,” “not,” “never,” or “without.”

GOAL:
Unsupported hypotheses should test for:
- Invented topics
- Invented intent
- Invented social roles
- Invented domains (e.g., politics, health, finance)
- Invented situational context

OUTPUT FORMAT (JSON only):

{
  "supported": [
    "...",
    "..."
  ],
  "unsupported": [
    "...",
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}

TRANSCRIPT:
{{TRANSCRIPT}}
"""


def build_messages(transcript: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.replace("{{TRANSCRIPT}}", transcript)},
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate semantic restraint (supported/unsupported) hypotheses for AfriSpeech-NAMES using Llama.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to Entailment/metadata_afri-names.csv")
    parser.add_argument("--output_dir", type=str, default="results/Entailment/AfriNames/Llama/restraint", help="Output directory")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit_rows", type=int, default=None, help="If set, process only the first N rows")
    args = parser.parse_args()

    rows = read_csv_dicts(args.csv_path)
    out_dir = ensure_dir(args.output_dir)
    out_jsonl = out_dir / "afri_names_restraint.jsonl"

    llm = LlamaHF(model_id=args.model_id)

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


