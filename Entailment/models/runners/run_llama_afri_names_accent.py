from __future__ import annotations

import argparse
from typing import Dict, List

from Entailment.models.llms.llama_hf import LlamaHF, extract_json
from Entailment.models.utils.io import read_csv_dicts, ensure_dir, write_jsonl


SYSTEM_MSG = "You evaluate semantic stability under accent variation. Always return strictly valid JSON."

USER_PROMPT_TEMPLATE = """Accent-Conditioned Semantic Drift Hypothesis Generation
(Fairness & Robustness Under Accent Variation)
You are evaluating semantic stability in spoken language understanding.

You are given a verbatim transcript of a short spoken audio recording.
The transcript represents the meaning of the audio and is provided only for annotation.

IMPORTANT:
This task is designed to test whether models infer DIFFERENT meanings
from similar short utterances spoken with different accents or pronunciations.

TASK:
Generate hypotheses that should be interpreted IDENTICALLY
regardless of the speaker’s accent, pronunciation, or speaking style.

Generate the following:

1) ACCENT-INVARIANT hypotheses (4 items)
   – Statements whose truth should NOT depend on accent or speaker identity.
   – These should rely only on the semantic content of the utterance.

2) ACCENT-SENSITIVE LURE hypotheses (4 items)
   – Statements that models might incorrectly infer due to accent,
     pronunciation, or training bias.
   – These should NOT be inferable from the transcript alone.

CONSTRAINTS (must follow strictly):
- Do NOT quote or closely paraphrase the transcript.
- Do NOT assume speaker nationality, ethnicity, gender, age, or location.
- Do NOT assume intent or topic unless explicitly supported.
- Each hypothesis must be one complete sentence.
- Avoid explicit negation terms such as “no,” “not,” “never,” or “without.”

GOAL:
Accent-sensitive lures should test whether a model:
- Assigns social or institutional roles
- Infers geographic or cultural background
- Injects domain meaning
- Changes interpretation due to accent alone

OUTPUT FORMAT (JSON only):

{
  "accent_invariant": [
    "...",
    "...",
    "...",
    "..."
  ],
  "accent_sensitive_lures": [
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
    parser = argparse.ArgumentParser(description="Generate accent-invariant vs accent-sensitive lure hypotheses for AfriSpeech-NAMES using Llama.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to Entailment/metadata_afri-names.csv")
    parser.add_argument("--output_dir", type=str, default="results/Entailment/AfriNames/Llama/accent_drift", help="Output directory")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit_rows", type=int, default=None, help="If set, process only the first N rows")
    args = parser.parse_args()

    rows = read_csv_dicts(args.csv_path)
    out_dir = ensure_dir(args.output_dir)
    out_jsonl = out_dir / "afri_names_accent.jsonl"

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


