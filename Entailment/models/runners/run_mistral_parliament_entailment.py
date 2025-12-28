from __future__ import annotations

import argparse
from typing import Dict, List

from Entailment.models.llms.mistral_hf import MistralHF, extract_json
from Entailment.models.utils.io import read_csv_dicts, ensure_dir, write_jsonl


SYSTEM_MSG = "You are a helpful assistant with expert knowledge in spoken language understanding, discourse analysis, and semantic inference in institutional settings such as parliamentary or legislative proceedings. Always return strictly valid JSON."

USER_PROMPT_TEMPLATE = """You are a helpful assistant with expert knowledge in spoken language understanding, discourse analysis, and semantic inference in institutional settings such as parliamentary or legislative proceedings.

You are given a verbatim transcript of a single spoken audio recording from a formal parliamentary context. The transcript represents the semantic content conveyed by the audio and is provided only for annotation purposes.

Using the transcript and your knowledge of parliamentary discourse and language use, generate hypotheses for a Spoken Natural Language Inference task.

Instructions:
Generate three hypotheses for each category:
- Entailment: statements that are definitely true given the meaning of the spoken audio.
- Neutral: statements that might be true but cannot be determined from the audio alone.
- Contradiction: statements that are definitely false given the meaning of the audio.

Constraints:
- Do not quote or closely paraphrase the transcript.
- Do not state that the speaker “says” or “mentions” something.
- Hypotheses must reflect semantic inference, not surface wording.
- Do not use explicit negation terms such as “no,” “not,” or “never.”
- Each hypothesis must be a single complete sentence.

Respond only in valid JSON with the structure:
{
  "entailment": ["...", "...", "..."],
  "neutral": ["...", "...", "..."],
  "contradiction": ["...", "...", "..."]
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
    parser = argparse.ArgumentParser(description="Generate NLI hypotheses (entailment/neutral/contradiction) for parliament transcripts using Mistral.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata_afrispeech-parliament.csv")
    parser.add_argument("--output_dir", type=str, default="results/Entailment/Parliament/Mistral/entailment_hypotheses", help="Output directory")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="HF model id")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_csv_dicts(args.csv_path)
    out_dir = ensure_dir(args.output_dir)
    out_jsonl = out_dir / "entailment_hypotheses.jsonl"

    llm = MistralHF(args.model_id)

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









