# ICL Overlay for NLI (Afri-200 + Medical)

This overlay keeps your zero-shot inference scripts unchanged.

## Two experimental versions
- **Version A (audio-only ICL, preferred for your claim):**
  - Examples include `AudioFile + Hypothesis + Answer`.
  - Tests whether ICL improves reasoning from audio-grounded context.
- **Version B (audio + transcript ICL, diagnostic):**
  - Examples include `AudioFile + Transcript + Hypothesis + Answer`.
  - Tests whether gains come from stronger linguistic text conditioning.
- **Version C (audio-exemplar-only ICL, strict ablation):**
  - Examples include `AudioFile + Answer` only.
  - Removes exemplar transcript+hypothesis content to isolate pure exemplar label anchoring.

For a fair A/B/C, keep the **target section identical** and change only the examples.
By default, this builder does that (`--include_target_transcript` is OFF).

Interpretation target:
- If **Version B >> Version A**, errors are more likely linguistic/decision bias than purely acoustic.

## What it does
- Uses the same **10 fixed exemplars** across both versions.
- Builds shot-sweep datasets for **N=1..10**.
- Targets NLI (entailment task) on:
  - Afri-200 (`afrispeech200_nli_top100.jsonl`)
  - Medical (`medical_nli.jsonl`)

## Generate overlay files

```bash
cd /orange/ufdatastudios/c.okocha/afrispeech-entailment
python3 inference/build_nli_icl_overlay.py
```

Default generation is token-optimized:
- `--audio_id_mode basename` (default) reduces long path tokens.
- target transcript is excluded unless `--include_target_transcript` is set.

Or generate one variant only:

```bash
python3 inference/build_nli_icl_overlay.py --variant audio_only
python3 inference/build_nli_icl_overlay.py --variant audio_plus_transcript
python3 inference/build_nli_icl_overlay.py --variant audio_exemplar_only
```

Optional context controls:

```bash
# cap transcript text to 220 chars in Version B examples
python3 inference/build_nli_icl_overlay.py --variant audio_plus_transcript --max_transcript_chars 220

# if needed for ablation, include target transcript too
python3 inference/build_nli_icl_overlay.py --variant audio_plus_transcript --include_target_transcript
```

Outputs are written to:

- `result/Entailment/ICL_overlay_nli/exemplars_10.jsonl`
- `result/Entailment/ICL_overlay_nli/audio_only/afrispeech200_nli_icl_shot{1..10}.jsonl`
- `result/Entailment/ICL_overlay_nli/audio_only/medical_nli_icl_shot{1..10}.jsonl`
- `result/Entailment/ICL_overlay_nli/audio_plus_transcript/afrispeech200_nli_icl_shot{1..10}.jsonl`
- `result/Entailment/ICL_overlay_nli/audio_plus_transcript/medical_nli_icl_shot{1..10}.jsonl`
- `result/Entailment/ICL_overlay_nli/audio_exemplar_only/afrispeech200_nli_icl_shot{1..10}.jsonl`
- `result/Entailment/ICL_overlay_nli/audio_exemplar_only/medical_nli_icl_shot{1..10}.jsonl`
- `result/Entailment/ICL_overlay_nli/icl_overlay_manifest.json`

## Run with existing model scripts

Use each generated `*_shotN.jsonl` as the model input JSONL exactly like zero-shot.

Using the sweep helper:

```bash
# Version A: audio-only ICL
ICL_VARIANT=audio_only MODEL=qwen2audio DRY_RUN=1 bash inference/run_nli_icl_sweep_overlay.sh

# Version B: audio+transcript ICL
ICL_VARIANT=audio_plus_transcript MODEL=qwen2audio DRY_RUN=1 bash inference/run_nli_icl_sweep_overlay.sh

# Version C: audio-exemplar-only ICL
ICL_VARIANT=audio_exemplar_only MODEL=qwen2audio DRY_RUN=1 bash inference/run_nli_icl_sweep_overlay.sh
```

Set `DRY_RUN=0` to execute.

### Qwen2-Audio 7B (not Qwen2.5-Omni)
Use your existing Qwen2-Audio inference runner; only swap `--input_jsonl` to the shot file.

### Kimi
Use your existing Kimi inference runner; only swap `--jsonl_path`/`--input_jsonl` to the shot file.

### AudioFlamingo2
Use your existing AF2 inference runner; only swap `--jsonl_path` to the shot file.

### AudioFlamingo3
Use your existing AF3 inference runner; only swap `--jsonl_path` to the shot file.

## Notes
- The overlay is text-level: exemplar context is prepended to each hypothesis string.
- This is intentional so it works as a clean add-on to zero-shot templates.
- `Entailment == NLI` in this setup (labels: ENTAILMENT / CONTRADICTION / NEUTRAL).
- Strongly recommended for stability: run `audio_only` up to shot10; run `audio_plus_transcript` with transcript caps (e.g., 160-260 chars) if context pressure appears.
