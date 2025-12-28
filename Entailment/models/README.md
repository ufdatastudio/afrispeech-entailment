### Entailment/models

Clean runners and a shared Llama HF backend for generating annotations from parliamentary transcripts in `metadata_afrispeech-parliament.csv`.

#### Layout
- `llms/llama_hf.py`: Minimal HF Transformers client.
- `runners/`:
  - `run_llama_parliament_entailment.py`: NLI hypotheses (entailment/neutral/contradiction)
  - `run_llama_parliament_consistency.py`: consistency vs inconsistency statements
  - `run_llama_parliament_intent.py`: communicative intent statements
  - `run_llama_parliament_commonsense.py`: commonsense inferences
- `utils/io.py`: CSV/JSONL helpers

#### Input
`Entailment/metadata_afrispeech-parliament.csv` with columns:
- `file_name`, `transcript`

#### Usage
Examples (adjust model if desired):

```bash
python -m Entailment.models.runners.run_llama_parliament_entailment \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-parliament.csv \
  --output_dir results/Entailment/Parliament/Llama/entailment_hypotheses \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct

python -m Entailment.models.runners.run_llama_parliament_consistency \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-parliament.csv \
  --output_dir results/Entailment/Parliament/Llama/consistency

python -m Entailment.models.runners.run_llama_parliament_intent \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-parliament.csv \
  --output_dir results/Entailment/Parliament/Llama/intent

python -m Entailment.models.runners.run_llama_parliament_commonsense \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-parliament.csv \
  --output_dir results/Entailment/Parliament/Llama/commonsense
```

Each script writes a `.jsonl` file with one JSON object per row: `{file_name, transcript, model_id, output}`.

### Medical runners
- Input CSV: `Entailment/metadata_medical.csv` with columns: `file_name`, `transcript`, `domain`

```bash
python -m Entailment.models.runners.run_llama_medical_nli \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_medical.csv \
  --output_dir results/Entailment/Medical/Llama/nli

python -m Entailment.models.runners.run_llama_medical_consistency \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_medical.csv \
  --output_dir results/Entailment/Medical/Llama/consistency

python -m Entailment.models.runners.run_llama_medical_plausibility \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_medical.csv \
  --output_dir results/Entailment/Medical/Llama/plausibility
```

### AfriSpeech-200 runners
- Input CSV: `Entailment/metadata_afrispeech-200.csv` with columns: `file_name`, `transcript`

```bash
python -m Entailment.models.runners.run_llama_afrispeech200_nli \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-200.csv \
  --output_dir results/Entailment/AfriSpeech200/Llama/nli

python -m Entailment.models.runners.run_llama_afrispeech200_consistency \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-200.csv \
  --output_dir results/Entailment/AfriSpeech200/Llama/consistency

python -m Entailment.models.runners.run_llama_afrispeech200_plausibility \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-200.csv \
  --output_dir results/Entailment/AfriSpeech200/Llama/plausibility
```

### AfriSpeech-General runners
- Input CSV: `Entailment/metadata_afrispeech-general.csv` with columns: `file_name`, `transcript`, `domain`

```bash
python -m Entailment.models.runners.run_llama_afrispeech_general_consistency \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-general.csv \
  --output_dir results/Entailment/AfriSpeechGeneral/Llama/consistency

python -m Entailment.models.runners.run_llama_afrispeech_general_plausibility \
  --csv_path /orange/ufdatastudios/c.okocha/AI-Jobs-Research/Entailment/metadata_afrispeech-general.csv \
  --output_dir results/Entailment/AfriSpeechGeneral/Llama/plausibility
```


