# Audio Entailment Inference System

This directory contains templates and configuration for running audio-language model inference on the Afro_entailment benchmark.

## Structure

```
inference/
├── templates/
│   ├── infer_jsonl.py          # Generic inference script (copy to each model folder)
│   └── run_infer.sh            # SLURM batch script template (copy and customize)
├── task_config.json            # Task configuration mapping
└── README.md                   # This file
```

## Quick Start

### 1. Set up a new model project

For each model (e.g., Kimi, Whisper, etc.), create a project folder:

```bash
mkdir -p /orange/ufdatastudios/c.okocha/MODEL-NAME
cd /orange/ufdatastudios/c.okocha/MODEL-NAME
```

### 2. Copy templates

```bash
cp /orange/ufdatastudios/c.okocha/Afro_entailment/inference/templates/infer_jsonl.py .
cp /orange/ufdatastudios/c.okocha/Afro_entailment/inference/templates/run_infer.sh .
chmod +x run_infer.sh
```

### 3. Customize `infer_jsonl.py`

Edit the `init_model()` and `generate_with_model()` functions to match your model's API:

```python
def init_model(model_path: str):
    # Example for Kimi:
    from kimia_infer.api.kimia import KimiAudio
    return KimiAudio(model_path=model_path, load_detokenizer=True)

def generate_with_model(model, messages, sampling_params, max_new_tokens):
    # Example for Kimi:
    _, text = model.generate(
        messages,
        **sampling_params,
        output_type="text",
        max_new_tokens=max_new_tokens
    )
    return text
```

### 4. Create task-specific SLURM scripts

For each task, copy `run_infer.sh` and customize:

```bash
# Example: Medical Consistency
cp run_infer.sh run_medical_consistency.sh

# Edit run_medical_consistency.sh:
# - Update job name
# - Set JSONL_PATH, AUDIO_DIR, TASK, MODEL_PATH
# - Set output paths
```

Or use the helper script (see below).

## Task Configuration

All tasks are defined in `task_config.json`. Each task has:
- `task`: Task type (nli, consistency, plausibility, etc.)
- `jsonl_path`: Path to input JSONL with hypotheses
- `audio_dir`: Directory containing audio files
- `output_prefix`: Prefix for output files

## Running Inference

### Single task

```bash
sbatch run_medical_consistency.sh
```

### Batch submission (15 jobs per task)

Create a script that submits all 15 batches:

```bash
#!/bin/bash
for i in {1..15}; do
  sbatch run_medical_consistency.sh --batch $i
done
```

## Output Structure

Outputs are organized by model and task:

```
MODEL-PROJECT/
└── outputs/
    └── MODEL/
        └── TASK/
            ├── results/
            │   ├── MODEL_DATASET_TASK.jsonl
            │   └── MODEL_DATASET_TASK.json
            └── logs/
                ├── infer_JOBID.out
                └── infer_JOBID.err
```

## Supported Tasks

- **Medical**: Consistency, NLI
- **Parliament**: NLI, Consistency, Intent, Commonsense
- **AfriSpeech-200**: Consistency, NLI, Plausibility
- **General**: Consistency, Plausibility
- **AfriNames**: Restraint, Accent Drift

## Notes

- Parliament audio paths are placeholders - update when audio is available
- Each model folder should have its own virtual environment
- Model-specific sampling parameters can be customized in `infer_jsonl.py`
- The script supports resume mode (`--resume`) to skip already-processed items

