# Inference System Setup Complete

## ✅ What's Been Set Up

1. **Output Structure**: All results organized by model and task in `/orange/ufdatastudios/c.okocha/Afro_entailment/outputs/`
   - Structure: `outputs/MODEL_NAME/TASK_NAME/results/` and `logs/`
   - Existing Kimi results moved to `outputs/Kimi/medical_nli/results/`

2. **Templates Created**:
   - `inference/templates/infer_jsonl.py` - Generic inference script (JSONL-based)
   - `inference/templates/run_infer.sh` - SLURM batch script template
   - `inference/templates/infer_jsonl_kimi_example.py` - Kimi customization example

3. **Configuration Files**:
   - `inference/task_config.json` - All 14 tasks with paths configured
   - `inference/generate_slurm_scripts.py` - Script generator

4. **Helper Scripts**:
   - `inference/setup_kimi.sh` - Quick setup for Kimi model

## 📁 Current Output Structure

```
outputs/
└── Kimi/
    └── medical_nli/
        ├── results/
        │   ├── kimi_med_nli_optionA.jsonl  (existing results)
        │   └── kimi_med_nli_optionA.json
        └── logs/
```

## 🚀 Next Steps: Run Another Kimi Inference

### Option 1: Use the setup script

```bash
cd /orange/ufdatastudios/c.okocha/Afro_entailment
./inference/setup_kimi.sh
```

This will:
- Copy `infer_jsonl.py` to Kimi-Audio folder (if needed)
- Generate all SLURM scripts in `/orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts/`

### Option 2: Manual setup

```bash
# 1. Generate SLURM scripts
cd /orange/ufdatastudios/c.okocha/Afro_entailment
python3 inference/generate_slurm_scripts.py \
    --model_name Kimi \
    --model_path moonshotai/Kimi-Audio-7B-Instruct \
    --project_dir /orange/ufdatastudios/c.okocha/Kimi-Audio \
    --output_dir /orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts

# 2. Submit a job (example: medical consistency)
cd /orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts
sbatch run_medical_consistency.sh
```

## 📋 Available Tasks

All 14 tasks are configured:
- `medical_consistency`, `medical_nli`
- `parliament_nli`, `parliament_consistency`, `parliament_intent`, `parliament_commonsense`
- `afrispeech200_consistency`, `afrispeech200_nli`, `afrispeech200_plausibility`
- `general_consistency`, `general_plausibility`
- `afrinames_restraint`, `afrinames_accent_drift`

## 📍 Important Notes

1. **Model Dependencies**: Each model runs from its own project folder with its own venv
   - Kimi runs from `/orange/ufdatastudios/c.okocha/Kimi-Audio`
   - Scripts `cd` into the model folder before running

2. **Output Location**: All outputs go to `/orange/ufdatastudios/c.okocha/Afro_entailment/outputs/MODEL/TASK/`
   - This is independent of where you submit the job from
   - Logs and results are organized by model and task

3. **Customization**: Before running, ensure `infer_jsonl.py` in the model folder has:
   - `init_model()` function customized for your model
   - `generate_with_model()` function customized for your model
   - See `inference/templates/infer_jsonl_kimi_example.py` for Kimi example

## 🔍 Verify Setup

Check that SLURM scripts were generated correctly:

```bash
cd /orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts
ls -la run_*.sh
head -30 run_medical_consistency.sh  # Check paths are correct
```

The script should:
- `cd` to `/orange/ufdatastudios/c.okocha/Kimi-Audio`
- Use correct JSONL and audio paths from `task_config.json`
- Output to `outputs/Kimi/TASK_NAME/results/`

