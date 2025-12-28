# Afro_entailment

**Audio–Semantic Reasoning Benchmark for Speech-Based Audio Language Models**

A comprehensive multi-domain benchmark for evaluating audio–semantic reasoning in speech-based audio language models (ALMs). This benchmark tests semantic grounding, inference robustness, and bias sensitivity across accented speech, institutional discourse, medical communication, and short-form utterances.

## Overview

Unlike traditional audio captioning, this benchmark focuses on **semantic reasoning over spoken content**, using transcripts only as annotation scaffolding. At inference time, models are evaluated solely on **audio + text hypotheses**, ensuring that models learn to infer meaning from speech rather than rely on surface cues.

## Core Research Goals

This benchmark evaluates whether audio language models can:

- **Infer meaning from speech** rather than rely on surface cues
- **Distinguish entailment, neutrality, and contradiction** from audio alone
- **Maintain semantic restraint** when speech content is sparse
- **Avoid hallucination and over-inference**, especially with accented speech
- **Remain robust** across domains, accents, and discourse styles

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Task Taxonomy](#task-taxonomy)
- [Downloading Audio Files](#downloading-audio-files)
- [Generating Hypotheses](#generating-hypotheses)
- [Running Model Inference](#running-model-inference)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Hugging Face account with access to gated datasets
- CUDA-capable GPU (recommended for model inference)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Afro_entailment
```

2. **Create a virtual environment using uv:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
uv pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install transformers datasets huggingface-hub pandas tqdm soundfile
```

4. **Set up Hugging Face authentication:**
```bash
# Option 1: Login interactively
python -c "from huggingface_hub import login; login()"

# Option 2: Set environment variable
export HF_TOKEN=your_token_here
```

## Project Structure

```
Afro_entailment/
├── Audio/                          # Audio files (gitignored)
│   ├── data/                       # Raw audio files
│   ├── medical/                   # Medical domain audio
│   ├── general/                    # General domain audio
│   └── Afrispeech_Common_Audio/    # Common audio dataset
├── Entailment/
│   ├── metadata_*.csv             # Dataset metadata files
│   ├── models/                     # Model runners and utilities
│   │   ├── llms/                   # LLM backends (Llama, Mistral)
│   │   ├── runners/                # Task-specific runners
│   │   └── utils/                  # I/O utilities
│   └── AudioEntailment/            # Additional audio entailment data
├── result/                         # Generated results and annotations
├── download_*.py                   # Audio download scripts
├── separate_audio_files.py        # Audio file organization script
└── README.md
```

## Datasets

The benchmark uses five speech datasets from distinct domains:

| Dataset | Domain | Description | Size |
|---------|--------|-------------|------|
| **AfriSpeech–Parliament** | Institutional speech | Parliamentary and legislative discourse | ~7,400 samples |
| **Medical** | Clinical speech | Medical explanations and health-related dialogue | ~20 samples |
| **AfriSpeech-200** | Conversational speech | African-accented read and conversational speech | ~6,300 samples |
| **General** | Open-domain speech | Short everyday utterances | ~29 samples |
| **AfriNames** | Short-form speech | Minimal utterances with strong accent variation | ~6,300 samples |

### Dataset Metadata

All datasets are provided as CSV files in `Entailment/` with the following structure:

- **Required columns:** `file_name`, `transcript`
- **Optional columns:** `domain`, `accent`, `country`, `age_group`, `duration`

## Task Taxonomy

The benchmark supports five audio–semantic reasoning tasks:

### Tier 1: Core Tasks

1. **Spoken Natural Language Inference (NLI)**
   - Labels: Entailment / Neutral / Contradiction
   - Goal: Test logical semantic inference from spoken audio
   - Premise: Audio | Hypothesis: Text

2. **Audio–Text Semantic Consistency**
   - Labels: Consistent / Inconsistent
   - Goal: Verify whether a hypothesis aligns with audio meaning
   - Simplified version of NLI for robustness evaluation

### Tier 2: Extension & Ablation Tasks

3. **Audio–Intent Inference**
   - Goal: Infer speaker intent (e.g., support, explanation, request)
   - Used primarily for institutional and medical speech

4. **Semantic Plausibility Judgment**
   - Labels: Plausible / Implausible
   - Goal: Assess commonsense alignment without strict logical entailment

5. **Spoken Commonsense Reasoning**
   - Goal: Infer unstated but obvious world knowledge
   - Example: Audio: "I second the motion." → Question: "Is a formal decision-making process occurring?"

### Dataset–Task Alignment

| Dataset | NLI | Consistency | Intent | Plausibility | Commonsense |
|---------|-----|-------------|--------|--------------|-------------|
| AfriSpeech–Parliament | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Medical | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐ |
| AfriSpeech-200 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| General | ⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ❌ |
| AfriNames | ❌ | ⭐⭐ | ❌ | ❌ | ❌ |

### AfriNames: Fairness & Robustness Diagnostics

AfriNames serves as a critical fairness evaluation tool with two diagnostic tasks:

1. **Semantic Restraint / Over-Inference Detection**
   - Tests whether models hallucinate meaning when audio content is minimal
   - Expected behavior: neutral or "cannot determine"
   - Failure mode: confident but unsupported inference

2. **Accent-Conditioned Semantic Drift**
   - Tests whether accent alone alters semantic judgments
   - Expected behavior: stable predictions across accents
   - Failure mode: inconsistent reasoning driven by accent

## Downloading Audio Files

The repository includes scripts to download audio files from Hugging Face datasets.

### Available Download Scripts

- `download_audio.py` - Download AfriSpeech-Parliament audio files
- `download_afrispeech200_audio.py` - Download AfriSpeech-200 audio files
- `download_afri_names_audio.py` - Download AfriNames audio files
- `download_and_separate_dialog.py` - Download and separate AfriSpeech-Dialog (medical/general)

### Usage Example

```bash
# Download AfriSpeech-Parliament audio files
python download_audio.py YOUR_HF_TOKEN

# Download AfriSpeech-200 audio files
python download_afrispeech200_audio.py YOUR_HF_TOKEN

# Download and separate AfriSpeech-Dialog
python download_and_separate_dialog.py YOUR_HF_TOKEN

# Separate already-downloaded audio files by domain
python separate_audio_files.py
```

**Note:** Audio files are automatically excluded from git via `.gitignore`. They will be downloaded to the `Audio/` directory.

## Generating Hypotheses

Hypotheses are generated using LLM runners. Each dataset has corresponding runners for different tasks.

### Available Runners

#### Parliament Dataset
- `run_llama_parliament_entailment.py` - NLI hypotheses
- `run_llama_parliament_consistency.py` - Consistency statements
- `run_llama_parliament_intent.py` - Intent inference
- `run_llama_parliament_commonsense.py` - Commonsense reasoning

#### Medical Dataset
- `run_llama_medical_nli.py` - NLI hypotheses
- `run_llama_medical_consistency.py` - Consistency statements
- `run_llama_medical_plausibility.py` - Plausibility judgments

#### AfriSpeech-200 Dataset
- `run_llama_afrispeech200_nli.py` - NLI hypotheses
- `run_llama_afrispeech200_consistency.py` - Consistency statements
- `run_llama_afrispeech200_plausibility.py` - Plausibility judgments

#### AfriSpeech-General Dataset
- `run_llama_afrispeech_general_consistency.py` - Consistency statements
- `run_llama_afrispeech_general_plausibility.py` - Plausibility judgments

#### AfriNames Dataset
- `run_llama_afri_names_restraint.py` - Semantic restraint evaluation
- `run_llama_afri_names_accent.py` - Accent drift evaluation

### Usage Example

```bash
# Generate NLI hypotheses for Parliament dataset
python -m Entailment.models.runners.run_llama_parliament_entailment \
  --csv_path Entailment/metadata_afrispeech-parliament.csv \
  --output_dir result/Entailment/Parliament/Llama/entailment_hypotheses \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct

# Generate consistency hypotheses for Medical dataset
python -m Entailment.models.runners.run_llama_medical_consistency \
  --csv_path Entailment/metadata_medical.csv \
  --output_dir result/Entailment/Medical/Llama/consistency

# Generate restraint hypotheses for AfriNames
python -m Entailment.models.runners.run_llama_afri_names_restraint \
  --csv_path Entailment/metadata_afri-names.csv \
  --output_dir result/Entailment/AfriNames/Llama/restraint
```

### Output Format

Each runner generates a `.jsonl` file with one JSON object per row:

```json
{
  "file_name": "data/example.wav",
  "transcript": "Speaker transcript...",
  "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "output": {
    "entailment": ["hypothesis 1", "hypothesis 2", "hypothesis 3"],
    "neutral": ["hypothesis 1", "hypothesis 2", "hypothesis 3"],
    "contradiction": ["hypothesis 1", "hypothesis 2", "hypothesis 3"]
  }
}
```

## Running Model Inference

### Model Requirements

- **Recommended:** 4–6 audio language models spanning:
  - Proprietary vs open-source
  - Speech-first vs multimodal
  - Different training paradigms

### Hypothesis Requirements

Multiple hypotheses per label are strongly recommended:

| Task | Recommendation |
|------|----------------|
| NLI | 2–3 hypotheses per class |
| Consistency | 2 variants per label |
| Intent | 1–2 per intent |
| Plausibility | 2 per label |
| AfriNames diagnostics | Multiple neutral controls |

**Why multiple hypotheses?**
- Reduces prompt sensitivity
- Enables variance analysis
- Prevents cherry-picking effects

### Model Prompting Philosophy

Across all tasks:
- Models receive **audio + text hypothesis**
- **No transcript access** during inference
- **No dataset metadata** exposed
- Prompts are **task-specific but dataset-agnostic**

This ensures fair comparison across datasets and models.

## Evaluation

The benchmark evaluates:

- **Accuracy** (where applicable)
- **Calibration and confidence**
- **Error asymmetry** across accents/domains
- **Hallucination rate** on low-information audio
- **Consistency** across hypothesis variants

### Evaluation Focus

1. **Failure modes** - Where and why models fail
2. **Over-inference patterns** - When models hallucinate
3. **Robustness gaps** - Domain and accent-specific weaknesses

## Hypothesis Construction

For each audio clip:

1. The transcript is used **only to generate hypotheses**
2. Hypotheses are:
   - **Abstract** - Not verbatim quotes
   - **Non-verbatim** - Semantic paraphrases
   - **Semantically grounded** - Reflect meaning, not wording
3. The transcript is **never exposed during inference**

Each dataset CSV includes:
- `file_name` - Path to audio file
- `transcript` - Ground truth transcript (for annotation only)
- Task-specific hypothesis columns (generated by runners)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{afro_entailment,
  title={Afro_entailment: Audio–Semantic Reasoning Benchmark for Speech-Based Audio Language Models},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.

---

**Note:** Audio files and large datasets are excluded from git via `.gitignore`. Download them using the provided scripts before running inference.
