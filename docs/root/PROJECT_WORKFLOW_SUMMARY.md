# AfriSpeech-Entailment Project: Workflow & Architecture Summary

## Project Overview

**AfriSpeech-Entailment** is a comprehensive audio-semantic reasoning benchmark for evaluating how well speech-based audio language models (ALMs) can infer meaning from spoken content. The key innovation is that models are evaluated on **audio + text hypotheses**, NOT transcripts, ensuring they learn to infer from speech directly.

The project evaluates whether ALMs can:
- Infer meaning from speech (not surface cues)
- Distinguish entailment, neutrality, and contradiction from audio alone
- Maintain semantic restraint (avoid hallucination)
- Remain robust across domains and accents

---

## Key Datasets

| Dataset | Domain | Audio Dir | Size | Metadata File | Result Dir |
|---------|--------|-----------|------|---------------|-----------|
| **Interview NLI (NEW)** | Interview/Speech | `/Audio/` | ? | `Entailment/interview_nli_hypotheses.jsonl` | `result/Entailment/` |
| **AfriSpeech-200** | General African Speech | `/Audio/Afrispeech200/` | 300 samples | `metadata_afrispeech-200.csv` | `result/Entailment/AfriSpeech200/` |
| **Medical** | Medical Communication | `/Audio/medical/` | 193 samples | `metadata_medical.csv` | `result/Entailment/Medical/` |
| **AfriNames** | African Names/Speech | `/Audio/Afrinames/` | Various | `metadata_afri-names.csv` | `result/Entailment/AfriNames/` |
| **Parliament** | Parliamentary Discourse | `/Audio/` | ~7400 | `metadata_afrispeech-parliament.csv` | `result/Entailment/Parliament/` |
| **General** | General Speech | `/Audio/general/` | Various | `metadata_afrispeech-general.csv` | `result/Entailment/AfriSpeechGeneral/` |

---

## Core Task Types

### 1. **NLI (Natural Language Inference)**
- **Labels**: ENTAILMENT, NEUTRAL, CONTRADICTION
- **Difficulty levels**: easy, medium, hard
- **Example output file**: `outputs/AudioFlamingo3Local/medical_consistency_v2_explicit/results/AudioFlamingo3Local_medical_consistency_v2_explicit.jsonl`
- **Metrics**: ACC, P_macro, R_macro, F1_macro, EACC (Entailment Acc), NACC (Neutral Acc), CACC (Contradiction Acc)

### 2. **Consistency**
- **Labels**: CONSISTENT, INCONSISTENT
- **Checks if** audio content is consistent with a text statement

### 3. **Plausibility**
- **Labels**: PLAUSIBLE, IMPLAUSIBLE
- **Checks if** a statement is plausible given audio content

### 4. **Intent** (Parliament only)
- Identifies communicative intent from speech

### 5. **Commonsense** (Parliament only)
- Common sense reasoning over spoken content

---

## Workflow: From JSONL Input to Final Metrics

### **Step 1: Input Data Format**

**Interview NLI Hypotheses** (`Entailment/interview_nli_hypotheses.jsonl`):
```jsonl
{
  "audio_id": "08f",
  "premise_source": "CHI",
  "hypotheses": [
    {
      "label": "entailment",
      "difficulty": "easy|medium|hard",
      "text": "Hypothesis statement..."
    },
    {"label": "contradiction", ...},
    {"label": "neutral", ...}
  ]
}
```

**Task Config** (`inference/task_config.json`):
```json
{
  "interview_nli": {
    "task": "nli",
    "jsonl_path": "/path/to/interview_nli_hypotheses.jsonl",
    "audio_dir": "/path/to/Audio",
    "output_prefix": "interview_nli"
  }
}
```

---

### **Step 2: Batch Inference on Audio Models**

**Process**:
1. Read JSONL file with hypotheses and audio IDs
2. For each audio + hypothesis pair:
   - Load audio file
   - Run through ALM (AudioFlamingo3Local, Kimi, Qwen2.5Omni, etc.)
   - Get prediction (ENTAILMENT/NEUTRAL/CONTRADICTION)
   - Output enriched JSONL

**Inference Templates** (`inference/templates/`):
- `infer_jsonl.py` - Generic template (customizable per model)
- `infer_jsonl_audioflamingo2.py` - AudioFlamingo2-specific
- `infer_jsonl_audioflamingo3.py` - AudioFlamingo3-specific
- `infer_jsonl_clap.py`, `infer_jsonl_kimi_example.py`, etc.

**Model-Specific Inference** (lives in model folders):
- `/orange/ufdatastudios/c.okocha/AudioFlamingo3Local/` → runs `infer_jsonl_audioflamingo3.py`
- `/orange/ufdatastudios/c.okocha/Kimi-Audio/` → runs `infer_jsonl_kimi.py`
- `/orange/ufdatastudios/c.okocha/Qwen2.5Omni/` → runs inference
- etc.

**Output Format** (Example from Medical):
```jsonl
{
  "item_id": "247554f8-f233-4861-bc1a-8fc327b5d5df_2b500b633e5d5ecce35433cbbb859ddc_8bW4oSXn__hyp_0",
  "file_name": "data/247554f8-f233-4861-bc1a-8fc327b5d5df_2b500b633e5d5ecce35433cbbb859ddc_8bW4oSXn.wav",
  "audio_path": "/orange/.../Audio/medical/247554f8-f233-4861-bc1a-8fc327b5d5df_2b500b633e5d5ecce35433cbbb859ddc_8bW4oSXn.wav",
  "hypothesis": "Mr. Solomon's symptoms of cough and fever may indicate a respiratory infection.",
  "gold": "CONSISTENT",
  "pred_raw": "INCONSISTENT",
  "pred": "INCONSISTENT",
  "error": null,
  "ts": "2026-01-05T06:55:23.210212Z"
}
```

---

### **Step 3: Run Evaluation Script**

**Script**: `evaluation.py` or `custom_eval.py`

**Input**: CSV file with columns:
```
dataset, task, alm, gold, pred
```

**Command**:
```bash
python evaluation.py \
  --predictions_csv /path/to/predictions.csv \
  --out_dir results_tables_separated
```

**Output Metrics CSV**:
```csv
task,dataset,alm,N,ACC,P_macro,R_macro,F1_macro,EACC,NACC,CACC
nli,Medical,AudioFlamingo3,193,0.621761658,0.5512345679,0.5954337900,0.5160396986,0.9863013699,0.0166666667,0.7833333333
```

**Metrics Explanation**:
- `N` = Total samples
- `ACC` = Accuracy (overall)
- `P_macro` = Macro-averaged Precision
- `R_macro` = Macro-averaged Recall
- `F1_macro` = Macro-averaged F1
- `EACC` = Per-class Accuracy (Entailment)
- `NACC` = Per-class Accuracy (Neutral)
- `CACC` = Per-class Accuracy (Contradiction)

**Output Directory Structure**:
```
results_tables_separated/
├── metrics_nli_generative.csv
├── metrics_nli_constructive.csv
├── metrics_consistency_constructive.csv
├── metrics_consistency_generative.csv
├── metrics_plausibility_constructive.csv
├── metrics_plausibility_generative.csv
├── table_nli_generative.tex
├── table_nli_constructive.tex
└── ...
```

---

## Interview NLI Experiment: Next Steps

### **For the NEW Interview NLI Experiment:**

1. **Prepare Input**:
   - Input JSONL: `Entailment/interview_nli_hypotheses.jsonl` ✓ (already exists)
   - Audio files: Need to verify location (likely in `/Audio/` or subdirectory)
   - Create/update `inference/task_config.json` with interview_nli entry

2. **Run Batch Inference** on each ALM:
   ```bash
   # For AudioFlamingo3Local
   cd /orange/ufdatastudios/c.okocha/AudioFlamingo3Local
   python infer_jsonl_audioflamingo3.py \
     --task interview_nli \
     --jsonl /orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl \
     --audio_dir /orange/ufdatastudios/c.okocha/afrispeech-entailment/Audio/ \
     --output_dir outputs/
   
   # For other models (Kimi, Qwen2.5Omni, SALMONN, etc.)
   # Similar commands in their respective directories
   ```

3. **Aggregate Results**:
   - Collect all model outputs into single CSV
   - Run `evaluation.py` or `custom_eval.py`
   - Generate metrics tables and LaTeX output

4. **Example Output Path**:
   - Results: `outputs/AudioFlamingo3Local/interview_nli/results/AudioFlamingo3Local_interview_nli.jsonl`
   - Metrics: `results_tables_separated/metrics_nli_interview.csv`

---

## Key Inference Pipeline Steps

### **Generic Inference Flow** (from `inference/templates/infer_jsonl.py`):

1. **Load Configuration**:
   ```python
   config = task_config["interview_nli"]
   jsonl_path = config["jsonl_path"]
   audio_dir = config["audio_dir"]
   ```

2. **Read JSONL**:
   ```python
   for line in open(jsonl_path):
       data = json.loads(line)
       audio_id = data["audio_id"]
       hypotheses = data["hypotheses"]
   ```

3. **For Each Hypothesis**:
   ```python
   for hyp in hypotheses:
       audio_path = f"{audio_dir}/{audio_id}.wav"
       prompt = PROMPTS["nli"].format(hypothesis=hyp["text"])
       prediction = model.infer(audio_path, prompt)
       
       output_record = {
           "item_id": f"{audio_id}__{hyp_index}",
           "hypothesis": hyp["text"],
           "gold": hyp["label"],
           "pred": prediction,
           "ts": datetime.now().isoformat()
       }
   ```

4. **Write Output JSONL**:
   ```python
   with open(output_file, 'w') as f:
       for record in results:
           f.write(json.dumps(record) + '\n')
   ```

---

## Prompt Templates (from `PROMPTS.md`)

### **NLI Prompt**:
```
You are given an audio recording of spoken speech and a text statement.

Based only on the content of the audio, determine whether the statement is:
- Entailed by the audio
- Contradicted by the audio
- Neither entailed nor contradicted by the audio

Do not assume any information not present in the audio.

Respond with one of the following labels only:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}
```

### **Consistency Prompt**:
```
You are given an audio recording and a text statement.

Determine whether the text is consistent with the meaning conveyed in the audio.

Respond with one of the following labels only:
CONSISTENT
INCONSISTENT

STATEMENT:
{hypothesis}
```

---

## File Organization

```
/orange/ufdatastudios/c.okocha/afrispeech-entailment/
├── Entailment/
│   ├── interview_nli_hypotheses.jsonl          ← INPUT (your JSONL file)
│   ├── metadata_afri-names.csv
│   ├── metadata_afrispeech-200.csv
│   ├── metadata_medical.csv
│   └── models/
│       ├── runners/                             ← LLM hypothesis generation
│       ├── llms/
│       └── utils/
├── Audio/
│   ├── Afrispeech200/
│   ├── medical/
│   ├── Afrinames/
│   ├── general/
│   └── (interview audio files - TBD)
├── inference/
│   ├── task_config.json                        ← Register tasks here
│   ├── templates/
│   │   ├── infer_jsonl.py
│   │   ├── infer_jsonl_audioflamingo3.py
│   │   ├── infer_jsonl_kimi_example.py
│   │   └── ...
│   ├── generate_slurm_scripts.py
│   └── README.md
├── result/
│   └── Entailment/
│       ├── Medical/
│       ├── AfriSpeech200/
│       └── (interview_nli/ - OUTPUT GOES HERE)
├── outputs/
│   ├── AudioFlamingo3Local/
│   │   └── medical_consistency_v2_explicit/results/
│   ├── Kimi/
│   ├── Qwen2.5Omni/
│   ├── SALMONN/
│   └── (interview_nli/ - GOES HERE)
├── results_tables_separated/
│   ├── metrics_nli_generative.csv
│   ├── metrics_consistency_constructive.csv
│   └── ...
├── evaluation.py                                ← Evaluation script
├── custom_eval.py                               ← Alternative eval
├── evaluate_results.py
├── PROMPTS.md
└── README.md
```

---

## Summary

### **Input Files for Interview NLI**:
1. ✅ JSONL hypotheses: `Entailment/interview_nli_hypotheses.jsonl`
2. ✅ Audio files: Location to be confirmed
3. ✅ Task config: Update `inference/task_config.json`
4. ✅ Prompts: Already in `PROMPTS.md`

### **Execution Steps**:
1. Run inference on each ALM (AudioFlamingo3Local, Kimi, Qwen2.5Omni, SALMONN, etc.)
2. Aggregate JSONL outputs
3. Convert to CSV with columns: `[dataset, task, alm, gold, pred]`
4. Run `evaluation.py` → generates metrics CSV + LaTeX tables
5. Results appear in `results_tables_separated/` directory

### **Output**:
- Per-model JSONL files in `outputs/{ModelName}/interview_nli/results/`
- Aggregated metrics in `results_tables_separated/metrics_nli_interview.csv`
- Per-class accuracies (EACC, NACC, CACC) for NLI tasks

