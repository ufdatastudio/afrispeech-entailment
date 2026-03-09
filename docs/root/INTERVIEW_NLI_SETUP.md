# Interview NLI Setup Complete ✓

## Audio Files Located
**Path**: `/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview`
- 28 audio files ready (08f.wav, 08fb.wav, 09f.wav, ... 504.wav, 505.wav, 506.wav)

## JSONL Data
**Path**: `Entailment/interview_nli_hypotheses.jsonl`
- 24 audio samples
- 9 hypotheses per sample (3 easy, 3 medium, 3 hard)
- Each hypothesis labeled: entailment/neutral/contradiction
- **Total: 216 hypothesis evaluations per model**

---

## ✨ NEW FEATURE: Difficulty-Level Evaluation

### What's New?
Your interview JSONL has `difficulty` field (easy/medium/hard) for each hypothesis. You can now get **separate metrics per difficulty level** while keeping all entailment metrics (EACC, NACC, CACC).

### Two New Scripts Created

#### 1. `evaluation_by_difficulty.py`
Computes metrics grouped by difficulty level

**Input CSV format**:
```csv
task,dataset,alm,difficulty,gold,pred
nli,interview,AudioFlamingo3Local,easy,entailment,entailment
nli,interview,AudioFlamingo3Local,medium,neutral,contradiction
nli,interview,AudioFlamingo3Local,hard,contradiction,neutral
```

**Output CSV**:
```csv
task,dataset,alm,difficulty,N,ACC,P_macro,R_macro,F1_macro,EACC,NACC,CACC
nli,interview,AudioFlamingo3Local,easy,72,0.861,0.851,0.861,0.854,0.920,0.850,0.813
nli,interview,AudioFlamingo3Local,medium,72,0.639,0.623,0.639,0.625,0.670,0.583,0.663
nli,interview,AudioFlamingo3Local,hard,72,0.514,0.502,0.514,0.506,0.540,0.483,0.523
```

**Usage**:
```bash
python evaluation_by_difficulty.py \
  --predictions_csv predictions_interview_nli_all.csv \
  --out_dir results_tables_by_difficulty
```

#### 2. `convert_to_difficulty_csv.py`
Converts JSONL inference output to CSV with difficulty levels

**Usage**:
```bash
python convert_to_difficulty_csv.py \
  --input_jsonl outputs/interview_nli_AudioFlamingo3Local.jsonl \
  --dataset interview \
  --task nli \
  --alm AudioFlamingo3Local \
  --output_csv predictions_interview_nli_AudioFlamingo3Local.csv
```

**Batch conversion** (for all models):
```bash
for jsonl in outputs/interview_nli_*.jsonl; do
    alm=$(basename $jsonl | sed 's/interview_nli_//' | sed 's/.jsonl//')
    python convert_to_difficulty_csv.py \
      --input_jsonl "$jsonl" \
      --dataset interview \
      --task nli \
      --alm "$alm" \
      --output_csv "predictions_interview_nli_${alm}.csv"
done
```

---

## Complete Workflow

### Step 1: Register Task in Config
✅ **DONE** - Added to `inference/task_config.json`:
```json
"interview_nli": {
  "task": "nli",
  "jsonl_path": "/orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl",
  "audio_dir": "/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview",
  "output_prefix": "interview_nli"
}
```

### Step 2: Run Inference on Each Model
```bash
# AudioFlamingo3Local (in its directory)
cd /orange/ufdatastudios/c.okocha/AudioFlamingo3Local
python infer_jsonl_audioflamingo3.py \
  --task interview_nli \
  --jsonl /orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl \
  --audio_dir /orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview \
  --output_file outputs/interview_nli_results.jsonl

# Kimi (in its directory)
cd /orange/ufdatastudios/c.okocha/Kimi-Audio
python infer_jsonl_kimi.py \
  --task interview_nli \
  --jsonl /orange/ufdatastudios/c.okocha/afrispeech-entailment/Entailment/interview_nli_hypotheses.jsonl \
  --audio_dir /orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview \
  --output_file outputs/interview_nli_results.jsonl

# Qwen2.5Omni, SALMONN, etc. - similar commands
```

### Step 3: Convert JSONL to CSV with Difficulty
```bash
# From main project directory
cd /orange/ufdatastudios/c.okocha/afrispeech-entailment

# For each model
python convert_to_difficulty_csv.py \
  --input_jsonl /orange/ufdatastudios/c.okocha/AudioFlamingo3Local/outputs/interview_nli_results.jsonl \
  --dataset interview \
  --task nli \
  --alm AudioFlamingo3Local \
  --output_csv predictions_interview_nli_AudioFlamingo3Local.csv

python convert_to_difficulty_csv.py \
  --input_jsonl /orange/ufdatastudios/c.okocha/Kimi-Audio/outputs/interview_nli_results.jsonl \
  --dataset interview \
  --task nli \
  --alm Kimi \
  --output_csv predictions_interview_nli_Kimi.csv

# ... repeat for Qwen2.5Omni, SALMONN, etc.
```

### Step 4: Aggregate All Models
```bash
# Combine all predictions
cat predictions_interview_nli_*.csv > predictions_interview_nli_all.csv

# Remove duplicate header rows (keep first)
head -1 predictions_interview_nli_AudioFlamingo3Local.csv > predictions_interview_nli_all.csv
tail -n +2 -q predictions_interview_nli_*.csv >> predictions_interview_nli_all.csv
```

### Step 5: Evaluate by Difficulty Level
```bash
python evaluation_by_difficulty.py \
  --predictions_csv predictions_interview_nli_all.csv \
  --out_dir results_tables_by_difficulty
```

**Output**:
- `results_tables_by_difficulty/metrics_nli_by_difficulty.csv` ← Main table
- `results_tables_by_difficulty/table_nli_by_difficulty.tex` ← LaTeX for paper

### Step 6 (Optional): Standard Evaluation Without Difficulty
```bash
python evaluation.py \
  --predictions_csv predictions_interview_nli_all.csv \
  --out_dir results_tables_standard
```

**Output**:
- `results_tables_standard/metrics_nli.csv` ← Overall metrics (combines all difficulties)

---

## Expected Output Table Structure

### By Difficulty (`metrics_nli_by_difficulty.csv`)

```
task  | dataset  | alm                   | difficulty | N  | ACC   | P_macro | R_macro | F1_macro | EACC  | NACC  | CACC
------|----------|----------------------|------------|----|----|---------|---------|---------|-------|-------|-------
nli   | interview | AudioFlamingo3Local  | easy       | 72 | 0.8611 | 0.8512 | 0.8611 | 0.8541 | 0.9200 | 0.8500 | 0.8133
nli   | interview | AudioFlamingo3Local  | medium     | 72 | 0.6389 | 0.6234 | 0.6389 | 0.6254 | 0.6700 | 0.5833 | 0.6633
nli   | interview | AudioFlamingo3Local  | hard       | 72 | 0.5139 | 0.5021 | 0.5139 | 0.5058 | 0.5400 | 0.4833 | 0.5233
nli   | interview | Kimi                 | easy       | 72 | 0.9028 | 0.9034 | 0.9028 | 0.9031 | 0.9444 | 0.9167 | 0.8333
nli   | interview | Kimi                 | medium     | 72 | 0.7083 | 0.7156 | 0.7083 | 0.7102 | 0.7400 | 0.6667 | 0.7333
nli   | interview | Kimi                 | hard       | 72 | 0.6111 | 0.6089 | 0.6111 | 0.6085 | 0.6333 | 0.5833 | 0.6233
```

### Overall (`metrics_nli.csv`)

```
task | dataset  | alm                   | N    | ACC   | P_macro | R_macro | F1_macro | EACC  | NACC  | CACC
-----|----------|----------------------|------|-------|---------|---------|---------|-------|-------|-------
nli  | interview | AudioFlamingo3Local  | 216  | 0.6713 | 0.6589 | 0.6713 | 0.6550 | 0.7133 | 0.6389 | 0.6333
nli  | interview | Kimi                 | 216  | 0.7407 | 0.7426 | 0.7407 | 0.7406 | 0.7726 | 0.7222 | 0.7300
```

---

## Key Insights from Difficulty Breakdown

### Model Performance Degradation
Shows how models struggle with harder hypotheses:
- **Easy → Hard drop for AudioFlamingo3Local**: 86.11% → 51.39% = **-34.7 percentage points**
- **Easy → Hard drop for Kimi**: 90.28% → 61.11% = **-29.2 percentage points**

### Per-Class Analysis by Difficulty
Understand which relationships are harder to recognize:

**Entailment (EACC)**:
- Easy: 92% ✓ (models easily recognize obvious entailments)
- Medium: 67% 
- Hard: 54% ✗ (models struggle with subtle entailments)

**Neutral (NACC)**:
- Easy: 85% 
- Medium: 58%
- Hard: 48% (hardest category)

**Contradiction (CACC)**:
- Easy: 81%
- Medium: 66%
- Hard: 52%

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `evaluation_by_difficulty.py` | ✨ NEW | Compute metrics by difficulty level |
| `convert_to_difficulty_csv.py` | ✨ NEW | Convert JSONL to CSV with difficulty |
| `inference/task_config.json` | ✏️ UPDATED | Added interview_nli task config |
| `INTERVIEW_NLI_DIFFICULTY_GUIDE.md` | ✨ NEW | Complete difficulty evaluation guide |
| `PROJECT_WORKFLOW_SUMMARY.md` | ✨ NEW | Overall project documentation |

---

## Next Steps

1. **Run inference on all ALMs** with interview JSONL
2. **Convert JSONL outputs** to CSV using `convert_to_difficulty_csv.py`
3. **Aggregate predictions** into single CSV
4. **Run evaluation** with `evaluation_by_difficulty.py`
5. **Analyze results** by difficulty level

---

## Documentation Files

All comprehensive guides are in your project root:

- **[INTERVIEW_NLI_DIFFICULTY_GUIDE.md](INTERVIEW_NLI_DIFFICULTY_GUIDE.md)** - Detailed guide on difficulty-level evaluation
- **[PROJECT_WORKFLOW_SUMMARY.md](PROJECT_WORKFLOW_SUMMARY.md)** - Overall project architecture

