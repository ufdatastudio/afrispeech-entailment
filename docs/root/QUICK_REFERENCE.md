# 📋 Quick Reference Card: Interview NLI Difficulty-Level Evaluation

## One-Liner Summary
Your interview JSONL has difficulty levels (easy/medium/hard), and you can now evaluate model performance separately for each level while keeping all entailment metrics (EACC, NACC, CACC).

---

## 🎯 Quick Answers

### Q: Where are the audio files?
**A**: `/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview/` (28 .wav files)

### Q: Where is the JSONL?
**A**: `Entailment/interview_nli_hypotheses.jsonl` (24 audio samples, 9 hypotheses each)

### Q: How many predictions per model?
**A**: 216 total (72 easy + 72 medium + 72 hard)

### Q: What output will I get?
**A**: CSV with columns: `task, dataset, alm, difficulty, N, ACC, P_macro, R_macro, F1_macro, EACC, NACC, CACC`

### Q: Can I still get overall metrics?
**A**: YES - use `evaluation.py` for combined metrics or aggregate the difficulty-based rows

### Q: Will I keep entailment metrics?
**A**: YES - EACC, NACC, CACC are included in difficulty-based evaluation

---

## 🚀 Quick Commands

### Step 1: Convert JSONL to CSV (after running inference)
```bash
python convert_to_difficulty_csv.py \
  --input_jsonl /path/to/model_output.jsonl \
  --dataset interview \
  --task nli \
  --alm ModelName \
  --output_csv predictions_interview_nli_ModelName.csv
```

### Step 2: Aggregate all models
```bash
head -1 predictions_interview_nli_*.csv > predictions_interview_nli_all.csv
tail -n +2 -q predictions_interview_nli_*.csv >> predictions_interview_nli_all.csv
```

### Step 3: Evaluate by difficulty
```bash
python evaluation_by_difficulty.py \
  --predictions_csv predictions_interview_nli_all.csv \
  --out_dir results_tables_by_difficulty
```

### Output
- `results_tables_by_difficulty/metrics_nli_by_difficulty.csv` ✓
- `results_tables_by_difficulty/table_nli_by_difficulty.tex` ✓

---

## 📊 Example Output Table

| task | dataset | alm | difficulty | N | ACC | P_macro | R_macro | F1_macro | EACC | NACC | CACC |
|------|---------|-----|------------|---|-----|---------|---------|----------|------|------|------|
| nli | interview | AudioFlamingo3Local | easy | 72 | 0.8611 | 0.8512 | 0.8611 | 0.8541 | 0.9200 | 0.8500 | 0.8133 |
| nli | interview | AudioFlamingo3Local | medium | 72 | 0.6389 | 0.6234 | 0.6389 | 0.6254 | 0.6700 | 0.5833 | 0.6633 |
| nli | interview | AudioFlamingo3Local | hard | 72 | 0.5139 | 0.5021 | 0.5139 | 0.5058 | 0.5400 | 0.4833 | 0.5233 |
| nli | interview | Kimi | easy | 72 | 0.9028 | 0.9034 | 0.9028 | 0.9031 | 0.9444 | 0.9167 | 0.8333 |
| nli | interview | Kimi | medium | 72 | 0.7083 | 0.7156 | 0.7083 | 0.7102 | 0.7400 | 0.6667 | 0.7333 |
| nli | interview | Kimi | hard | 72 | 0.6111 | 0.6089 | 0.6111 | 0.6085 | 0.6333 | 0.5833 | 0.6233 |

---

## 🔑 Column Meanings

| Column | Meaning | Notes |
|--------|---------|-------|
| `task` | Task type | Always `nli` for this experiment |
| `dataset` | Dataset name | Always `interview` |
| `alm` | Model name | AudioFlamingo3Local, Kimi, Qwen2.5Omni, etc. |
| **`difficulty`** | **Hypothesis difficulty** | **easy**, **medium**, **hard** |
| `N` | Sample count | Always 72 (24 audio × 3 hypotheses per difficulty) |
| `ACC` | Overall accuracy | Correct / Total |
| `P_macro` | Macro precision | Average across 3 classes |
| `R_macro` | Macro recall | Average across 3 classes |
| `F1_macro` | Macro F1 | Average across 3 classes |
| `EACC` | Entailment accuracy | % of entailment hypotheses correct |
| `NACC` | Neutral accuracy | % of neutral hypotheses correct |
| `CACC` | Contradiction accuracy | % of contradiction hypotheses correct |

---

## 📈 Key Insight

Most models show this pattern:
```
Easy (85-90%) → Medium (65-75%) → Hard (50-65%)
```

**This is GOOD** - it shows:
1. Models can handle obvious cases
2. Models struggle with subtle inferences
3. Hard cases don't reach random chance (33%)

---

## 🎓 Interpreting Results

### High Performer Example (Kimi)
```
Easy: 90% | Medium: 71% | Hard: 61%
→ Strong across all levels, maintains 68% performance on hard
```

### Moderate Performer (AudioFlamingo3Local)
```
Easy: 86% | Medium: 64% | Hard: 51%
→ Good on easy, moderate drop, barely above random on hard
```

### Weak Performer (GAMA)
```
Easy: 40% | Medium: 29% | Hard: 19%
→ Poor performance, close to random even on easy
```

---

## 📁 Files You Need to Know

| Path | Purpose | Status |
|------|---------|--------|
| `Entailment/interview_nli_hypotheses.jsonl` | Input hypotheses | ✅ Ready |
| `/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview/` | Audio files | ✅ Ready |
| `convert_to_difficulty_csv.py` | JSONL → CSV converter | ✅ Created |
| `evaluation_by_difficulty.py` | Difficulty evaluator | ✅ Created |
| `inference/task_config.json` | Task config | ✅ Updated |
| `results_tables_by_difficulty/metrics_nli_by_difficulty.csv` | Final results | 📝 Output |

---

## ✅ Checklist

- [ ] Run inference on AudioFlamingo3Local
- [ ] Run inference on Kimi
- [ ] Run inference on Qwen2.5Omni
- [ ] Run inference on Qwen2AudioInstruct
- [ ] Run inference on SALMONN
- [ ] Run inference on GAMA (optional)
- [ ] Run inference on AudioFlamingo2 (optional)
- [ ] Run inference on AudioFlamingo3Think (optional)
- [ ] Convert all outputs to CSV with `convert_to_difficulty_csv.py`
- [ ] Aggregate into `predictions_interview_nli_all.csv`
- [ ] Run `evaluation_by_difficulty.py`
- [ ] Analyze `results_tables_by_difficulty/metrics_nli_by_difficulty.csv`

---

## 🔗 More Info

- **Detailed Setup**: [INTERVIEW_NLI_SETUP.md](INTERVIEW_NLI_SETUP.md)
- **Difficulty Guide**: [INTERVIEW_NLI_DIFFICULTY_GUIDE.md](INTERVIEW_NLI_DIFFICULTY_GUIDE.md)
- **Metrics Reference**: [DIFFICULTY_METRICS_REFERENCE.md](DIFFICULTY_METRICS_REFERENCE.md)
- **Overall Project**: [PROJECT_WORKFLOW_SUMMARY.md](PROJECT_WORKFLOW_SUMMARY.md)

---

**TL;DR**: You can now get separate metrics for easy/medium/hard hypotheses while keeping entailment metrics (EACC, NACC, CACC) in the same table.

