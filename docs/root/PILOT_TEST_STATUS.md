# Pilot Test Status: Interview NLI Prompt Validation

## ✅ Job Submitted Successfully

**Job ID**: 23445440  
**Status**: Pending (in queue)  
**Partition**: hpg-b200  
**Submitted**: January 20, 2026

## 📊 Test Configuration

### Samples
- **Audio samples**: 3 (08f, 09m, 506)
- **Total hypotheses**: 27
- **Difficulty distribution**: 9 easy, 9 medium, 9 hard
- **Label distribution**: 9 entailment, 9 contradiction, 9 neutral

### Model
- **Model**: Kimi (moonshotai/Kimi-Audio-7B-Instruct)
- **Prompt**: Version 1 - Standard (interview_nli)
- **Max tokens**: 10
- **Task**: interview_nli

### Resources
- **GPUs**: 1
- **Memory**: 48GB
- **Time limit**: 1 hour
- **Partition**: hpg-b200

## 📁 Output Locations

### Results
```
/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/results/Kimi_interview_nli_pilot.jsonl
```

### Logs
```
/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/logs/infer_23445440.out
/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/logs/infer_23445440.err
```

## 🔍 Monitoring Commands

### Check job status
```bash
squeue -u c.okocha
```

### Watch log output
```bash
tail -f /orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/logs/infer_23445440.out
```

### Run monitoring script
```bash
./monitor_pilot.sh
```

## 📝 Next Steps (After Job Completes)

### 1. Convert results to CSV
```bash
cd /orange/ufdatastudios/c.okocha/afrispeech-entailment

python3 convert_to_difficulty_csv.py \
    --jsonl /orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/results/Kimi_interview_nli_pilot.jsonl \
    --output /orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/pilot_results.csv \
    --task interview_nli \
    --dataset interview \
    --alm Kimi
```

### 2. Evaluate by difficulty
```bash
python3 evaluation_by_difficulty.py \
    --csv /orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/pilot_results.csv \
    --output_csv /orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/pilot_metrics.csv
```

### 3. Analyze results
- Review overall accuracy (target: >60%)
- Check difficulty breakdown:
  - Easy: expect ~75-85% accuracy
  - Medium: expect ~60-70% accuracy
  - Hard: expect ~40-55% accuracy
- Identify error patterns:
  - Over-inference (NEUTRAL → ENTAILMENT)
  - Audio misunderstanding
  - Label confusion

### 4. If results are good (>60% overall)
- Proceed with full batch inference on all 24 audio samples
- Generate SLURM script for full interview_nli task

### 5. If results need improvement
- Try alternative prompt (Version 2: Concise, or Version 4: Chain-of-Thought)
- Adjust max_new_tokens if responses are truncated
- Consider testing with different model (e.g., AudioFlamingo3Local)

## 📚 Files Created

1. **Entailment/interview_nli_hypotheses_PILOT.jsonl** - Pilot dataset (3 samples)
2. **inference/task_config.json** - Updated with interview_nli_pilot task
3. **inference/run_pilot_interview_nli.sh** - SLURM batch script
4. **PROMPTS.md** - Updated with interview_nli prompt
5. **inference/templates/infer_jsonl.py** - Updated with interview_nli prompt
6. **monitor_pilot.sh** - Job monitoring script
7. **add_interview_prompt_to_kimi.py** - Script to update Kimi's inference file

## 🎯 Success Criteria

- **Minimum acceptable**: 60% overall accuracy
- **Good performance**: 70%+ overall accuracy
- **Excellent performance**: 75%+ overall accuracy

Per difficulty:
- Easy: 75%+
- Medium: 60%+
- Hard: 45%+

## 📖 Related Documentation

- [INTERVIEW_NLI_PROMPT_VARIANTS.md](INTERVIEW_NLI_PROMPT_VARIANTS.md) - All 4 prompt versions
- [PROMPTS.md](PROMPTS.md) - Official prompt documentation
- [evaluation_by_difficulty.py](evaluation_by_difficulty.py) - Difficulty-based evaluation
- [convert_to_difficulty_csv.py](convert_to_difficulty_csv.py) - JSONL to CSV converter

---

**Last updated**: January 20, 2026  
**Job submitted**: ✅ Complete  
**Status**: Waiting for job to run (currently pending in queue)
