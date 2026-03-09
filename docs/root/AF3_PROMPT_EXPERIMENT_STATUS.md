# AudioFlamingo3 Alternative Prompts Experiment

## Jobs Submitted:
- **Job 24273211**: AudioFlamingo3_v2 (Option 2 - Few-shot examples)
- **Job 24273231**: AudioFlamingo3_v3 (Option 3 - Explicit reasoning)  
- **Job 24273232**: AudioFlamingo3_v4 (Option 4 - Simple direct)

## Prompt Details:

### Version 2 (Option 2): Few-shot style with examples
```
You will listen to an audio clip and evaluate a statement about it.

Examples:
- If audio says "I love swimming" and statement is "The speaker enjoys water activities" → ENTAILMENT
- If audio says "I hate vegetables" and statement is "The speaker loves healthy food" → CONTRADICTION  
- If audio says "It's sunny today" and statement is "The speaker has brown eyes" → NEUTRAL

Now evaluate this audio clip:
Statement: {hypothesis}

Classification: [ENTAILMENT/CONTRADICTION/NEUTRAL]
```

### Version 3 (Option 3): More explicit reasoning
```
Listen carefully to the audio clip.

Statement to evaluate: {hypothesis}

Step 1: What is the main content/topic of the audio?
Step 2: Does the audio content relate to the statement?
Step 3: If related, does it support, contradict, or remain neutral toward the statement?

Final answer (choose one): ENTAILMENT | CONTRADICTION | NEUTRAL
```

### Version 4 (Option 4): Simpler, more direct
```
Audio content vs Statement: {hypothesis}

The relationship is:
- ENTAILMENT (audio supports the statement)
- CONTRADICTION (audio opposes the statement)
- NEUTRAL (audio is unrelated or insufficient)

Answer:
```

## Expected Outputs:
- `/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo3_v2/interview_nli/results/AudioFlamingo3_v2_interview_nli.jsonl`
- `/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo3_v3/interview_nli/results/AudioFlamingo3_v3_interview_nli.jsonl`
- `/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/AudioFlamingo3_v4/interview_nli/results/AudioFlamingo3_v4_interview_nli.jsonl`

## Next Steps After Jobs Complete:
1. Check job completion: `squeue -u c.okocha`
2. Check results distribution for each version
3. Convert to CSV files:
   ```bash
   cd /orange/ufdatastudios/c.okocha/afrispeech-entailment
   
   # For v2
   python3 convert_to_difficulty_csv.py \
       --jsonl outputs/AudioFlamingo3_v2/interview_nli/results/AudioFlamingo3_v2_interview_nli.jsonl \
       --output outputs/AudioFlamingo3_v2/interview_nli/interview_nli_results.csv \
       --task interview_nli --dataset interview --alm AudioFlamingo3_v2
   
   # For v3
   python3 convert_to_difficulty_csv.py \
       --jsonl outputs/AudioFlamingo3_v3/interview_nli/results/AudioFlamingo3_v3_interview_nli.jsonl \
       --output outputs/AudioFlamingo3_v3/interview_nli/interview_nli_results.csv \
       --task interview_nli --dataset interview --alm AudioFlamingo3_v3
   
   # For v4  
   python3 convert_to_difficulty_csv.py \
       --jsonl outputs/AudioFlamingo3_v4/interview_nli/results/AudioFlamingo3_v4_interview_nli.jsonl \
       --output outputs/AudioFlamingo3_v4/interview_nli/interview_nli_results.csv \
       --task interview_nli --dataset interview --alm AudioFlamingo3_v4
   ```

## Comparison with Original:
- **Original AF3**: 90% ENTAILMENT bias (187/207 predictions)
- **Goal**: Test if different prompts can reduce bias and produce more balanced predictions

The new prompts aim to:
- Remove "stuttering" context that might bias responses
- Provide clear examples (v2) or structured reasoning (v3)
- Use simpler, more direct language (v4)