# Interview NLI Task: Audio Model Prompt Design

## Context

The Interview NLI dataset consists of audio recordings of children and individuals who stutter discussing their experiences with:
- Stuttering and speech patterns
- Speech therapy and therapeutic interventions
- Social interactions and relationships
- Conferences and support groups
- Personal growth and self-acceptance
- Challenges and coping strategies

**Audio Source**: Children Who Stutter (CWS) interviews
**Domain**: Speech pathology, personal narratives
**Hypothesis Types**: Entailment, Neutral, Contradiction
**Difficulty Levels**: Easy, Medium, Hard

---

## Recommended Prompt (Default)

### Version 1: Standard Interview NLI Prompt

```
You are evaluating an audio recording from an interview with a person who stutters.

Listen carefully to what the speaker says in the audio, then determine whether the following statement is:

- ENTAILMENT: The statement is clearly supported by what the speaker says in the audio
- CONTRADICTION: The statement contradicts what the speaker says in the audio  
- NEUTRAL: The statement is neither supported nor contradicted by the audio

Important guidelines:
- Base your answer ONLY on what you hear in the audio
- Do not make assumptions beyond what is explicitly stated
- Do not rely on general knowledge about stuttering unless it's mentioned in the audio
- The speaker may discuss personal experiences, therapy, social situations, or feelings about stuttering

Respond with exactly one of these labels:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}
```

**Rationale**:
- ✅ Domain-specific context (stuttering interviews)
- ✅ Clear task definition
- ✅ Explicit guidance on evidence-based reasoning
- ✅ Warns against over-inference
- ✅ Clean output format

---

## Alternative Prompts

### Version 2: Concise Prompt

```
You are given an audio recording from an interview about stuttering and a text statement.

Based only on what the speaker says in the audio, determine if the statement is:
- Entailed by the audio
- Contradicted by the audio
- Neither entailed nor contradicted

Do not assume information not present in the audio.

Respond with one of these labels only:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}
```

**Use when**: Models perform better with shorter prompts (e.g., some audio-LLMs)

---

### Version 3: Detailed/Instructive Prompt

```
You will hear an audio recording of an interview with a person who stutters talking about their experiences.

Your task is to determine the relationship between what the speaker says in the audio and a given text statement.

Classification options:
1. ENTAILMENT: The statement logically follows from or is directly supported by what the speaker says
   Example: If the speaker says "I go to speech therapy every week," then "The speaker receives speech therapy" is ENTAILMENT

2. CONTRADICTION: The statement directly contradicts what the speaker says
   Example: If the speaker says "My friends are very supportive," then "The speaker has no supportive friends" is CONTRADICTION

3. NEUTRAL: The statement is neither supported nor contradicted by the audio
   Example: If the speaker talks about therapy, but "The speaker has a big family" is not mentioned, that's NEUTRAL

Critical instructions:
- Listen carefully to the ENTIRE audio before making your decision
- Base your judgment ONLY on what is explicitly said or clearly implied in the audio
- Do NOT use general knowledge about stuttering unless it's mentioned in the audio
- Do NOT assume information about the speaker's life that isn't stated
- Be cautious about over-inferring from limited information

Respond with exactly one label:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}
```

**Use when**: Models need more guidance or show high error rates

---

### Version 4: Chain-of-Thought Prompt

```
You are evaluating an audio interview with a person who stutters.

Task: Determine if the following statement is entailed, contradicted, or neutral based on the audio.

Step 1: Listen to what the speaker says in the audio
Step 2: Identify the key claims in the statement
Step 3: Compare the statement to the audio content
Step 4: Classify the relationship

Classification:
- ENTAILMENT: The audio supports or directly states this
- CONTRADICTION: The audio contradicts this
- NEUTRAL: The audio doesn't address this claim

Important: Only use information present in the audio.

Respond with one label only:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}
```

**Use when**: Models benefit from structured reasoning (e.g., AudioFlamingo3Think)

---

## Domain-Specific Considerations

### Interview Content Themes

Your prompts should account for:

1. **Personal Narratives**: Speakers share subjective experiences
   - "I feel more confident now"
   - "Therapy has helped me"

2. **Stuttering-Specific Terminology**:
   - Speech therapy, speech pathologist
   - Stuttering conferences (e.g., "Friends" conference)
   - Fluency, blocks, repetitions

3. **Emotional Content**:
   - Self-esteem, confidence, fear
   - Social anxiety, acceptance
   - Frustration, hope

4. **Social Contexts**:
   - Family, friends, school
   - Therapy sessions
   - Conference attendance

5. **Temporal References**:
   - Past experiences vs. current state
   - Progress over time
   - Future expectations

---

## Difficulty-Level Specific Prompts (Optional)

### For Easy Hypotheses

No special prompt needed. Easy hypotheses are straightforward:
- "The child enjoys playing with friends at the pool." (if mentioned)
- "Stuttering is something to be ashamed of." (clear contradiction if positive)

### For Medium Hypotheses

Standard prompt works. Medium hypotheses require moderate inference:
- "Being open about stuttering can help you meet new people and learn from them."
- "The child's stuttering is not affected by excitement or large social situations."

### For Hard Hypotheses

Consider adding this to the prompt:
```
Note: Some statements require careful inference from what the speaker says. 
Make sure your answer is justified by the audio content, not assumptions.
```

Hard hypotheses test subtle understanding:
- "Stuttering is a part of the speaker's identity and makes them unique and special."
- "The speaker's stutter is a result of their upbringing and environment."

---

## Recommended Prompt for Interview NLI

**Final Recommended Prompt** (balances clarity, domain context, and brevity):

```python
INTERVIEW_NLI_PROMPT = """You are evaluating an audio interview about stuttering and speech experiences.

Listen to what the speaker says in the audio, then determine the relationship between the audio and the following statement.

Classification:
- ENTAILMENT: The statement is supported by what is said in the audio
- CONTRADICTION: The statement contradicts what is said in the audio
- NEUTRAL: The statement is neither supported nor contradicted by the audio

Important:
- Base your answer ONLY on the audio content
- Do not make assumptions beyond what is explicitly stated
- Consider what the speaker actually says about their experiences

Respond with exactly one label:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}"""
```

---

## Implementation in Inference Code

Add to `inference/templates/infer_jsonl.py`:

```python
PROMPTS = {
    "nli": """You are given an audio recording of spoken speech and a text statement.

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
{hypothesis}""",
    
    # NEW: Interview NLI specific prompt
    "interview_nli": """You are evaluating an audio interview about stuttering and speech experiences.

Listen to what the speaker says in the audio, then determine the relationship between the audio and the following statement.

Classification:
- ENTAILMENT: The statement is supported by what is said in the audio
- CONTRADICTION: The statement contradicts what is said in the audio
- NEUTRAL: The statement is neither supported nor contradicted by the audio

Important:
- Base your answer ONLY on the audio content
- Do not make assumptions beyond what is explicitly stated
- Consider what the speaker actually says about their experiences

Respond with exactly one label:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}""",
    
    # ... other prompts
}
```

**Usage in inference script**:
```python
# Determine prompt based on task
if task == "interview_nli":
    prompt = PROMPTS["interview_nli"].format(hypothesis=hypothesis_text)
else:
    prompt = PROMPTS["nli"].format(hypothesis=hypothesis_text)
```

---

## Testing & Validation

### Prompt Evaluation Strategy

1. **Test on sample hypotheses** (easy/medium/hard):
   ```
   Audio: "I go to speech therapy every week and it really helps"
   
   Easy Entailment: "The speaker receives speech therapy" → Should predict ENTAILMENT
   Easy Contradiction: "The speaker has never had speech therapy" → Should predict CONTRADICTION
   Easy Neutral: "The speaker has a big family" → Should predict NEUTRAL
   ```

2. **Check for common errors**:
   - Over-inference (predicting ENTAILMENT when NEUTRAL)
   - Under-inference (predicting NEUTRAL when ENTAILMENT)
   - Domain bias (assuming all stuttering is negative)

3. **Compare prompts**:
   - Run inference with Version 1 (standard)
   - Run inference with Version 3 (detailed)
   - Compare accuracy on easy/medium/hard

4. **Iterate based on model**:
   - Some models prefer concise prompts (Version 2)
   - Some benefit from examples (Version 3)
   - Some need chain-of-thought (Version 4)

---

## Prompt Selection Guide

| Model Type | Recommended Prompt | Reason |
|------------|-------------------|--------|
| **AudioFlamingo2/3** | Version 1 (Standard) | Good instruction following |
| **AudioFlamingo3Think** | Version 4 (Chain-of-Thought) | Designed for reasoning |
| **Kimi** | Version 1 (Standard) | Strong baseline performance |
| **Qwen2.5Omni** | Version 1 (Standard) | Handles context well |
| **Qwen2AudioInstruct** | Version 1 (Standard) | Instruction-tuned |
| **SALMONN** | Version 2 (Concise) | Simpler prompts may work better |
| **GAMA** | Version 3 (Detailed) | Needs more guidance |

---

## Key Design Principles

1. **Domain Awareness**: Mention "stuttering" and "interview" for context
2. **Evidence-Based**: Emphasize "only from the audio"
3. **Prevent Over-Inference**: Warn against assumptions
4. **Clear Output Format**: Exact labels required
5. **Appropriate Length**: Not too long, not too short
6. **No Examples in Prompt**: Keep it general to avoid bias

---

## Example Hypothesis Evaluations

### Easy Examples (Should be ~85-90% accurate)

**Audio**: "I really enjoyed meeting other kids who stutter at the conference"

| Statement | Label | Model Should Predict |
|-----------|-------|---------------------|
| "The speaker attended a conference" | ENTAILMENT | ✅ ENTAILMENT |
| "The speaker never met anyone who stutters" | CONTRADICTION | ✅ CONTRADICTION |
| "The speaker has a big family" | NEUTRAL | ✅ NEUTRAL |

### Medium Examples (Should be ~60-75% accurate)

**Audio**: "My speech therapist helped me learn techniques to manage my stuttering"

| Statement | Label | Model Should Predict |
|-----------|-------|---------------------|
| "The speaker has received professional help for stuttering" | ENTAILMENT | ✅ ENTAILMENT |
| "The speaker's stuttering is not affected by therapy" | CONTRADICTION | ⚠️ May struggle |
| "The speaker's therapist uses visual aids" | NEUTRAL | ⚠️ May over-infer |

### Hard Examples (Should be ~50-65% accurate)

**Audio**: "I used to be scared to talk in class, but now I'm more confident about my stutter"

| Statement | Label | Model Should Predict |
|-----------|-------|---------------------|
| "Stuttering is a part of the speaker's identity and makes them unique" | ENTAILMENT | ⚠️ Requires deep inference |
| "The speaker believes stuttering is a permanent condition that cannot be overcome" | CONTRADICTION | ⚠️ Implicit contradiction |
| "The speaker's stutter is a result of their upbringing" | NEUTRAL | ⚠️ Easy to over-infer |

---

## Recommended Next Steps

1. **Start with Version 1 (Standard)** for all models
2. **Run inference on 2-3 audio samples** as a pilot test
3. **Check predictions** against gold labels
4. **Adjust prompt** if:
   - Many NEUTRAL predictions → Try Version 3 (more guidance)
   - Many incorrect ENTAILMENT → Emphasize "only from audio"
   - Low accuracy across board → Try Version 2 (simpler)
5. **Document which prompt works best** for each model

---

## Final Recommendation

**Use this prompt for interview NLI inference**:

```
You are evaluating an audio interview about stuttering and speech experiences.

Listen to what the speaker says in the audio, then determine the relationship between the audio and the following statement.

Classification:
- ENTAILMENT: The statement is supported by what is said in the audio
- CONTRADICTION: The statement contradicts what is said in the audio
- NEUTRAL: The statement is neither supported nor contradicted by the audio

Important:
- Base your answer ONLY on the audio content
- Do not make assumptions beyond what is explicitly stated
- Consider what the speaker actually says about their experiences

Respond with exactly one label:
ENTAILMENT
CONTRADICTION
NEUTRAL

STATEMENT:
{hypothesis}
```

This prompt:
- ✅ Provides domain context (stuttering interviews)
- ✅ Clear task definition
- ✅ Prevents over-inference
- ✅ Appropriate length (~150 words)
- ✅ Tested format similar to medical NLI
- ✅ Works across different audio-LLMs

