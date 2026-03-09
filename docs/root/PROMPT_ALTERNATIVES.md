## Current Prompts Analysis

### AudioFlamingo2 Current Prompt:
```
You are listening to a child discussing their experience with stuttering.
Listen carefully to the audio and then evaluate the hypothesis below.

Determine if the hypothesis is:
- ENTAILMENT: The audio clearly supports or implies the hypothesis
- CONTRADICTION: The audio clearly contradicts the hypothesis  
- NEUTRAL: The audio neither supports nor contradicts the hypothesis

Output ONLY one of these three labels:
ENTAILMENT
CONTRADICTION
NEUTRAL

Hypothesis: "{hypothesis}"
```

**Result**: 88% NEUTRAL bias (183/207 predictions)

### AudioFlamingo3 Current Prompt:
Uses `--prompt_variant "caption_reasoning"` - likely a more complex reasoning-based prompt.

**Result**: 90% ENTAILMENT bias (187/207 predictions)

## Suggested Alternative Prompts

### Option 1: Zero-shot without context priming
```
Listen to this audio clip and determine the relationship between the audio content and the following statement:

Statement: {hypothesis}

Choose exactly one:
A) ENTAILMENT - The audio supports this statement
B) CONTRADICTION - The audio contradicts this statement  
C) NEUTRAL - The audio neither supports nor contradicts this statement

Answer:
```

### Option 2: Few-shot style with examples
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

### Option 3: More explicit reasoning
```
Listen carefully to the audio clip.

Statement to evaluate: {hypothesis}

Step 1: What is the main content/topic of the audio?
Step 2: Does the audio content relate to the statement?
Step 3: If related, does it support, contradict, or remain neutral toward the statement?

Final answer (choose one): ENTAILMENT | CONTRADICTION | NEUTRAL
```

### Option 4: Simpler, more direct
```
Audio content vs Statement: {hypothesis}

The relationship is:
- ENTAILMENT (audio supports the statement)
- CONTRADICTION (audio opposes the statement)
- NEUTRAL (audio is unrelated or insufficient)

Answer:
```

### Option 5: Question format
```
After listening to the audio, answer this question:

Does the audio clip support, contradict, or have no clear relationship to this statement: "{hypothesis}"?

Reply with exactly one word: ENTAILMENT, CONTRADICTION, or NEUTRAL.
```

## Recommendations

1. **Try Option 1 first** - removes stuttering context that might bias toward NEUTRAL
2. **Try Option 4** - most concise and direct
3. **For AudioFlamingo3**, try switching from "caption_reasoning" to a simpler prompt variant
4. **Keep original results** as requested - save new results with different suffixes like `_v2`

The current prompts may be:
- Too focused on "stuttering" context (biasing AF2 toward NEUTRAL)
- Too complex or leading AF3 toward ENTAILMENT through reasoning chains