# Prompt Documentation

This document contains the prompts used for hypothesis generation (annotation) and model inference (evaluation) in the Afro_entailment benchmark.

## Table of Contents

- [LLM Generation Prompts](#llm-generation-prompts)
  - [AfriNames Dataset](#afrinames-dataset)
  - [AfriSpeech–Parliament Dataset](#afrispeechparliament-dataset)
  - [Medical Dataset](#medical-dataset)
  - [AfriSpeech-200 Dataset](#afrispeech-200-dataset)
  - [General Dataset](#general-dataset)
- [Audio Language Model Inference Prompts](#audio-language-model-inference-prompts)
- [Methodological Notes](#methodological-notes)

---

## LLM Generation Prompts

These prompts are used by LLMs (e.g., Llama, Mistral) to generate hypotheses from transcripts. The generated hypotheses are then used for model evaluation during inference.

---

### AfriNames Dataset

The AfriNames dataset uses two specialized prompts for generating diagnostic hypotheses that test semantic restraint and fairness under accent variation.

#### Prompt 1: Semantic Restraint / Over-Inference

**Purpose:** Generate hypotheses to test whether models can correctly withhold inference when audio content is minimal.

```
Semantic Restraint / Over-Inference Hypothesis Generation
(Hallucination & Overgeneralization Evaluation)
You are a careful evaluator of spoken language meaning.

You are given a verbatim transcript of a very short spoken audio recording.
The transcript represents all meaningful semantic information available from the audio.
The transcript is provided only for annotation purposes.

This audio contains LIMITED semantic content.
Your task is to generate hypotheses that test whether a model can
correctly WITHHOLD inference when the audio does not provide enough evidence.

TASK:
Generate hypotheses for evaluating semantic restraint and hallucination.

Generate the following:

1) SUPPORTED hypotheses (2 items)
   – Statements that are clearly supported by the meaning of the audio.
   – These should be minimal and cautious.

2) UNSUPPORTED hypotheses (6 items)
   – Statements that may sound reasonable in general speech datasets,
     but are NOT supported by the meaning of this audio.
   – These hypotheses should tempt a model to over-infer.

CONSTRAINTS (must follow strictly):
- Do NOT quote or closely paraphrase the transcript.
- Do NOT describe what the speaker "says" or "mentions."
- Do NOT add roles, topics, locations, intent, emotions, or domains
  unless they are explicitly supported.
- Each hypothesis must be one complete sentence.
- Avoid explicit negation terms such as "no," "not," "never," or "without."

GOAL:
Unsupported hypotheses should test for:
- Invented topics
- Invented intent
- Invented social roles
- Invented domains (e.g., politics, health, finance)
- Invented situational context

OUTPUT FORMAT (JSON only):

{
  "supported": [
    "...",
    "..."
  ],
  "unsupported": [
    "...",
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}

TRANSCRIPT:
{{TRANSCRIPT}}
```

**What This Prompt Provides (Methodologically):**

- **Hallucination rate** - Measures how often models infer unsupported claims
- **Over-inference rate** - Quantifies when models go beyond available evidence
- **Clean signal for semantic restraint** - Tests model ability to withhold judgment
- **Works perfectly for short utterances** - Designed for minimal semantic content

#### Prompt 2: Accent-Conditioned Semantic Drift

**Purpose:** Generate hypotheses to test whether models infer different meanings from similar utterances spoken with different accents.

```
Accent-Conditioned Semantic Drift Hypothesis Generation
(Fairness & Robustness Under Accent Variation)
You are evaluating semantic stability in spoken language understanding.

You are given a verbatim transcript of a short spoken audio recording.
The transcript represents the meaning of the audio and is provided only for annotation.

IMPORTANT:
This task is designed to test whether models infer DIFFERENT meanings
from similar short utterances spoken with different accents or pronunciations.

TASK:
Generate hypotheses that should be interpreted IDENTICALLY
regardless of the speaker's accent, pronunciation, or speaking style.

Generate the following:

1) ACCENT-INVARIANT hypotheses (4 items)
   – Statements whose truth should NOT depend on accent or speaker identity.
   – These should rely only on the semantic content of the utterance.

2) ACCENT-SENSITIVE LURE hypotheses (4 items)
   – Statements that models might incorrectly infer due to accent,
     pronunciation, or training bias.
   – These should NOT be inferable from the transcript alone.

CONSTRAINTS (must follow strictly):
- Do NOT quote or closely paraphrase the transcript.
- Do NOT assume speaker nationality, ethnicity, gender, age, or location.
- Do NOT assume intent or topic unless explicitly supported.
- Each hypothesis must be one complete sentence.
- Avoid explicit negation terms such as "no," "not," "never," or "without."

GOAL:
Accent-sensitive lures should test whether a model:
- Assigns social or institutional roles
- Infers geographic or cultural background
- Injects domain meaning
- Changes interpretation due to accent alone

OUTPUT FORMAT (JSON only):

{
  "accent_invariant": [
    "...",
    "...",
    "...",
    "..."
  ],
  "accent_sensitive_lures": [
    "...",
    "...",
    "...",
    "..."
  ]
}

TRANSCRIPT:
{{TRANSCRIPT}}
```

**What This Prompt Provides:**

- **Semantic drift by accent** - Measures how accent affects semantic interpretation
- **Fairness gaps** - Tests same text with different audio realizations
- **Very clean analysis:**
  - Same hypotheses
  - Different speakers
  - Compare predictions + confidence

**How These Two Prompts Work Together:**

| Prompt | What it Tests |
|--------|---------------|
| **Prompt 1** | Can the model withhold inference? |
| **Prompt 2** | Does accent change meaning when it shouldn't? |

Both evaluate semantic restraint and accent-conditioned semantic drift under minimal spoken context.

---

### AfriSpeech–Parliament Dataset

**Tasks:**
- ✅ Spoken NLI
- ✅ Audio–Text Semantic Consistency
- ✅ Audio–Intent Inference
- ✅ Spoken Commonsense Reasoning

#### 1.1 Spoken NLI

```
You are a helpful assistant with expert knowledge in spoken language understanding, discourse analysis, and semantic inference in institutional settings such as parliamentary or legislative proceedings.

You are given a verbatim transcript of a single spoken audio recording from a formal parliamentary context. The transcript represents the semantic content conveyed by the audio and is provided only for annotation purposes.

Using the transcript and your knowledge of parliamentary discourse and language use, generate hypotheses for a Spoken Natural Language Inference task.

Instructions:
Generate three hypotheses for each category:
- Entailment: statements that are definitely true given the meaning of the spoken audio.
- Neutral: statements that might be true but cannot be determined from the audio alone.
- Contradiction: statements that are definitely false given the meaning of the audio.

Constraints:
- Do not quote or closely paraphrase the transcript.
- Do not state that the speaker "says" or "mentions" something.
- Hypotheses must reflect semantic inference, not surface wording.
- Do not use explicit negation terms such as "no," "not," or "never."
- Each hypothesis must be a single complete sentence.

Respond only in valid JSON with the structure:
{
  "entailment": ["...", "...", "..."],
  "neutral": ["...", "...", "..."],
  "contradiction": ["...", "...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 1.2 Semantic Consistency

```
You are assisting in the construction of an audio–text semantic consistency dataset.

You are given a transcript of a spoken parliamentary audio recording. The transcript is provided only to determine the meaning conveyed by the audio.

Task:
Generate four text statements:
- Two statements that are semantically consistent with the audio.
- Two statements that are semantically inconsistent with the audio.

Constraints:
- Statements should concern institutional roles, procedures, or formal discourse.
- Do not quote or restate the transcript.
- Do not use explicit negation words.
- Each statement must be one sentence.

Respond only in JSON:
{
  "consistent": ["...", "..."],
  "inconsistent": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 1.3 Audio–Intent Inference

```
You are given a transcript of a spoken audio recording from a parliamentary or legislative setting.

The transcript represents the meaning of the spoken audio and is provided only for annotation purposes.

Task:
Infer the primary communicative intent of the speaker.

Generate three intent statements describing what the speaker is trying to accomplish, such as supporting a proposal, providing justification, requesting clarification, or contributing to formal proceedings.

Constraints:
- Do not quote the transcript.
- Do not describe literal wording.
- Each intent must be expressed as a single sentence.

Respond only in JSON:
{
  "intent": ["...", "...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 1.4 Spoken Commonsense Reasoning

```
You are given a transcript of a spoken audio recording from a formal parliamentary context.

The transcript reflects the semantic content conveyed by the audio and is provided only for annotation.

Task:
Generate three commonsense inferences that a typical listener could reasonably make based on shared knowledge of parliamentary procedures and institutional norms, even if these facts are not explicitly stated.

Constraints:
- Inferences must rely on social or institutional knowledge.
- Do not quote or paraphrase the transcript.
- Do not use explicit negation.
- Each inference must be one sentence.

Respond only in JSON:
{
  "commonsense_inference": ["...", "...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

---

### Medical Dataset

**Tasks:**
- ✅ Spoken NLI
- ✅ Semantic Consistency
- ✅ Semantic Plausibility Judgment

#### 2.1 Spoken NLI

```
You are a helpful assistant with expertise in spoken medical communication and semantic inference.

You are given a transcript of a spoken medical audio recording. The transcript represents the meaning conveyed by the audio and is provided only for annotation purposes.

Task:
Generate hypotheses for a Spoken Natural Language Inference task.

Generate:
- Three entailed hypotheses
- Three neutral hypotheses
- Three contradictory hypotheses

Constraints:
- Hypotheses must focus on medical facts, procedures, or advice.
- Do not quote or closely paraphrase the transcript.
- Do not use explicit negation terms.
- Each hypothesis must be one sentence.

Respond only in JSON:
{
  "entailment": ["...", "...", "..."],
  "neutral": ["...", "...", "..."],
  "contradiction": ["...", "...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 2.2 Semantic Consistency

```
You are given a transcript of a spoken medical audio recording.

Task:
Generate four text statements:
- Two that are semantically consistent with the audio.
- Two that are semantically inconsistent with the audio.

Constraints:
- Statements should concern medical knowledge, procedures, or guidance.
- Avoid quoting the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "consistent": ["...", "..."],
  "inconsistent": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 2.3 Semantic Plausibility Judgment

```
You are given a transcript of a spoken medical audio recording.

Task:
Generate four statements:
- Two plausible statements that align with the medical context of the audio.
- Two implausible statements that are unlikely given the audio context.

Constraints:
- Plausibility judgments may rely on medical commonsense.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "plausible": ["...", "..."],
  "implausible": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

---

### AfriSpeech-200 Dataset

**Tasks:**
- ✅ Spoken NLI
- ✅ Semantic Consistency
- ✅ Semantic Plausibility Judgment

(General-purpose spoken content)

#### 3.1 Spoken NLI

```
You are given a transcript of a spoken audio recording from a general-purpose speech dataset.

The transcript represents the meaning conveyed by the audio and is provided only for annotation.

Task:
Generate hypotheses for a Spoken Natural Language Inference task.

Generate:
- Three entailed hypotheses
- Three neutral hypotheses
- Three contradictory hypotheses

Constraints:
- Hypotheses should reflect semantic understanding rather than literal wording.
- Do not quote or closely paraphrase the transcript.
- Do not use explicit negation.
- Each hypothesis must be one sentence.

Respond only in JSON:
{
  "entailment": ["...", "...", "..."],
  "neutral": ["...", "...", "..."],
  "contradiction": ["...", "...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 3.2 Semantic Consistency

```
You are given a transcript of a spoken audio recording.

Task:
Generate four statements:
- Two semantically consistent with the audio.
- Two semantically inconsistent with the audio.

Constraints:
- Statements should be grounded in the general meaning of the speech.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "consistent": ["...", "..."],
  "inconsistent": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 3.3 Semantic Plausibility

```
You are given a transcript of a spoken audio recording.

Task:
Generate four statements:
- Two plausible given the content and context of the audio.
- Two implausible given the content and context.

Constraints:
- Plausibility may rely on everyday commonsense.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "plausible": ["...", "..."],
  "implausible": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

---

### General Dataset

**Tasks:**
- ✅ Semantic Consistency
- ✅ Semantic Plausibility Judgment

#### 4.1 Semantic Consistency

```
You are given a transcript of a short spoken audio recording from a general-domain dataset.

Task:
Generate four statements:
- Two that are semantically consistent with the audio.
- Two that are semantically inconsistent with the audio.

Constraints:
- Statements should remain high-level and avoid specific assumptions.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "consistent": ["...", "..."],
  "inconsistent": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

#### 4.2 Semantic Plausibility

```
You are given a transcript of a short spoken audio recording.

Task:
Generate four statements:
- Two plausible statements.
- Two implausible statements.

Constraints:
- Statements should rely on general commonsense.
- Avoid adding unnecessary detail.
- Do not quote the transcript.
- Do not use explicit negation.
- Each statement must be one sentence.

Respond only in JSON:
{
  "plausible": ["...", "..."],
  "implausible": ["...", "..."]
}

Transcript:
{{TRANSCRIPT}}
```

---

## Audio Language Model Inference Prompts

The model inference prompts should be **task-generic, not dataset-specific**. You vary:
- The hypothesis
- The audio
- **Not** the reasoning instructions

These canonical prompts work across all datasets in the benchmark.

### Spoken NLI Prompt

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
```

### Semantic Consistency Prompt

```
You are given an audio recording and a text statement.

Determine whether the text is consistent with the meaning conveyed in the audio.

Respond with one of the following labels only:
CONSISTENT
INCONSISTENT
```

### Intent Inference Prompt

```
You are given an audio recording of spoken speech.

Which of the following best describes the speaker's communicative intent?

Choose the single best answer from the provided options.
```

(Options come from your CSV.)

### Plausibility Prompt

```
You are given an audio recording and a text statement.

Is the statement plausible given what is heard in the audio?

Respond with:
PLAUSIBLE
IMPLAUSIBLE
```

### Commonsense Reasoning Prompt

```
You are given an audio recording of spoken speech.

Based on common knowledge and the content of the audio, answer the following question.

Respond with:
YES
NO
```

---

## Methodological Notes

### Why This Prompt Strategy Is Good

1. **Minimal instruction bias** - Prompts are task-focused, not dataset-specific
2. **Forces grounding in audio** - Models must rely on audio content, not external knowledge
3. **Works across all datasets** - Same prompt template applies universally
4. **Easy to reuse across models** - Consistent evaluation framework
5. **Easy to audit** - Transparent and reproducible prompting strategy

### Strategic Recommendation for First Paper

For your first paper, we strongly recommend focusing on:

**Main Results:**
- Spoken NLI
- Audio–Text Semantic Consistency

**Secondary Analysis:**
- Intent (AfriSpeech–Parliament, AfriSpeech-200)
- Plausibility (Medical, General)
- Commonsense (AfriSpeech–Parliament only)

This keeps the story tight, coherent, and defensible while demonstrating the core capabilities of audio-semantic reasoning.
