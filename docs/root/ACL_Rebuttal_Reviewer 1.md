## 1. Cascade ASR + Text-LLM Baseline

We agree that a cascade comparison is essential for isolating acoustic versus reasoning failures.

We evaluated six cascade systems combining two state-of-the-art ASR models (Whisper large-v3, IBM Granite Speech) with three instruction-tuned LLMs (Mistral, LLaMA, Qwen). We report both ASR WER and task-level F1.

### Entailment Results (Cascade)

Across datasets, entailment F1 remains moderate despite strong ASR:

- AfriSpeech200: F1 ranges **0.52–0.69**
- Medical: F1 ranges **0.32–0.63**

Notably, some configurations exhibit high entailment accuracy but low non-entailment accuracy (e.g., EACC = 1.00 but NACC ≈ 0.05–0.30), indicating persistent over-entailment even under high-quality transcription.

### Consistency Results (Cascade)

In contrast, consistency performance is consistently high:

- AfriSpeech200: **0.85–0.92 ACC**
- AfriSpeechGeneral: **≈0.97–0.98 ACC**
- Medical: **0.89–0.95 ACC**

### Interpretation

If failures were purely acoustic, strong ASR would eliminate semantic instability. Instead:

- Logical contradiction detection remains strong.
- Entailment remains unstable and class-skewed.

This indicates that over-entailment and Accent Drift cannot be explained solely by transcription degradation.

Thank you for this important observation. We agree that near-chance performance of contrastive models requires careful validation of templating and thresholding.

In our original pipeline, predictions were produced using raw cosine similarity with argmax decoding (no post-hoc calibration). To directly address your concern, we conducted a focused calibration audit across CLAP, MSCLAP, and MSCLAP_2022 outputs, including binary threshold sweeps and separability diagnostics (AUROC, d′, and score-margin analysis).

Post-hoc thresholding improves binary-task performance on average (+0.076 accuracy, +0.094 macro-F1). However, separability remains weak across most tasks. For example:

- Many tasks exhibit **AUROC ≈ 0.50** and **d′ ≈ 0**, indicating near-random class separation.
- Similarity margins are small and highly overlapping across labels.
- Only Accent Drift shows modest separability (AUROC ≈ 0.61), while Consistency and Plausibility remain near chance even after calibration.

These results indicate that while threshold optimization helps slightly, the dominant limitation is weak intrinsic class separation in global audio–text embeddings for structured reasoning tasks. In other words, the issue is architectural rather than purely templating- or threshold-related.

We now report these calibration and separability analyses explicitly and discuss this limitation transparently in comparison to end-to-end ALM and cascade baselines.
