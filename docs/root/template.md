Please use markdown format. Here is the given a template. You can use AI to help collect ideas but your review should be written by you and should not look like AI output

# <7105> <Are Vision Language Models Cross-Cultural Theory of Mind Reasoners?>
## Author:

## Paper Summary*

This paper tackles an important and timely question: whether modern Vision-Language Models (VLMs) genuinely perform cross-cultural Theory of Mind (ToM) reasoning or merely approximate it through language priors and safety-aligned heuristics. The authors introduce a new benchmark, CulturalToM-VQA, and provide a multi-angle evaluation including task, complexity, cultural region, and visual ablation analyses. The work is strong in motivation, dataset construction effort, and diagnostic experimentation, but also exhibits methodological and conceptual limitations that temper the strength of its claims.


## Summary Of Strengths*

1. Clear and Compelling Motivation

The paper is well-motivated. It correctly identifies that most VLM and ToM benchmarks are Western-centric and that social reasoning is culturally mediated rather than universal. The framing of “illusion of empathy” and “social desirability bias” is conceptually strong and aligns with current discourse in AI alignment and evaluation. The introduction clearly communicates why cross-cultural ToM is both technically and ethically important 

2. Novel Benchmark Contribution

The creation of CulturalToM-VQA (5,095 items) is a meaningful contribution. The benchmark is not merely a dataset of cultural artifacts; it attempts to probe mental-state inference in culturally grounded scenes, which is rarer in existing work. The structured taxonomy of six ToM tasks × four complexity levels is a thoughtful design decision that goes beyond flat VQA datasets 

3. Multi-Dimensional Evaluation

The evaluation is commendably broad:

- Model generation comparison (2023–2025 models)

- Prompting strategies (Zero-Shot, Zero-Shot CoT, Compositional CoT)

- Task-wise and complexity-wise breakdowns

- Cross-cultural region analysis

- Visual ablation and “blind” baselines

- Lexical shortcut exclusion

- Social desirability bias quantification

This layered diagnostic approach is a major strength. Many papers stop at aggregate accuracy; this paper goes further to analyze why models succeed or fail.

4. Visual Ablation and Bias Analysis

The visual ablation experiments are arguably the strongest section. Showing that frontier models retain very high accuracy even without images is a powerful finding, directly supporting the claim that performance may stem from parametric priors rather than true visual grounding 

The explicit quantification of positivity drift and “social desirability bias” is also a valuable contribution. Few evaluation papers operationalize bias with this level of concreteness.

5. Transparent Limitations and Ethics Section

The paper does an unusually good job acknowledging its own weaknesses—image necessity ambiguity, inter-question dependence, English-only evaluation, model involvement in dataset construction, and potential training data leakage.


## Summary Of Weaknesses*
1. Dataset Dependence and Scene Reuse

A central limitation is that 5,095 questions are derived from only 394 images. This introduces strong inter-item correlation and allows models to exploit recurring scene archetypes or scripts rather than demonstrating genuine generalization. The authors acknowledge this, but it significantly weakens claims about cross-cultural reasoning robustness. 

In effect, the benchmark risks measuring pattern familiarity within curated scene clusters rather than transferable ToM competence.

2. English-Only Evaluation Undermines “Cross-Cultural” Claim

Although the dataset is culturally diverse in imagery, all questions are in English. This creates a mismatch: the benchmark evaluates culturally themed content through a single linguistic lens. The current design conflates cultural imagery understanding with English-mediated reasoning 


3. Ambiguity of Image Necessity

Human raters only achieved moderate agreement on whether an image was strictly necessary. This is a critical issue because the core claim is about visually grounded ToM. If many questions are answerable via commonsense text priors, then visual ablation results become less surprising and less diagnostic. The paper frames this as an inherent ToM property, but it also reveals a design tension between “situated context” and “visual necessity” 


4. Over-Interpretation of High Accuracy

The paper reports >93% accuracy for frontier models and “near-human” performance, yet later shows that much of this can be approximated without images and partially explained by positivity heuristics. There is a rhetorical tension between celebrating near-human performance and demonstrating that the task may not require deep visual reasoning. The conclusion could be more cautious and less dual-toned 

5. Static Image Limitation for False Belief Tasks

False belief reasoning is inherently temporal and often narrative. Evaluating it through static images plus text may structurally disadvantage both humans and models or produce artifacts. The authors justify this as multimodal integration difficulty, but the construct validity of static false-belief tasks remains debatable.

6. Cultural Representation Granularity

Although the dataset includes multiple countries, the notion of “culture” appears largely national or regional. In reality, cultural norms vary by subculture, class, religion, and context. The benchmark risks cultural essentialism, which the ethics section itself acknowledges. While unavoidable to some degree, the taxonomy may still oversimplify lived cultural complexity.


## Comments Suggestions And Typos*


Confidence*
5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.



## Soundness*

3.5


## Excitement*
4 = Exciting: I would mention this paper to others and/or make an effort to attend its presentation in a conference.


## Overall Assessment*
3 = Findings: I think this paper could be accepted to the Findings of the ACL.


## Ethical Concerns*
None

## Limitations And Societal Impact



## Needs Ethics Review
No



## Reproducibility*
4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.



## Datasets*
3 = Potentially useful: Someone might find the new datasets useful for their work.

