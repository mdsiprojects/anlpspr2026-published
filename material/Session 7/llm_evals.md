# LLM Evaluation: Concepts, Methods, and Tools

**Session 7 — ANLP SPR 2026 (7 September 2026)**
*Reference document for lecture content and notebook development*

---

## 1. Why LLM Evaluation is Hard

Evaluating LLMs is fundamentally different from evaluating traditional software or classical ML models. Several properties make it uniquely challenging:

**Non-determinism.** LLMs produce different outputs for the same input depending on temperature and sampling settings. A single test run is not representative — evaluation requires averaging over multiple samples or controlling for determinism (temperature = 0).

**Task diversity.** A single LLM is expected to perform translation, summarisation, reasoning, coding, question answering, and creative writing — often simultaneously. No single metric captures quality across all tasks.

**Ground truth ambiguity.** For open-ended tasks (e.g. "write a summary of this article"), there is no single correct answer. Multiple valid responses exist, making automatic scoring against a reference unreliable.

**Sensitivity to prompt wording.** As covered in the prompting section, small changes in how a question is phrased can produce very different outputs, making evaluation results prompt-specific rather than model-specific.

**Rapid capability growth.** Models improve fast. Benchmarks that were hard in 2022 are saturated by 2024. Evaluation frameworks must evolve alongside the models they measure.

**Goodhart's Law.** Once a benchmark becomes a target, it ceases to be a good measure. Models trained or fine-tuned to maximise benchmark scores may not generalise — a phenomenon called *benchmark contamination* or *overfitting to leaderboards*.

---

## 2. Types of Evaluation

### 2.1 Automatic vs Human Evaluation

| | Automatic | Human |
|---|---|---|
| **Speed** | Fast, scalable | Slow, expensive |
| **Consistency** | Deterministic | Variable (inter-annotator disagreement) |
| **Coverage** | Limited to what metrics can capture | Can assess nuance, tone, coherence |
| **Bias** | Metric-specific biases | Human biases and fatigue |
| **Use case** | Regression testing, large-scale benchmarking | Final model selection, safety evaluation |

In practice, both are used together. Automatic metrics provide fast feedback during development; human evaluation validates results before deployment.

### 2.2 Intrinsic vs Extrinsic Evaluation

- **Intrinsic evaluation** measures properties of the model's output directly — e.g. fluency, coherence, factual accuracy — independent of any downstream application.
- **Extrinsic evaluation** measures how well the model performs on a real downstream task — e.g. does using this LLM improve customer support resolution rates?

Intrinsic evaluation is easier to standardise; extrinsic evaluation is more meaningful in practice.

### 2.3 Reference-based vs Reference-free Evaluation

- **Reference-based**: compare model output against a gold-standard human-written reference (e.g. BLEU, ROUGE). Requires human-annotated data.
- **Reference-free**: evaluate output quality without a reference, either using heuristics, classifiers, or another model as judge (e.g. LLM-as-Judge, reward models). More flexible but less objective.

---

## 3. Classical NLP Metrics

These metrics predate LLMs but remain widely used, especially for translation and summarisation tasks.

### 3.1 Perplexity

Perplexity measures how well a language model predicts a test corpus. Lower perplexity = the model finds the text more probable = better fit.

$$\text{Perplexity}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N}$$

Equivalently, it is the exponentiated average negative log-likelihood per token:

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1})\right)$$

- **Interpretation**: if perplexity = 10, the model is as uncertain as if it had to choose uniformly among 10 equally likely words at every step.
- **Limitation**: perplexity measures fluency and language fit — not factual accuracy, helpfulness, or instruction-following.
- **Use**: comparing models on the same held-out corpus; tracking improvement during pre-training.

### 3.2 BLEU (Bilingual Evaluation Understudy)

BLEU measures the overlap of n-grams between a model's output and one or more reference translations. Originally designed for machine translation (Papineni et al., 2002).

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ = modified n-gram precision (fraction of model n-grams found in any reference)
- $w_n$ = weight for each n-gram order (typically uniform: $w_n = 1/N$)
- BP = brevity penalty (penalises outputs that are too short)

**Limitations:**
- Only measures surface-level overlap — synonyms are penalised even if semantically correct
- Correlates poorly with human judgement for open-ended generation tasks
- Not suitable for evaluating conversational or instruction-following outputs

**Use**: machine translation, text summarisation benchmarks.

### 3.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE measures recall-oriented n-gram overlap between generated summaries and reference summaries (Lin, 2004).

Key variants:
- **ROUGE-N**: n-gram overlap (ROUGE-1 = unigrams, ROUGE-2 = bigrams)
- **ROUGE-L**: longest common subsequence — captures sentence-level structure

$$\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{n\text{-gram} \in \text{ref}} \text{Count}_\text{match}(n\text{-gram})}{\sum_{\text{ref}} \sum_{n\text{-gram} \in \text{ref}} \text{Count}(n\text{-gram})}$$

**Use**: summarisation benchmarks (CNN/DailyMail, XSum).
**Limitation**: same surface-overlap issues as BLEU; does not capture meaning.

### 3.4 BERTScore

BERTScore uses contextual embeddings from a pre-trained BERT model to compute token-level similarity between candidate and reference, capturing semantic meaning rather than surface overlap (Zhang et al., 2020).

$$\text{BERTScore F1} = \frac{2 \cdot P_\text{BERT} \cdot R_\text{BERT}}{P_\text{BERT} + R_\text{BERT}}$$

Where precision and recall are computed by matching each token in the candidate/reference to its most similar token in the other text using cosine similarity of contextual embeddings.

**Advantage over BLEU/ROUGE**: semantically equivalent phrases score highly even without exact word overlap.
**Use**: summarisation, translation, generation quality evaluation.

### 3.5 METEOR

METEOR (Metric for Evaluation of Translation with Explicit ORdering) improves on BLEU by incorporating stemming, synonymy, and paraphrase matching, with explicit recall weighting (Banerjee & Lavie, 2005).

**Use**: machine translation; often correlates better with human judgement than BLEU.

---

## 4. LLM Benchmark Suites

Benchmarks provide standardised tasks with known answers, enabling reproducible comparisons across models.

### 4.1 MMLU (Massive Multitask Language Understanding)

- **What**: 57-subject multiple-choice exam covering STEM, humanities, social sciences, and professional domains (law, medicine, finance)
- **Format**: 4-choice questions, accuracy measured
- **Why it matters**: tests breadth of world knowledge and reasoning, not just language fluency
- **Score range**: GPT-2 ≈ 26% (random = 25%), GPT-4 ≈ 86%+
- **Paper**: Hendrycks et al. (2021)
- **Limitation**: susceptible to contamination if training data includes MMLU questions

### 4.2 HellaSwag

- **What**: commonsense reasoning — choose the most plausible continuation of a paragraph
- **Format**: 4-choice sentence completion
- **Why it matters**: designed to be adversarially hard for models that exploit statistical shortcuts
- **Score range**: human ≈ 95%, early GPT-2 ≈ 40%, modern LLMs ≈ 85-95%
- **Paper**: Zellers et al. (2019)

### 4.3 TruthfulQA

- **What**: 817 questions designed to elicit false answers — tests whether models reproduce common human misconceptions
- **Format**: open-ended generation, evaluated for truthfulness and informativeness
- **Why it matters**: directly measures hallucination tendency and sycophancy
- **Paper**: Lin et al. (2022)
- **Key finding**: larger models are not automatically more truthful — they can be more confidently wrong

### 4.4 HumanEval

- **What**: 164 hand-crafted Python programming problems with unit tests
- **Format**: model generates code, unit tests determine pass/fail
- **Metric**: pass@k — probability that at least one of k samples passes all unit tests
- **Why it matters**: functional correctness is unambiguous — code either works or it doesn't
- **Score range**: GPT-4 ≈ 67-85% pass@1, Claude 3 Opus ≈ 84%
- **Paper**: Chen et al. (2021, OpenAI Codex)

### 4.5 BIG-Bench (Beyond the Imitation Game)

- **What**: 204 tasks across diverse skills — logic, maths, linguistics, world knowledge, creativity — designed to be hard for contemporary models
- **Why it matters**: crowdsourced from 450+ researchers, covers long-tail capabilities not in standard benchmarks
- **BIG-Bench Hard**: 23 tasks where models were below average human performance; now used to track frontier model progress
- **Paper**: Srivastava et al. (2023)

### 4.6 MATH

- **What**: 12,500 competition mathematics problems across 7 difficulty levels (algebra, calculus, number theory, etc.)
- **Format**: free-form answer, evaluated for exact match
- **Why it matters**: pure reasoning — hard to memorise, tests step-by-step mathematical thinking
- **Score range**: GPT-4 ≈ 42% (original), GPT-4 with tools ≈ 87%+
- **Paper**: Hendrycks et al. (2021)

### 4.7 GSM8K (Grade School Math)

- **What**: 8,500 linguistically diverse grade-school maths word problems
- **Why it matters**: simpler than MATH but evaluates multi-step arithmetic reasoning with natural language; widely used to test CoT prompting improvements
- **Paper**: Cobbe et al. (2021)

---

## 4a. The Benchmark Vocabulary: What Labs Actually Report

When OpenAI, Anthropic, and Google announce a new model, they publish a model card or technical report that includes benchmark scores. These scores serve a specific purpose: they allow the model to be positioned relative to competitors on a shared, reproducible scale. Understanding which benchmarks appear in these announcements — and what each actually tests — is essential for critically reading model releases.

Below is the core vocabulary. These benchmarks appear repeatedly across lab announcements.

### Knowledge and Reasoning

**MMLU (Massive Multitask Language Understanding)**
- 57-subject multiple choice exam: STEM, law, medicine, history, ethics, and more
- Tests breadth of world knowledge at undergraduate-to-professional level
- The most widely reported benchmark — appears in virtually every major model release
- Limitation: heavily saturated by 2024-2025; top models score 85-90%+, making differentiation hard

**GPQA Diamond**
- Graduate-level science questions (biology, chemistry, physics) written by PhD experts
- "Google-proof" — questions designed so that searching the web won't reliably give you the answer
- Harder than MMLU; scores remain meaningful at the frontier — humans with PhDs score ~65%, top models ~75-85%
- Anthropic and OpenAI both report this in recent Claude and GPT-4o/o-series releases

**Humanity's Last Exam (HLE)**
- A 2024-2025 benchmark of ~3,000 extremely difficult questions contributed by domain experts worldwide
- Designed to be unsolvable by current models at launch — intended to stay challenging as capabilities grow
- Scores are low even for frontier models (single digits to low tens of percent at time of release)
- Increasingly appearing in model announcements as MMLU becomes too easy

**ARC-Challenge (AI2 Reasoning Challenge)**
- Grade-school science questions requiring genuine reasoning, not just recall
- Filtered to include only questions that retrieval and word-frequency models fail on
- Commonly reported alongside MMLU as a knowledge + reasoning double-check

**WinoGrande**
- Large-scale commonsense reasoning via pronoun disambiguation (Winograd schema format)
- Tests whether models understand real-world context, not just language patterns

### Mathematics

**MATH**
- 12,500 competition-level problems across 7 difficulty levels (AMC/AIME/Olympiad style)
- Tests multi-step mathematical reasoning and symbolic manipulation
- Commonly reported for both base and reasoning models

**AIME (American Invitational Mathematics Examination)**
- Real competition problems from the AIME exam (high-school maths olympiad)
- Increasingly used for reasoning models (o1, o3, DeepSeek R1) where MATH has become saturated
- AIME 2024 and AIME 2025 are used as living benchmarks — harder to contaminate since problems are newly released each year
- OpenAI reports AIME scores for o-series reasoning models

**GSM8K (Grade School Math)**
- Multi-step arithmetic word problems at grade school level
- Widely reported but increasingly saturated — most frontier models score 95%+
- Still used as a baseline check for smaller/cheaper models

### Coding

**HumanEval**
- 164 Python programming problems with unit tests; metric is pass@1 (does the first attempt pass all tests?)
- The standard baseline for coding capability — reported by all three labs
- Limitation: relatively easy for frontier models; scores above 90% are common

**SWE-bench Verified**
- Real GitHub issues from popular Python repositories; model must write a patch that passes the repo's existing test suite
- Much harder and more realistic than HumanEval — tests end-to-end software engineering ability
- Scores remained below 50% for most models into 2024; Claude 3.7 Sonnet and similar models pushed this significantly
- Increasingly the benchmark of choice for agentic coding evaluation

**LiveCodeBench**
- Competitive programming problems collected continuously from Codeforces, LeetCode, and AtCoder after a cutoff date
- Designed to reduce contamination: problems post-date training data
- Used by labs to demonstrate reasoning and coding on genuinely unseen problems

### Long Context and Instruction Following

**NIAH (Needle in a Haystack)**
- Buries a specific fact ("the needle") inside a very long document and asks the model to retrieve it
- Tests whether models can attend to information anywhere in their context window, not just the beginning and end
- Anthropic reported near-perfect recall (>99%) for Claude 3 Opus on this test
- Increasingly standard as models compete on 128K–1M token context windows

**IFEval (Instruction Following Evaluation)**
- Tests whether models follow explicit formatting and structural instructions (e.g. "respond in exactly 3 bullet points", "use the word 'banana' exactly once")
- Measures instruction adherence, not just answer quality
- Appearing in newer model cards as instruction-following becomes a key product differentiator

### Multimodal (Vision + Language)

**MMMU (Massive Multidisciplinary Multimodal Understanding)**
- Like MMLU but with images — questions require understanding charts, diagrams, scientific figures
- Standard for vision-language model evaluation
- Reported by Google (Gemini) and OpenAI (GPT-4o) for multimodal benchmarks

**DocVQA / ChartQA**
- Document and chart visual question answering — tests ability to read and reason over structured visual information
- Practical relevance: real-world documents, PDFs, spreadsheets

### Professional and Academic Exams

Some labs — particularly OpenAI's GPT-4 technical report — reported performance on real-world professional exams:

| Exam | Domain | Why reported |
|---|---|---|
| Bar Exam | Law | Professional licensing standard |
| USMLE (Medical Licensing) | Medicine | High-stakes domain benchmark |
| SAT / GRE | Academic reasoning | General population reference point |
| AP Exams | Subject knowledge | Curriculum-aligned knowledge test |

These are not used for ongoing model comparisons (too coarse), but were powerful communication tools when GPT-4 first demonstrated professional-level performance.

### A Practical Reading Guide

When you see a model announcement, read the benchmark table with these questions in mind:

1. **Is the benchmark saturated?** If all frontier models score 90%+ on a benchmark, a 1-2% difference is noise, not signal. Look for benchmarks where there is still spread.

2. **Is this a reasoning model being tested on reasoning benchmarks only?** Reasoning models (o1, o3, DeepSeek R1) are specifically optimised for AIME and MATH — comparing them on these benchmarks against non-reasoning models is not apples-to-apples.

3. **Is the evaluation few-shot or zero-shot?** The number of examples provided in the prompt significantly affects scores. Labs sometimes choose the setup that favours their model.

4. **What is *not* reported?** Labs choose which benchmarks to publish. A model card that omits TruthfulQA or bias benchmarks may be avoiding unflattering results.

5. **Does strong benchmark performance predict strong performance on your task?** Usually not directly — which is exactly why Section 8 (building your own eval) matters.

---

## 5. Leaderboards and Aggregated Rankings

### 5.1 Open LLM Leaderboard (Hugging Face)

- Tracks open-source model performance across a standardised set of benchmarks (MMLU, HellaSwag, TruthfulQA, Winogrande, GSM8K, ARC)
- Models are evaluated using `lm-evaluation-harness` in a controlled environment
- URL: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **Caveat**: leaderboard gaming is a known problem — some models are fine-tuned specifically on benchmark data

### 5.2 LMSYS Chatbot Arena

- Human preference-based ranking — real users compare outputs from two anonymous models and vote for the better response
- Rankings computed using an Elo rating system (same as chess)
- Considered one of the most reliable rankings because it measures real-world human preference, not benchmark performance
- URL: https://arena.ai (formerly chat.lmsys.org — rebranded to Arena AI)
- **Paper**: Zheng et al. (2023) — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

### 5.3 HELM (Holistic Evaluation of Language Models)

- Stanford CRFM's comprehensive evaluation framework
- Evaluates models across 42 scenarios and 7 metrics: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency
- Distinguishes between *core* scenarios (standard tasks) and *targeted* scenarios (safety, fairness)
- URL: https://crfm.stanford.edu/helm/latest/
- **Paper**: Liang et al. (2022)

### 5.4 Artificial Analysis

Artificial Analysis (https://artificialanalysis.ai) is an independent platform that evaluates and compares ~410 models across four dimensions simultaneously — making it one of the most practically useful resources for real-world model selection.

**What makes it different from academic benchmarks:** it does not just measure *intelligence* — it measures the full cost-quality-speed trade-off that practitioners actually face when choosing a model for a production application.

Evaluation dimensions:
- **Intelligence Index**: an aggregated score across 10 evaluations covering reasoning, knowledge, maths, and coding (including their own AA-Omniscience benchmark for knowledge reliability and hallucination)
- **Speed**: output tokens per second — critical for user-facing applications
- **Latency / TTFT**: time to first token — determines perceived responsiveness
- **Cost**: price per million input/output tokens
- **Context window**: how much text the model can process at once
- **Openness**: whether weights are available, API-only, etc.

**GDPval-AA** is their agentic benchmark — evaluating models on real-world work tasks rather than academic questions.

**Why this matters for practitioners:** a model that ranks #1 on intelligence may be 10× more expensive and 5× slower than a model that ranks #3. Artificial Analysis makes this trade-off visible. For example, a model used in a high-throughput document processing pipeline has very different requirements than one used in an interactive chat assistant — and these rankings help quantify that.

Metrics are updated every 72 hours (8 measurements per day), so the data reflects current provider performance, not just model capabilities at release time.

### 5.5 METR (Model Evaluation and Threat Research)

METR (https://metr.org) is a research nonprofit that evaluates frontier AI models for *autonomous capabilities and safety risks* — a fundamentally different question from "how well does this model answer exam questions?"

**Mission**: scientifically measure whether and when AI systems might pose catastrophic risks, with a focus on broad autonomous capabilities and the ability of AI to accelerate its own development.

**Key evaluations:**

- **HCAST (Human-Calibrated Autonomy Software Tasks)**: measures whether AI agents can autonomously complete diverse software tasks end-to-end — not just answer questions about code, but actually execute multi-step engineering work
- **RE-Bench**: evaluates AI R&D capabilities on machine learning research engineering tasks, benchmarked against human expert performance
- **Time Horizon metric**: quantifies the length (in time) of software tasks AI agents can independently complete. This metric has doubled roughly every 7 months over the past 6 years — a finding with significant implications for understanding AI capability growth

**Key findings relevant to this course:**
- Models show increasingly clear examples of *reward hacking* — exploiting scoring bugs rather than solving problems as intended (observed in GPT-5, o3). This is a real-world instance of Goodhart's Law in action.
- METR found that AI coding tools made *experienced developers 19% slower* on real-world tasks despite strong benchmark performance — a stark illustration of the gap between benchmark scores and production utility.
- METR evaluates models on behalf of AI labs before public release (pre-deployment safety evaluations), making their work part of the governance infrastructure around frontier models.

**Why this matters:** METR-style evaluations represent a shift from "what can this model answer?" to "what can this model *do autonomously*?" As AI systems are deployed in agentic settings (Session 9), this kind of capability evaluation becomes critical for safety and deployment decisions.

---

## 5a. Industry Evaluation Resources in the LLM Selection Lifecycle

In practice, choosing an LLM for a real project involves a pipeline of evaluation — not a single benchmark. The resources above play different roles at different stages:

```
Stage 1: Discovery → Stage 2: Shortlisting → Stage 3: Task-Specific Validation → Stage 4: Production Monitoring
```

**Stage 1 — Discovery (What models exist and are broadly capable?)**
Use: Open LLM Leaderboard, LMSYS Chatbot Arena, Artificial Analysis Intelligence Index
These tell you which models are competitive on general tasks and worth considering.

**Stage 2 — Shortlisting (Which models fit my constraints?)**
Use: Artificial Analysis (speed, cost, latency, context window)
Here you apply your real-world constraints: budget, latency requirements, whether open weights are needed, context length. A model that fails your constraints is eliminated regardless of intelligence score.

**Stage 3 — Task-Specific Validation (Does this model work for *my* task?)**
Use: Your own eval pipeline (Section 8), LLM-as-judge, task-specific benchmarks (HumanEval for code, TruthfulQA for factual QA, etc.)
**This is the hardest and most important stage.** Industry leaderboards measure general capability — they cannot tell you whether the model performs well on your specific domain, your data distribution, or your user population. A model that ranks #5 globally may outperform #1 on your specific task. This step cannot be skipped.

**Stage 4 — Production Monitoring (Is the model still performing after deployment?)**
Use: LangSmith, custom logging, continuous eval on sampled production traffic
Models are updated, APIs change, and user behaviour shifts. Production evaluation is ongoing, not a one-time gate.

**The key insight:** industry leaderboards and platforms like Artificial Analysis and METR give you an *evidence-based shortlist* — they replace uninformed guessing with structured comparison. But they are proxies for your actual use case. The further your task is from standard benchmarks (e.g., niche domain, specific language, specialised task format), the less predictive leaderboard rankings are, and the more critical your own evaluation becomes.

---

## 6. Evaluation Tools and Frameworks

### 6.1 lm-evaluation-harness (EleutherAI)

The de facto standard tool for evaluating open-source language models on standardised benchmarks.

```bash
pip install lm-eval
lm_eval --model hf --model_args pretrained=gpt2 --tasks mmlu --num_fewshot 5
```

- Supports 200+ tasks out of the box
- Used by the Hugging Face Open LLM Leaderboard
- Extensible — add custom tasks in a few lines
- GitHub: https://github.com/EleutherAI/lm-evaluation-harness

### 6.2 OpenAI Evals

OpenAI provides two related but distinct evaluation tools:

**6.2a — Open-source eval framework (GitHub)**

The original community framework for running structured evals against OpenAI models. Useful for browsing pre-built evaluation sets.

- Community eval registry — browse and reuse existing evaluation sets
- GitHub: https://github.com/openai/evals

**6.2b — OpenAI Evals API (platform feature)**

OpenAI's newer API-based evaluation platform, available to API users. Rather than running evals locally, you define the evaluation structure via the API and OpenAI handles the execution. This is particularly relevant as a managed alternative to building your own eval loop (see Section 8).

The core model: an eval = **data source config** (what you're testing on) + **testing criteria** (how you grade the output).

**Grader types** — the building blocks of OpenAI's evaluation system:

| Grader | What it does | When to use |
|---|---|---|
| `string_check` | Exact or partial string match (`eq`, `ne`, `like`, `ilike`) | Classification tasks with fixed output labels |
| `text_similarity` | Scores overlap using BLEU, cosine, fuzzy match, etc. | Summarisation, open-ended generation with a reference |
| `label_model` | Another LLM classifies the output from a set of labels | Sentiment, intent, category — where output is open text |
| `score_model` | Another LLM assigns a numeric score to the output | Quality, helpfulness, coherence scoring |
| `python` | Custom Python `grade(sample, item) -> float` function | Any custom logic that doesn't fit the above |

The simplest grader is `string_check` — it directly mirrors the `exact_match` function in the custom eval loop in Section 8, but configured declaratively:

```json
{
    "type": "string_check",
    "name": "Match output to correct label",
    "input": "{{ sample.output_text }}",
    "operation": "eq",
    "reference": "{{ item.correct_label }}"
}
```

The `label_model` grader is the platform-managed version of LLM-as-Judge (Section 7) — you provide a prompt, a list of valid labels, and the model classifies each output:

```json
{
    "type": "label_model",
    "model": "gpt-4o-mini",
    "name": "Sentiment grader",
    "input": [
        {"role": "system", "content": "Classify the sentiment as positive, neutral, or negative."},
        {"role": "user", "content": "Statement: {{ item.input }}"}
    ],
    "labels": ["positive", "neutral", "negative"],
    "passing_labels": ["positive"]
}
```

**Key insight for practitioners:** the OpenAI Evals API formalises the same pattern we build by hand in Section 8. Understanding the DIY version first (test cases → run model → grade → aggregate) makes the platform abstraction much easier to reason about — and helps you know when to reach for a managed tool vs. when to roll your own.

- Docs: https://platform.openai.com/docs/guides/evals

### 6.3 LangSmith

LangChain's tracing and evaluation platform — records every LLM call, input, output, and latency, then runs evaluators over logged traces.

- Supports both automatic metrics and LLM-as-judge evaluators
- Integrates tightly with LangChain applications
- URL: https://smith.langchain.com

### 6.4 Promptfoo

Open-source CLI tool for prompt testing and model comparison — define test cases in YAML, run against multiple models, get a visual comparison report.

```bash
npx promptfoo eval
```

- Good for AT2-style projects: compare models/prompts systematically
- GitHub: https://github.com/promptfoo/promptfoo

---

## 7. LLM-as-Judge

Instead of using human annotators or fixed metrics, use a strong LLM (e.g. GPT-4) to evaluate the output of another LLM. This scales human-quality evaluation to thousands of examples.

### 7.1 MT-Bench

- 80 multi-turn questions across writing, roleplay, extraction, reasoning, maths, coding, knowledge
- GPT-4 scores each response 1–10 with reasoning
- Enables automated comparison of instruction-following models at scale
- **Paper**: Zheng et al. (2023)

### 7.2 Alpaca Eval

- Compares model responses against a reference model (e.g. text-davinci-003)
- LLM judge decides which response is preferred
- Win rate against reference = the metric
- GitHub: https://github.com/tatsu-lab/alpaca_eval

### 7.3 LLM-as-Judge Implementation Pattern

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

def llm_judge(question: str, response: str, criteria: str) -> dict:
    prompt = f"""You are an expert evaluator. Score the following response.

Question: {question}
Response: {response}

Evaluation criteria: {criteria}

Provide:
1. A score from 1-10
2. A brief justification (2-3 sentences)

Respond in JSON: {{"score": <int>, "justification": "<str>"}}"""

    result = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return result.choices[0].message.content
```

### 7.4 Limitations of LLM-as-Judge

- **Positional bias**: judges prefer the first response when given two options
- **Verbosity bias**: judges favour longer, more detailed responses regardless of quality
- **Self-enhancement bias**: a model tends to prefer its own outputs
- **Mitigations**: swap position of responses and average; use multiple judge models; calibrate against human judgements

---

## 8. Building a Custom Eval Pipeline

For real projects (like AT2), you often need evaluation tailored to your specific task and success criteria.

### 8.1 Define Your Evaluation Dimensions

Before writing any code, decide what you are measuring. Common dimensions:

| Dimension | Description | Example metric |
|---|---|---|
| Correctness | Is the answer factually right? | Exact match, F1 |
| Fluency | Is the output well-written? | Perplexity, human rating |
| Faithfulness | Is the output grounded in the provided context? | NLI-based, LLM judge |
| Relevance | Does the output address the question? | Cosine sim, LLM judge |
| Safety | Does the output avoid harmful content? | Classifier, red-team |
| Latency | How fast is the response? | Wall-clock time |
| Cost | How many tokens does it consume? | Token count × price |

### 8.2 Build a Simple Eval Loop

```python
import json
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# 1. Define test cases: input + expected output
test_cases = [
    {
        "input": "What is the capital of Australia?",
        "expected": "Canberra",
    },
    {
        "input": "Summarise: 'The cat sat on the mat. It was comfortable.'",
        "expected": "A cat sat comfortably on a mat.",
    },
]

# 2. Run model on each test case
def run_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 3. Score each result
def exact_match(prediction: str, expected: str) -> bool:
    return expected.lower() in prediction.lower()

# 4. Aggregate results
results = []
for case in test_cases:
    prediction = run_model(case["input"])
    score = exact_match(prediction, case["expected"])
    results.append({
        "input": case["input"],
        "expected": case["expected"],
        "prediction": prediction,
        "correct": score
    })

accuracy = sum(r["correct"] for r in results) / len(results)
print(f"Accuracy: {accuracy:.1%}")
for r in results:
    print(f"  [{'✓' if r['correct'] else '✗'}] {r['input'][:50]}...")
    print(f"      Expected: {r['expected']}")
    print(f"      Got:      {r['prediction'][:80]}")
```

### 8.3 Prompt-based Scoring (LLM-as-Judge, Custom Criteria)

```python
def score_with_llm(question: str, response: str, rubric: str) -> int:
    """
    Use an LLM to score a response against a custom rubric.
    Returns an integer score 1-5.
    """
    prompt = f"""Score the response below using the rubric provided.
Return ONLY a JSON object with keys "score" (int 1-5) and "reason" (str).

Question: {question}
Response: {response}
Rubric: {rubric}"""

    result = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0  # reduces sampling variation; exact repeatability is not guaranteed
    )
    data = json.loads(result.choices[0].message.content)
    return data["score"], data["reason"]

# Example rubric for a summarisation task
rubric = """
5 - Captures all key points, concise, no hallucinations
4 - Captures most key points, minor omissions
3 - Captures some key points, some important gaps
2 - Mostly misses the key points or contains errors
1 - Completely wrong or incoherent
"""
```

### 8.4 Version and Track Your Evals

Treat evaluation like software:
- Version control your test cases alongside your prompts
- Track results over time — a table of `(date, model, prompt_version, score)` is invaluable
- Regression test: when you update a prompt, re-run all test cases to catch regressions
- Tools: LangSmith, Weights & Biases, or even a simple CSV

---

## 9. Practical Guidance for Practitioners

**Start with human evaluation on a small set.** Before writing any automated eval, manually review 20-30 model outputs. This builds intuition about what "good" looks like for your task — intuition you need to write useful automatic metrics.

**Pick metrics that match your task:**
- Translation / summarisation → ROUGE, BERTScore
- Code generation → pass@k with unit tests
- Open-ended QA → LLM-as-judge
- Classification → accuracy, F1
- Factual recall → exact match, F1

**Don't over-rely on a single number.** A model that scores 85% on MMLU may hallucinate on your domain. Benchmark scores are proxies — always validate on your actual use case.

**Evaluate failure modes, not just average performance.** What does the model get wrong? Is it consistent? Does it fail gracefully? A model that scores 80% but catastrophically fails on 5% of inputs may be worse than a 75% model that fails softly.

**Red-teaming is evaluation too.** Deliberately try to make the model fail — give it ambiguous inputs, contradictory context, adversarial prompts. This is especially important for safety-critical applications.

---

## References

- Papineni, K. et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." *ACL.*
- Lin, C.-Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop.*
- Zhang, T. et al. (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR.*
- Hendrycks, D. et al. (2021). "Measuring Massive Multitask Language Understanding." *ICLR.* (MMLU)
- Hendrycks, D. et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset." *NeurIPS.* (MATH)
- Zellers, R. et al. (2019). "HellaSwag: Can a Machine Really Finish Your Sentence?" *ACL.*
- Lin, S. et al. (2022). "TruthfulQA: Measuring How Models Mimic Human Falsehoods." *ACL.*
- Chen, M. et al. (2021). "Evaluating Large Language Models Trained on Code." *arXiv.* (HumanEval)
- Srivastava, A. et al. (2023). "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models." *TMLR.* (BIG-Bench)
- Cobbe, K. et al. (2021). "Training Verifiers to Solve Math Word Problems." *arXiv.* (GSM8K)
- Liang, P. et al. (2022). "Holistic Evaluation of Language Models." *arXiv.* (HELM)
- Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS.* (MT-Bench, Chatbot Arena)
- Guo, B. et al. (2023). "Evaluating Large Language Models: A Comprehensive Survey." *arXiv.*
- METR. (2024). "HCAST: Human-Calibrated Autonomy Software Tasks." https://metr.org/research/
- Artificial Analysis. (2025). "AI Model Intelligence & Performance Leaderboard." https://artificialanalysis.ai/leaderboards/models
