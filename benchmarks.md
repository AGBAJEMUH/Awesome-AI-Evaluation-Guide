# AI Evaluation Benchmarks & Datasets

Comprehensive reference for evaluating LLMs, RAG systems, agents, and domain-specific capabilities.

## Table of Contents

- [General Language Understanding](#general-language-understanding)
- [Domain-Specific Benchmarks](#domain-specific-benchmarks)
- [Agent & Tool Use](#agent--tool-use)
- [Retrieval & RAG](#retrieval--rag)
- [Safety & Ethics](#safety--ethics)
- [Multilingual & Low-Resource](#multilingual--low-resource)
- [Leaderboards](#leaderboards)

---

## General Language Understanding

### MMLU (Massive Multitask Language Understanding)
**Repository**: https://github.com/hendrycks/test

**What it measures**: Knowledge across 57 academic subjects (STEM, humanities, social sciences)

**Format**: 4-choice multiple-choice questions

**Size**: 15,908 questions

**Difficulty**: Undergraduate to professional level

**Example subjects**:
- Abstract Algebra
- US History
- Medical Genetics
- Moral Scenarios
- Professional Law

**Why it matters**: Industry-standard for measuring broad knowledge

**Typical scores**:
- GPT-4: ~86%
- GPT-3.5: ~70%
- Human expert: ~90%

**Use when**: Benchmarking general-purpose models

---

### MMLU-Pro
**Repository**: https://github.com/TIGER-AI-Lab/MMLU-Pro

**What it measures**: Enhanced MMLU with 10-choice questions and reasoning focus

**Key differences from MMLU**:
- 10 choices instead of 4 (harder)
- More reasoning-intensive questions
- Reduced data contamination
- Expert-validated

**Size**: 12,000+ questions

**Why it matters**: Addresses MMLU's issues with memorization and guessing

**Use when**: You need harder, less contaminated evaluation

---

### BIG-bench (Beyond the Imitation Game)
**Repository**: https://github.com/google/BIG-bench

**What it measures**: Diverse reasoning tasks across 200+ tasks

**Categories**:
- Reasoning (logic, mathematics, common sense)
- Linguistics (syntax, semantics, translation)
- Social reasoning (theory of mind, ethics)
- World knowledge
- Creativity

**Size**: 200+ tasks, highly varied

**Why it matters**: Collaborative benchmark from 400+ authors, covers edge cases

**Use when**: Comprehensive capability assessment beyond standard benchmarks

---

### HELM (Holistic Evaluation of Language Models)
**Website**: https://crfm.stanford.edu/helm/latest/

**What it measures**: Multi-dimensional evaluation across 7 metrics and 16 scenarios

**Metrics**:
1. Accuracy
2. Calibration
3. Robustness
4. Fairness
5. Bias
6. Toxicity
7. Efficiency

**Scenarios**: Question answering, summarization, toxicity detection, disinformation

**Why it matters**: First truly holistic benchmark considering multiple evaluation dimensions

**Use when**: You need comprehensive model assessment beyond accuracy

---

## Domain-Specific Benchmarks

### Code Generation

#### HumanEval
**Repository**: https://github.com/openai/human-eval

**What it measures**: Code synthesis from docstrings

**Format**: Python function completion with unit tests

**Size**: 164 programming problems

**Evaluation metric**: Pass@k (probability ≥1 of k samples passes tests)

**Typical scores**:
- GPT-4: ~67% (Pass@1)
- GPT-3.5: ~48% (Pass@1)

**Why it matters**: Standard for code generation evaluation

**Use when**: Evaluating code generation models

---

#### MBPP (Mostly Basic Programming Problems)
**Repository**: https://github.com/google-research/google-research/tree/master/mbpp

**What it measures**: Entry-level programming tasks

**Format**: Natural language description → Python code

**Size**: 974 programming problems

**Difficulty**: Easier than HumanEval (suitable for beginners)

**Why it matters**: Tests basic programming competence

**Use when**: Evaluating code assistants for beginners

---

### Mathematics

#### MATH
**Repository**: https://github.com/hendrycks/math

**What it measures**: Competition-level mathematics

**Topics**: Algebra, counting, geometry, number theory, precalculus, probability

**Size**: 12,500 problems

**Difficulty**: High school math competition level

**Format**: Free-form answer with step-by-step solutions

**Typical scores**:
- GPT-4: ~52%
- GPT-3.5: ~23%
- Human average: ~40%
- Math competition winners: ~90%

**Why it matters**: Tests advanced mathematical reasoning

**Use when**: Evaluating quantitative reasoning capabilities

---

#### GSM8K (Grade School Math 8K)
**Repository**: https://github.com/openai/grade-school-math

**What it measures**: Grade school math word problems

**Size**: 8,500 problems

**Difficulty**: Elementary/middle school level

**Format**: Word problem → numerical answer

**Why it matters**: Tests basic arithmetic and reasoning

**Use when**: Evaluating mathematical word problem solving

---

### Finance

#### FinEval
**Repository**: https://github.com/SUFE-AIFLM-Lab/FinEval

**What it measures**: Financial knowledge and reasoning (Chinese focus)

**Topics**: Accounting, regulation, markets, derivatives, corporate finance

**Size**: 4,661 questions across 34 subjects

**Language**: Chinese

**Why it matters**: Domain-specific financial evaluation

**Use when**: Evaluating models for financial applications

---

### Legal

#### LAiW (Legal AI Benchmark)
**Repository**: https://github.com/Dai-shen/LAiW

**What it measures**: Legal reasoning in Chinese law

**Components**:
- Legal information retrieval
- Foundation legal inference
- Complex case applications

**Size**: Multiple tasks with thousands of examples

**Language**: Chinese

**Why it matters**: Tests legal knowledge and reasoning

**Use when**: Evaluating legal AI systems

---

### Medical

#### MedQA
**Paper**: "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams"

**What it measures**: Medical exam question answering

**Source**: US Medical Licensing Examination (USMLE) style questions

**Size**: 61,000+ questions

**Languages**: English, Chinese

**Why it matters**: Clinical reasoning and medical knowledge

**Use when**: Evaluating medical AI assistants

---

## Agent & Tool Use

### AgentBench
**Repository**: https://github.com/THUDM/AgentBench

**What it measures**: LLMs as agents across simulated domains

**Environments**:
- Operating systems (OS interaction)
- Databases (SQL generation and execution)
- Knowledge graphs (query and reasoning)
- Card games (strategy and planning)
- Lateral thinking puzzles

**Size**: 8 distinct environments

**Why it matters**: Multi-turn interaction and tool use

**Use when**: Evaluating agentic capabilities

---

### GAIA (General AI Assistants)
**Dataset**: https://huggingface.co/datasets/gaia-benchmark/GAIA

**What it measures**: Tool use with grounded reasoning and web access

**Format**: Real-world questions requiring:
- Multi-step reasoning
- Tool/API use
- Web search
- File processing

**Difficulty levels**: 1-3 (increasing complexity)

**Why it matters**: Real-world assistant capabilities

**Use when**: Evaluating production assistants

---

### ToolBench
**Repository**: https://github.com/OpenBMB/ToolBench

**What it measures**: Tool learning and API use

**Features**:
- 16,000+ real-world APIs
- Instruction-solution pairs
- Automatic evaluator

**Why it matters**: Systematic tool use evaluation

**Use when**: Evaluating tool-using agents

---

## Retrieval & RAG

### BEIR (Benchmarking IR)
**Repository**: https://github.com/beir-cellar/beir

**What it measures**: Information retrieval across diverse domains

**Datasets**: 18 retrieval datasets including:
- MS MARCO
- Natural Questions
- HotpotQA
- FiQA (financial)
- SciFact (scientific)

**Metrics**: nDCG@10, MAP, Recall@100

**Why it matters**: Standard for retrieval evaluation

**Use when**: Evaluating retrieval systems

---

### MTEB (Massive Text Embedding Benchmark)
**Repository**: https://github.com/embeddings-benchmark/mteb

**What it measures**: Embedding quality across 8 tasks

**Tasks**:
1. Retrieval
2. Reranking
3. Semantic Textual Similarity
4. Classification
5. Clustering
6. Pair Classification
7. STS
8. Summarization

**Size**: 58 datasets across 112 languages

**Why it matters**: Comprehensive embedding evaluation

**Use when**: Selecting embedding models for RAG

---

### RAGTruth
**Repository**: https://github.com/zhengzangw/RAGTruth

**What it measures**: Hallucination and faithfulness in RAG

**Features**:
- Human-annotated hallucinations
- Fine-grained hallucination types
- Faithfulness metrics

**Why it matters**: Focuses on RAG-specific issues

**Use when**: Evaluating RAG faithfulness

---

## Safety & Ethics

### TruthfulQA
**Repository**: https://github.com/sylinrl/TruthfulQA

**What it measures**: Factuality and hallucination propensity

**Format**: Adversarially-written questions where humans would answer incorrectly due to false beliefs

**Size**: 817 questions across 38 categories

**Why it matters**: Tests resistance to common misconceptions

**Typical scores**:
- GPT-4: ~59%
- GPT-3.5: ~47%
- Human: ~100% (by design)

**Use when**: Measuring truthfulness and hallucination

---

### BBQ (Bias Benchmark for QA)
**Repository**: https://github.com/nyu-mll/BBQ

**What it measures**: Social bias in question answering

**Bias categories**:
- Age
- Disability status
- Gender identity
- Nationality
- Physical appearance
- Race/ethnicity
- Religion
- Socioeconomic status

**Format**: Context + question with ambiguous and disambiguated versions

**Why it matters**: Systematic bias measurement

**Use when**: Evaluating fairness and bias

---

### ToxiGen
**Repository**: https://github.com/microsoft/ToxiGen

**What it measures**: Toxic language generation and detection

**Size**: 274k toxic and benign statements

**Categories**: 13 minority groups

**Why it matters**: Toxicity in generation

**Use when**: Safety testing for production systems

---

### AdvBench (Adversarial Benchmark)
**Repository**: https://github.com/llm-attacks/llm-attacks

**What it measures**: Jailbreak and misuse resistance

**Format**: Adversarial prompts designed to elicit harmful responses

**Size**: Harmful behaviors and strings

**Why it matters**: Red-teaming and safety evaluation

**Use when**: Testing guardrails and safety measures

---

## Multilingual & Low-Resource

### XTREME (Cross-lingual TRansfer Evaluation)
**Repository**: https://github.com/google-research/xtreme

**What it measures**: Cross-lingual transfer across 40 languages

**Tasks**:
- Sentence classification
- Structured prediction
- Question answering
- Retrieval

**Why it matters**: Multilingual and low-resource evaluation

**Use when**: Evaluating multilingual models

---

### FLORES-200
**Repository**: https://github.com/facebookresearch/flores

**What it measures**: Machine translation for 200 languages

**Size**: 3,001 sentences translated to 200 languages

**Why it matters**: Low-resource language translation

**Use when**: Evaluating translation capabilities

---

## Leaderboards

### Open LLM Leaderboard (Hugging Face)
**URL**: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

**Benchmarks**:
- IFEval (instruction following)
- MMLU-Pro
- GPQA (graduate-level)
- MuSR (multi-step reasoning)
- MATH

**Why it matters**: Community-driven, transparent evaluation

---

### CompassRank (OpenCompass)
**URL**: https://rank.opencompass.org.cn/

**Benchmarks**: Multi-domain Chinese and English benchmarks

**Why it matters**: China-focused evaluation

---

### LMSYS Chatbot Arena
**URL**: https://chat.lmsys.org/

**What it measures**: Human preference via pairwise comparisons

**Format**: Users chat with anonymous models and vote for better response

**Why it matters**: Human-aligned evaluation at scale

---

## Benchmark Selection Guide

### Choose MMLU when:
- You need industry-standard knowledge benchmark
- You want broad subject coverage
- You're comparing to published results

### Choose HumanEval when:
- You're evaluating code generation
- You need reproducible code evaluation
- You want standard coding benchmark

### Choose TruthfulQA when:
- You're concerned about hallucinations
- You need to test factuality
- You're building production systems

### Choose BEIR when:
- You're evaluating retrieval systems
- You need cross-domain retrieval assessment
- You're building RAG systems

### Choose AgentBench when:
- You're evaluating agentic capabilities
- You need multi-turn task evaluation
- You're building autonomous agents

---

## Custom Benchmark Creation

For domain-specific applications, consider creating custom benchmarks:

1. **Collect real user queries** from production logs
2. **Create golden answers** through expert annotation
3. **Define evaluation criteria** specific to your domain
4. **Implement automated evaluation** with G-Eval or similar
5. **Track metrics over time** to measure improvement

**Example**: Medical RAG system
- Collect 500 real patient questions
- Have medical professionals create reference answers
- Use medical faithfulness G-Eval metric (threshold 0.9)
- Measure hallucination rate, accuracy, safety

---

## Further Reading

- [Benchmark Papers Collection](https://github.com/dair-ai/ML-Papers-of-the-Week)
- [Evaluation Best Practices](./docs/best-practices.md)
- [Creating Custom Evals](./docs/custom-evals.md)
