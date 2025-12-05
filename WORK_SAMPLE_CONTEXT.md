# Agentic AI Evaluation Framework (Fellowship Excerpt)

> **Contextual Note:** This document is a specific excerpt from the [Awesome AI Evaluation Guide](README.md) curated for the Berkman Klein Center Fellowship application. It highlights the project's focus on **domain-specific rigor**, **safety-critical evaluation**, and **interpretive auditing** for Agentic AI systems.

---

## 1. Key Research Insights
The following table distills recent findings into actionable engineering principles, illustrating the shift from pure accuracy to reliability and bias detection.

| Concept | Finding | Application |
|---------|---------|-------------|
| **Consistency vs Accuracy** | Models can have high accuracy but low consistency (SCORE Framework, 2025). | Evaluate reliability alongside correctness. |
| **Confidence Scoring** | Ensemble methods (majority voting) correlate better with accuracy than raw logprobs. | Use consensus for reliability. |
| **Bias Detection** | Bayes Factors prove superior to p-values for detecting subtle demographic shifts. | Use Bayesian methods for statistical rigor. |

---

## 2. Metric Selection: Domain-Specific Thresholds
A core contribution of this framework is establishing evaluation strictness based on the cost of failure in specific domains.

| Domain | Metric Type | Threshold | Rationale |
|--------|------------|-----------|-----------|
| **Medical** | Faithfulness | **> 0.9** | Patient safety critical; zero-tolerance for hallucinations. |
| **Legal** | Factual Accuracy | **> 0.95** | Regulatory compliance and liability. |
| **Financial** | Numerical Precision | **> 0.98** | Direct monetary implications. |
| **Education** | Answer Correctness | **> 0.85** | Pedagogical alignment. |

---

## 3. Implementation Example: Medical RAG Evaluation

The following code snippet demonstrates the **"LLM-as-a-Judge"** pattern applied to a high-stakes medical context. 

Crucially, this implementation prioritizes **Faithfulness** (logical adherence to retrieved context) over open-ended Correctness. This acts as a scalable filtering layer for **Human-in-the-Loop (HITL)** governance.

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Define a strict evaluation metric for clinical contexts
# This acts as a Safety Gate before Human Review
medical_faithfulness = GEval(
    name="Medical Faithfulness",
    evaluation_steps=[
        "Extract medical claims from the model output",
        "Verify each claim strictly against the provided clinical guidelines (RETRIEVAL_CONTEXT)",
        "Identify contradictions or unsupported claims",
        "HEAVILY PENALIZE hallucinations that are not found in the source text",
        "Emphasize clinical safety: if information is missing in context, the model must refuse to answer"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT # Validates against Ground Truth, not model knowledge
    ],
    threshold=0.9  # STRICT GOVERNANCE: Scores below 0.9 trigger mandatory HITL review
)

# This metric creates a traceable audit log for compliance review.