# Contributing to Awesome AI Evaluation Guide

Thank you for your interest in contributing! This guide aims to be a comprehensive, community-driven resource for AI evaluation.

## How to Contribute

### Ways to Contribute

1. **Add new evaluation methods** with working code examples
2. **Improve documentation** for existing metrics
3. **Report issues** or inaccuracies
4. **Share production case studies** and real-world applications
5. **Translate content** (especially Portuguese and Spanish for Latin American accessibility)
6. **Add domain-specific examples** (medical, legal, financial, etc.)

### Contribution Guidelines

#### Code Examples

All code examples should follow these standards:

1. **Complete and Runnable**: Examples must work out-of-the-box with dependencies in `requirements.txt`

2. **Well-Documented**: Include:
   - Module-level docstring with references to source papers/methods
   - Function docstrings with Args, Returns, and Examples
   - Inline comments for complex logic
   - Example usage in `if __name__ == "__main__"` block

3. **Type Hints**: Use type annotations for function signatures

4. **Error Handling**: Include try-except blocks for API calls and external dependencies

5. **Example Format**:
```python
"""
Metric Name

Brief description of what it measures and when to use it.

References:
- Author et al. (Year) "Paper Title"
- Documentation link
"""

from typing import List, Dict
import numpy as np


def calculate_metric(
    reference: str,
    candidate: str,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate metric with detailed explanation.

    Args:
        reference: Reference text (ground truth)
        candidate: Generated text (model output)
        threshold: Decision threshold (default: 0.5)

    Returns:
        results: Dictionary with scores and metadata

    Example:
        >>> ref = "The cat sat on the mat"
        >>> cand = "The cat is on the mat"
        >>> score = calculate_metric(ref, cand)
        >>> print(f"Score: {score['value']:.4f}")
    """
    # Implementation here
    pass


if __name__ == "__main__":
    # Example usage
    print("Example 1: Basic Usage")
    # ...
```

#### Documentation

1. **Markdown Format**: Use GitHub-flavored markdown
2. **Clear Structure**: Use headers, bullet points, code blocks
3. **Practical Focus**: Emphasize implementation and real-world usage
4. **Citations**: Include references to original papers and methods
5. **Examples**: Provide concrete use cases and code snippets

#### Domain-Specific Contributions

We especially welcome contributions for:

- **Global South contexts**: Evaluation in low-resource settings, multilingual scenarios
- **Critical domains**: Medical, legal, financial with appropriate safety considerations
- **Multi-agent systems**: Coordination, emergent behaviors, distributed explainability
- **Accessibility**: Making evaluation approachable for non-experts

### Submission Process

1. **Fork the repository**

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/add-new-metric
   ```

3. **Make your changes**:
   - Add code to appropriate `examples/` subdirectory
   - Add documentation to `docs/` subdirectory
   - Update README.md if adding new top-level section
   - Update requirements.txt if adding dependencies

4. **Test your code**:
   ```bash
   # Ensure all examples run
   python examples/your_new_metric.py

   # Run tests if applicable
   pytest tests/
   ```

5. **Commit with descriptive messages**:
   ```bash
   git commit -m "Add SelfCheckGPT hallucination detection example"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/add-new-metric
   ```

7. **Open a Pull Request**:
   - Provide clear description of changes
   - Reference any related issues
   - Include example output if applicable

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All examples run without errors
- [ ] Documentation is clear and complete
- [ ] References are properly cited
- [ ] No plagiarized content (original implementations, proper attribution)
- [ ] Type hints included
- [ ] Example usage provided
- [ ] Updated README.md if necessary
- [ ] Updated requirements.txt if new dependencies added

### Code Review Process

1. Maintainer will review PR within 1 week
2. Address any feedback or requested changes
3. Once approved, PR will be merged
4. You'll be added to contributors list

### Style Guide

**Python**:
- Follow PEP 8
- Use meaningful variable names
- Maximum line length: 100 characters
- Use f-strings for formatting
- Prefer list comprehensions for simple transformations

**Markdown**:
- Use ATX-style headers (`#` not underlines)
- One sentence per line for easier diffs
- Code blocks with language specifiers
- Links in reference style for long documents

### Reporting Issues

When reporting bugs or suggesting improvements:

1. **Use issue templates** if available
2. **Be specific**: Include error messages, OS, Python version
3. **Provide context**: What were you trying to do?
4. **Minimal reproducible example** when possible
5. **Check existing issues** to avoid duplicates

### Questions and Discussions

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue
- **Feature requests**: Open an Issue with [Feature Request] tag
- **Urgent security issues**: Email directly (see README for contact)

## Code of Conduct

### Our Standards

- **Respectful communication**: Be kind and professional
- **Constructive feedback**: Focus on improving the work
- **Inclusive language**: Welcome contributors of all backgrounds
- **Academic integrity**: Properly cite sources, no plagiarism
- **Focus on learning**: Help others understand evaluation methods

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Plagiarism or false attribution
- Spam or off-topic discussions
- Publishing others' private information

### Enforcement

Violations will result in:
1. Warning
2. Temporary ban
3. Permanent ban (for severe or repeated violations)

Report violations by opening a private issue or emailing maintainer.

## Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes for significant contributions
- GitHub contributors graph

## Development Setup

### Local Development

1. **Clone repository**:
   ```bash
   git clone https://github.com/hparreao/Awesome-AI-Evaluation-Guide.git
   cd Awesome-AI-Evaluation-Guide
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys** (for examples requiring APIs):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run examples**:
   ```bash
   python examples/traditional_metrics/perplexity.py
   ```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_traditional_metrics.py

# Run with coverage
pytest --cov=examples tests/
```

## Priority Areas

We're especially looking for contributions in:

1. **Multi-agent evaluation methods**
2. **Domain-specific RAG evaluation** (medical, legal, scientific)
3. **Low-resource / multilingual evaluation**
4. **Production deployment guides**
5. **Bias and fairness detection methods**
6. **Explainability integration**

## Questions?

Don't hesitate to:
- Open a GitHub Discussion
- Comment on relevant issues
- Reach out to maintainers

We're here to help you contribute successfully!

---

**Thank you for helping make AI evaluation more accessible and robust!**
