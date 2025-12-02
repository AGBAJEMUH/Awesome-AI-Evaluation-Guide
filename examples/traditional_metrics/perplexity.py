"""
Perplexity Calculation for Language Model Evaluation

Perplexity measures model uncertainty in predicting the next token.
Lower perplexity indicates better performance.

Formula: Perplexity = exp(-(1/N) * Σ log(P(w_i | context)))

"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import Union, List


def calculate_perplexity(
    text: str,
    model_name: str = "gpt2",
    device: str = "cpu"
) -> float:
    """
    Calculate perplexity of text using a pre-trained language model.

    Args:
        text: Input text to evaluate
        model_name: HuggingFace model identifier (default: "gpt2")
        device: Device to run model on ("cpu" or "cuda")

    Returns:
        perplexity_score: Perplexity value (lower is better)

    Example:
        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> perplexity = calculate_perplexity(text)
        >>> print(f"Perplexity: {perplexity:.2f}")
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Tokenize input
    encodings = tokenizer(text, return_tensors='pt').to(device)

    # Calculate loss (negative log-likelihood)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


def calculate_perplexity_sliding_window(
    text: str,
    model_name: str = "gpt2",
    max_length: int = 1024,
    stride: int = 512,
    device: str = "cpu"
) -> float:
    """
    Calculate perplexity for long texts using sliding window approach.

    For texts longer than model's max context length, this method
    uses overlapping windows to compute average perplexity.

    Args:
        text: Input text to evaluate
        model_name: HuggingFace model identifier
        max_length: Maximum sequence length per window
        stride: Overlap between windows (smaller = more overlap)
        device: Device to run model on

    Returns:
        average_perplexity: Average perplexity across all windows

    Example:
        >>> long_text = "..." * 1000  # Very long text
        >>> ppl = calculate_perplexity_sliding_window(long_text)
        >>> print(f"Average perplexity: {ppl:.2f}")
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Tokenize full text
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)

    # Calculate perplexity using sliding window
    seq_len = input_ids.size(1)
    nlls = []  # Negative log-likelihoods
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_window = input_ids[:, begin_loc:end_loc]
        target_ids = input_window.clone()
        target_ids[:, :-trg_len] = -100  # Only calculate loss on new tokens

        with torch.no_grad():
            outputs = model(input_window, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate average perplexity
    avg_nll = torch.stack(nlls).sum() / end_loc
    perplexity = torch.exp(avg_nll).item()

    return perplexity


def compare_models_perplexity(
    text: str,
    model_names: List[str]
) -> dict:
    """
    Compare perplexity across multiple language models.

    Args:
        text: Text to evaluate
        model_names: List of HuggingFace model identifiers

    Returns:
        results: Dictionary mapping model names to perplexity scores

    Example:
        >>> text = "Artificial intelligence is transforming technology."
        >>> models = ["gpt2", "gpt2-medium", "gpt2-large"]
        >>> results = compare_models_perplexity(text, models)
        >>> for model, ppl in sorted(results.items(), key=lambda x: x[1]):
        ...     print(f"{model}: {ppl:.2f}")
    """
    results = {}

    for model_name in model_names:
        try:
            ppl = calculate_perplexity(text, model_name)
            results[model_name] = ppl
            print(f"✓ {model_name}: {ppl:.2f}")
        except Exception as e:
            print(f"✗ {model_name}: Error - {str(e)}")
            results[model_name] = float('inf')

    return results


def calculate_perplexity_from_logprobs(logprobs: List[float]) -> float:
    """
    Calculate perplexity from pre-computed log probabilities.

    Useful when working with API responses that include logprobs
    (e.g., OpenAI API with logprobs=True).

    Args:
        logprobs: List of log probabilities for each token

    Returns:
        perplexity: Perplexity score

    Formula: perplexity = exp(-mean(logprobs))

    Example:
        >>> # From OpenAI API response
        >>> logprobs = [-0.5, -0.3, -0.8, -0.2]
        >>> ppl = calculate_perplexity_from_logprobs(logprobs)
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    if not logprobs:
        raise ValueError("logprobs list cannot be empty")

    # Calculate mean log probability
    mean_logprob = np.mean(logprobs)

    # Perplexity = exp(-mean(logprobs))
    perplexity = np.exp(-mean_logprob)

    return perplexity


# Example usage
if __name__ == "__main__":
    # Example 1: Basic perplexity calculation
    print("=" * 60)
    print("Example 1: Basic Perplexity Calculation")
    print("=" * 60)

    text = "The quick brown fox jumps over the lazy dog."
    perplexity = calculate_perplexity(text)
    print(f"Text: {text}")
    print(f"Perplexity: {perplexity:.2f}")
    print()

    # Example 2: Compare multiple models
    print("=" * 60)
    print("Example 2: Model Comparison")
    print("=" * 60)

    text = "Artificial intelligence is transforming how we interact with technology."
    models = ["gpt2", "gpt2-medium"]

    print(f"Text: {text}\n")
    results = compare_models_perplexity(text, models)
    print("\nRanking (lower is better):")
    for model, ppl in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {model}: {ppl:.2f}")
    print()

    # Example 3: Calculate from logprobs
    print("=" * 60)
    print("Example 3: Perplexity from Log Probabilities")
    print("=" * 60)

    # Simulated logprobs from API
    logprobs = [-0.5, -0.3, -0.8, -0.2, -0.6]
    ppl = calculate_perplexity_from_logprobs(logprobs)
    print(f"Log probabilities: {logprobs}")
    print(f"Perplexity: {ppl:.2f}")
    print()

    # Interpretation
    print("=" * 60)
    print("Interpretation Guide")
    print("=" * 60)
    print("Perplexity Range | Interpretation")
    print("-" * 60)
    print("< 10             | Excellent (model very confident)")
    print("10 - 50          | Good (typical for in-domain text)")
    print("50 - 100         | Fair (some uncertainty)")
    print("> 100            | Poor (high uncertainty, possible OOD)")
