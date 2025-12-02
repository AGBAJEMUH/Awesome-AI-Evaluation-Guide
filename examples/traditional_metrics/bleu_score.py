"""
BLEU Score (Bilingual Evaluation Understudy)

Precision-based metric measuring n-gram overlap between candidate and reference text.
Commonly used for machine translation and text generation evaluation.

"""

from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction
)
from typing import List, Union, Tuple
import numpy as np


def calculate_bleu(
    reference: Union[str, List[str]],
    candidate: str,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    smoothing: bool = True
) -> float:
    """
    Calculate BLEU score for single sentence evaluation.

    Args:
        reference: Reference text(s) - single string or list of reference strings
        candidate: Generated text (model output)
        weights: N-gram weights (default: uniform for 1-4 grams)
                 (1-gram, 2-gram, 3-gram, 4-gram)
        smoothing: Apply smoothing for short texts (default: True)

    Returns:
        bleu_score: BLEU score (0-1, higher is better)

    Example:
        >>> ref = "The cat sat on the mat"
        >>> cand = "The cat is sitting on the mat"
        >>> score = calculate_bleu(ref, cand)
        >>> print(f"BLEU: {score:.4f}")
    """
    # Handle single or multiple references
    if isinstance(reference, str):
        reference_tokens = [reference.split()]
    else:
        reference_tokens = [ref.split() for ref in reference]

    candidate_tokens = candidate.split()

    # Apply smoothing for short texts
    smoothing_function = SmoothingFunction().method1 if smoothing else None

    # Calculate BLEU
    bleu_score = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=weights,
        smoothing_function=smoothing_function
    )

    return bleu_score


def calculate_bleu_variants(
    reference: Union[str, List[str]],
    candidate: str
) -> dict:
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.

    Different n-gram weights emphasize different aspects:
    - BLEU-1: Word choice (unigram precision)
    - BLEU-2: Phrase quality (bigram precision)
    - BLEU-3: Sentence structure (trigram precision)
    - BLEU-4: Overall fluency (4-gram precision)

    Args:
        reference: Reference text(s)
        candidate: Generated text

    Returns:
        scores: Dictionary with BLEU-1 through BLEU-4 scores

    Example:
        >>> ref = "The cat sat on the mat"
        >>> cand = "The cat is on the mat"
        >>> scores = calculate_bleu_variants(ref, cand)
        >>> for name, score in scores.items():
        ...     print(f"{name}: {score:.4f}")
    """
    # Handle single or multiple references
    if isinstance(reference, str):
        reference_tokens = [reference.split()]
    else:
        reference_tokens = [ref.split() for ref in reference]

    candidate_tokens = candidate.split()
    smoothing = SmoothingFunction().method1

    scores = {
        'BLEU-1': sentence_bleu(
            reference_tokens, candidate_tokens,
            weights=(1, 0, 0, 0), smoothing_function=smoothing
        ),
        'BLEU-2': sentence_bleu(
            reference_tokens, candidate_tokens,
            weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
        ),
        'BLEU-3': sentence_bleu(
            reference_tokens, candidate_tokens,
            weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing
        ),
        'BLEU-4': sentence_bleu(
            reference_tokens, candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing
        )
    }

    return scores


def calculate_corpus_bleu(
    references: List[List[str]],
    candidates: List[str],
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """
    Calculate BLEU score for corpus (multiple sentences).

    Corpus-level BLEU is more reliable than averaging sentence-level scores.

    Args:
        references: List of reference lists (each can have multiple references)
                   Format: [[ref1_v1, ref1_v2], [ref2_v1], ...]
        candidates: List of candidate sentences
        weights: N-gram weights

    Returns:
        corpus_bleu_score: BLEU score for entire corpus

    Example:
        >>> references = [
        ...     ["The cat sat on the mat", "A cat is on the mat"],
        ...     ["The dog runs fast"]
        ... ]
        >>> candidates = [
        ...     "The cat is on the mat",
        ...     "The dog is running quickly"
        ... ]
        >>> score = calculate_corpus_bleu(references, candidates)
        >>> print(f"Corpus BLEU: {score:.4f}")
    """
    # Tokenize all references and candidates
    ref_tokens = [
        [ref.split() for ref in ref_list]
        for ref_list in references
    ]
    cand_tokens = [cand.split() for cand in candidates]

    # Calculate corpus BLEU
    score = corpus_bleu(ref_tokens, cand_tokens, weights=weights)

    return score


def calculate_self_bleu(candidates: List[str], sample_size: int = None) -> float:
    """
    Calculate Self-BLEU to measure diversity of generated texts.

    Self-BLEU measures how similar generated texts are to each other.
    Lower Self-BLEU indicates higher diversity (desirable for generation).

    Args:
        candidates: List of generated texts
        sample_size: Number of samples to use as references (default: all)

    Returns:
        self_bleu_score: Average BLEU when each text is compared to others

    Example:
        >>> generations = [
        ...     "The cat sat on the mat",
        ...     "A feline rested on the rug",
        ...     "The cat was on the mat"
        ... ]
        >>> diversity = calculate_self_bleu(generations)
        >>> print(f"Self-BLEU: {diversity:.4f} (lower = more diverse)")
    """
    if len(candidates) < 2:
        raise ValueError("Need at least 2 candidates for Self-BLEU")

    sample_size = sample_size or len(candidates)
    bleu_scores = []

    for i, candidate in enumerate(candidates):
        # Use all other candidates as references
        references = [
            cand for j, cand in enumerate(candidates[:sample_size])
            if i != j
        ]

        if references:
            score = calculate_bleu(references, candidate)
            bleu_scores.append(score)

    return np.mean(bleu_scores)


def bleu_with_analysis(
    reference: Union[str, List[str]],
    candidate: str
) -> dict:
    """
    Calculate BLEU score with detailed n-gram analysis.

    Provides insights into which n-grams are matching and missing.

    Args:
        reference: Reference text(s)
        candidate: Generated text

    Returns:
        analysis: Dictionary with BLEU score and n-gram statistics

    Example:
        >>> ref = "The quick brown fox jumps over the lazy dog"
        >>> cand = "The fast brown fox jumps over the sleepy dog"
        >>> analysis = bleu_with_analysis(ref, cand)
        >>> print(f"BLEU-4: {analysis['bleu_4']:.4f}")
        >>> print(f"Matching 1-grams: {analysis['unigram_precision']:.2%}")
    """
    # Calculate all BLEU variants
    scores = calculate_bleu_variants(reference, candidate)

    # Tokenize
    if isinstance(reference, str):
        ref_tokens = set(reference.split())
    else:
        ref_tokens = set()
        for ref in reference:
            ref_tokens.update(ref.split())

    cand_tokens = candidate.split()
    cand_set = set(cand_tokens)

    # Calculate precision statistics
    matching_tokens = ref_tokens & cand_set
    unigram_precision = len(matching_tokens) / len(cand_set) if cand_set else 0

    analysis = {
        **scores,
        'candidate_length': len(cand_tokens),
        'matching_unigrams': len(matching_tokens),
        'total_unigrams': len(cand_set),
        'unigram_precision': unigram_precision
    }

    return analysis


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic BLEU Calculation")
    print("=" * 60)

    reference = "The cat sat on the mat"
    candidate_good = "The cat is sitting on the mat"
    candidate_bad = "mat the on sat cat the"

    score_good = calculate_bleu(reference, candidate_good)
    score_bad = calculate_bleu(reference, candidate_bad)

    print(f"Reference: {reference}")
    print(f"\nGood candidate: {candidate_good}")
    print(f"BLEU Score: {score_good:.4f}")
    print(f"\nBad candidate: {candidate_bad}")
    print(f"BLEU Score: {score_bad:.4f}")
    print()

    # Example 2: Multiple references
    print("=" * 60)
    print("Example 2: Multiple Reference Translations")
    print("=" * 60)

    references = [
        "The cat sat on the mat",
        "A cat is sitting on a mat",
        "The feline rested on the rug"
    ]
    candidate = "The cat is on the mat"

    score = calculate_bleu(references, candidate)
    print(f"Candidate: {candidate}\n")
    print("References:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    print(f"\nMulti-reference BLEU: {score:.4f}")
    print()

    # Example 3: BLEU variants
    print("=" * 60)
    print("Example 3: BLEU-1 through BLEU-4")
    print("=" * 60)

    reference = "The quick brown fox jumps over the lazy dog"
    candidate = "The fast brown fox jumps over the sleepy dog"

    scores = calculate_bleu_variants(reference, candidate)
    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}\n")
    print("BLEU Variants:")
    for name, score in scores.items():
        print(f"  {name}: {score:.4f}")
    print()

    # Example 4: Self-BLEU for diversity
    print("=" * 60)
    print("Example 4: Self-BLEU (Diversity Measurement)")
    print("=" * 60)

    diverse_generations = [
        "The cat sat on the mat",
        "A feline rested comfortably on the rug",
        "The small kitten lay down on the carpet"
    ]

    repetitive_generations = [
        "The cat sat on the mat",
        "The cat was on the mat",
        "The cat sat on a mat"
    ]

    diverse_score = calculate_self_bleu(diverse_generations)
    repetitive_score = calculate_self_bleu(repetitive_generations)

    print("Diverse generations:")
    for i, gen in enumerate(diverse_generations, 1):
        print(f"  {i}. {gen}")
    print(f"Self-BLEU: {diverse_score:.4f} (lower = more diverse)\n")

    print("Repetitive generations:")
    for i, gen in enumerate(repetitive_generations, 1):
        print(f"  {i}. {gen}")
    print(f"Self-BLEU: {repetitive_score:.4f} (lower = more diverse)")
    print()

    # Example 5: Detailed analysis
    print("=" * 60)
    print("Example 5: BLEU with N-gram Analysis")
    print("=" * 60)

    reference = "The quick brown fox jumps over the lazy dog"
    candidate = "The fast brown fox jumps over the sleepy dog"

    analysis = bleu_with_analysis(reference, candidate)

    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}\n")
    print("Analysis:")
    print(f"  Candidate length: {analysis['candidate_length']} tokens")
    print(f"  Matching unigrams: {analysis['matching_unigrams']}/{analysis['total_unigrams']}")
    print(f"  Unigram precision: {analysis['unigram_precision']:.2%}")
    print(f"\n  BLEU-1: {analysis['BLEU-1']:.4f}")
    print(f"  BLEU-2: {analysis['BLEU-2']:.4f}")
    print(f"  BLEU-3: {analysis['BLEU-3']:.4f}")
    print(f"  BLEU-4: {analysis['BLEU-4']:.4f}")
    print()

    # Interpretation guide
    print("=" * 60)
    print("Interpretation Guide")
    print("=" * 60)
    print("BLEU Score Range | Quality Interpretation")
    print("-" * 60)
    print("< 0.10           | Almost no useful overlap")
    print("0.10 - 0.20      | Poor, hard to understand")
    print("0.20 - 0.30      | Understandable, many errors")
    print("0.30 - 0.40      | Good, some fluency issues")
    print("0.40 - 0.50      | High quality, minor issues")
    print("> 0.50           | Very high quality, near-human")
    print("\nNote: BLEU scores are task-dependent. Translation typically")
    print("      achieves higher scores than open-ended generation.")
