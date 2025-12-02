"""
ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

Recall-based metric for summarization evaluation. Measures overlap
between generated and reference summaries.

Main variants:
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest Common Subsequence
- ROUGE-Lsum: LCS for multi-sentence summaries

"""

from rouge_score import rouge_scorer
from typing import List, Dict, Union
import numpy as np


def calculate_rouge(
    reference: str,
    candidate: str,
    metrics: List[str] = ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer: bool = True
) -> dict:
    """
    Calculate ROUGE scores for summary evaluation.

    Args:
        reference: Reference summary (ground truth)
        candidate: Generated summary (model output)
        metrics: List of ROUGE variants to compute
                 Options: 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
        use_stemmer: Apply Porter stemming (default: True)

    Returns:
        scores: Dictionary with precision, recall, F1 for each metric

    Example:
        >>> ref = "The study found that exercise reduces blood pressure."
        >>> cand = "Exercise linked to reduced blood pressure."
        >>> scores = calculate_rouge(ref, cand)
        >>> print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
    """
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer)
    scores = scorer.score(reference, candidate)

    return scores


def calculate_rouge_detailed(
    reference: str,
    candidate: str
) -> dict:
    """
    Calculate all ROUGE variants with detailed metrics.

    Returns precision, recall, and F1 for each ROUGE type.

    Args:
        reference: Reference summary
        candidate: Generated summary

    Returns:
        detailed_scores: Dictionary with all ROUGE metrics

    Example:
        >>> ref = "Machine learning improves healthcare outcomes."
        >>> cand = "ML enhances medical results."
        >>> scores = calculate_rouge_detailed(ref, cand)
        >>> for metric, score in scores.items():
        ...     print(f"{metric}: P={score['precision']:.3f}, "
        ...           f"R={score['recall']:.3f}, F1={score['f1']:.3f}")
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        use_stemmer=True
    )

    raw_scores = scorer.score(reference, candidate)

    # Restructure for easier access
    detailed_scores = {}
    for metric_name, score in raw_scores.items():
        detailed_scores[metric_name] = {
            'precision': score.precision,
            'recall': score.recall,
            'f1': score.fmeasure
        }

    return detailed_scores


def calculate_rouge_corpus(
    references: List[str],
    candidates: List[str],
    aggregate: bool = True
) -> dict:
    """
    Calculate ROUGE scores for multiple summary pairs.

    Args:
        references: List of reference summaries
        candidates: List of generated summaries (same length as references)
        aggregate: If True, return average scores; if False, return all scores

    Returns:
        scores: Aggregated or individual ROUGE scores

    Example:
        >>> references = [
        ...     "Exercise reduces blood pressure significantly.",
        ...     "Coffee consumption linked to longevity."
        ... ]
        >>> candidates = [
        ...     "Exercise lowers blood pressure.",
        ...     "Coffee drinkers live longer."
        ... ]
        >>> scores = calculate_rouge_corpus(references, candidates)
        >>> print(f"Average ROUGE-1 F1: {scores['rouge1']['f1']:.4f}")
    """
    if len(references) != len(candidates):
        raise ValueError("References and candidates must have same length")

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    all_scores = {
        'rouge1': {'precision': [], 'recall': [], 'f1': []},
        'rouge2': {'precision': [], 'recall': [], 'f1': []},
        'rougeL': {'precision': [], 'recall': [], 'f1': []}
    }

    # Calculate scores for each pair
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            all_scores[metric]['precision'].append(scores[metric].precision)
            all_scores[metric]['recall'].append(scores[metric].recall)
            all_scores[metric]['f1'].append(scores[metric].fmeasure)

    if aggregate:
        # Return average scores
        aggregated = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            aggregated[metric] = {
                'precision': np.mean(all_scores[metric]['precision']),
                'recall': np.mean(all_scores[metric]['recall']),
                'f1': np.mean(all_scores[metric]['f1'])
            }
        return aggregated
    else:
        # Return individual scores
        return all_scores


def calculate_rouge_multi_reference(
    references: List[str],
    candidate: str
) -> dict:
    """
    Calculate ROUGE with multiple reference summaries.

    Takes the maximum score across all references for each metric.

    Args:
        references: List of acceptable reference summaries
        candidate: Generated summary

    Returns:
        best_scores: Best ROUGE scores across all references

    Example:
        >>> references = [
        ...     "Exercise reduces blood pressure significantly.",
        ...     "Physical activity lowers BP levels.",
        ...     "Working out decreases hypertension."
        ... ]
        >>> candidate = "Exercise lowers blood pressure."
        >>> scores = calculate_rouge_multi_reference(references, candidate)
        >>> print(f"Best ROUGE-1 F1: {scores['rouge1']['f1']:.4f}")
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    # Calculate scores against each reference
    all_scores = {
        'rouge1': {'precision': [], 'recall': [], 'f1': []},
        'rouge2': {'precision': [], 'recall': [], 'f1': []},
        'rougeL': {'precision': [], 'recall': [], 'f1': []}
    }

    for reference in references:
        scores = scorer.score(reference, candidate)

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            all_scores[metric]['precision'].append(scores[metric].precision)
            all_scores[metric]['recall'].append(scores[metric].recall)
            all_scores[metric]['f1'].append(scores[metric].fmeasure)

    # Take maximum score for each metric
    best_scores = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        best_scores[metric] = {
            'precision': max(all_scores[metric]['precision']),
            'recall': max(all_scores[metric]['recall']),
            'f1': max(all_scores[metric]['f1'])
        }

    return best_scores


def rouge_with_analysis(
    reference: str,
    candidate: str
) -> dict:
    """
    Calculate ROUGE with additional analysis metrics.

    Provides context on summary characteristics beyond ROUGE scores.

    Args:
        reference: Reference summary
        candidate: Generated summary

    Returns:
        analysis: ROUGE scores plus length and compression statistics

    Example:
        >>> ref = "The research study analyzed 1000 patients over 5 years."
        >>> cand = "Study analyzed 1000 patients."
        >>> analysis = rouge_with_analysis(ref, cand)
        >>> print(f"Compression ratio: {analysis['compression_ratio']:.2f}")
    """
    scores = calculate_rouge_detailed(reference, candidate)

    # Calculate additional metrics
    ref_words = reference.split()
    cand_words = candidate.split()

    analysis = {
        **scores,
        'reference_length': len(ref_words),
        'candidate_length': len(cand_words),
        'compression_ratio': len(cand_words) / len(ref_words) if ref_words else 0,
        'length_difference': abs(len(ref_words) - len(cand_words))
    }

    return analysis


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Basic ROUGE Calculation")
    print("=" * 70)

    reference = ("The study found that people who exercise regularly "
                 "tend to have significantly lower blood pressure.")
    candidate = "Regular exercise is linked to reduced blood pressure."

    scores = calculate_rouge(reference, candidate)

    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}\n")
    print("ROUGE Scores:")
    for metric, score in scores.items():
        print(f"  {metric.upper()}:")
        print(f"    Precision: {score.precision:.4f}")
        print(f"    Recall:    {score.recall:.4f}")
        print(f"    F1:        {score.fmeasure:.4f}")
    print()

    # Example 2: All ROUGE variants
    print("=" * 70)
    print("Example 2: Detailed ROUGE Analysis")
    print("=" * 70)

    reference = ("Machine learning algorithms are transforming healthcare "
                 "by enabling more accurate diagnoses and personalized treatments.")
    candidate = "ML improves healthcare through better diagnosis and treatment."

    detailed = calculate_rouge_detailed(reference, candidate)

    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}\n")
    print(f"{'Metric':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for metric, scores in detailed.items():
        print(f"{metric.upper():<12} "
              f"{scores['precision']:<12.4f} "
              f"{scores['recall']:<12.4f} "
              f"{scores['f1']:<12.4f}")
    print()

    # Example 3: Corpus evaluation
    print("=" * 70)
    print("Example 3: Corpus-Level ROUGE (Multiple Summaries)")
    print("=" * 70)

    references = [
        "Exercise reduces blood pressure significantly in adults.",
        "Coffee consumption is associated with increased longevity.",
        "Mediterranean diet improves cardiovascular health outcomes."
    ]

    candidates = [
        "Exercise lowers blood pressure in adults.",
        "Coffee drinkers tend to live longer lives.",
        "Mediterranean diet benefits heart health."
    ]

    corpus_scores = calculate_rouge_corpus(references, candidates)

    print("Corpus Statistics:")
    print(f"  Number of summary pairs: {len(references)}\n")
    print("Average ROUGE Scores:")
    print(f"{'Metric':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for metric, scores in corpus_scores.items():
        print(f"{metric.upper():<12} "
              f"{scores['precision']:<12.4f} "
              f"{scores['recall']:<12.4f} "
              f"{scores['f1']:<12.4f}")
    print()

    # Example 4: Multiple references
    print("=" * 70)
    print("Example 4: Multiple Reference Summaries")
    print("=" * 70)

    references = [
        "Exercise significantly reduces blood pressure in patients.",
        "Physical activity lowers BP levels substantially.",
        "Regular workouts decrease hypertension risk."
    ]
    candidate = "Exercise reduces blood pressure effectively."

    multi_ref_scores = calculate_rouge_multi_reference(references, candidate)

    print(f"Candidate: {candidate}\n")
    print("References:")
    for i, ref in enumerate(references, 1):
        print(f"  {i}. {ref}")
    print("\nBest ROUGE Scores (max across references):")
    print(f"{'Metric':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for metric, scores in multi_ref_scores.items():
        print(f"{metric.upper():<12} "
              f"{scores['precision']:<12.4f} "
              f"{scores['recall']:<12.4f} "
              f"{scores['f1']:<12.4f}")
    print()

    # Example 5: Analysis with compression
    print("=" * 70)
    print("Example 5: ROUGE with Compression Analysis")
    print("=" * 70)

    reference = ("The comprehensive research study meticulously analyzed "
                 "data from over 1000 patients collected over a period of 5 years.")
    candidate = "Study analyzed 1000 patients over 5 years."

    analysis = rouge_with_analysis(reference, candidate)

    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}\n")
    print("Summary Statistics:")
    print(f"  Reference length: {analysis['reference_length']} words")
    print(f"  Candidate length: {analysis['candidate_length']} words")
    print(f"  Compression ratio: {analysis['compression_ratio']:.2f}")
    print(f"  Length difference: {analysis['length_difference']} words\n")
    print("ROUGE Scores:")
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        if metric in analysis:
            print(f"  {metric.upper()} F1: {analysis[metric]['f1']:.4f}")
    print()

    # Interpretation guide
    print("=" * 70)
    print("Interpretation Guide")
    print("=" * 70)
    print("ROUGE Score Range | Quality Interpretation")
    print("-" * 70)
    print("< 0.20            | Poor coverage of reference content")
    print("0.20 - 0.35       | Fair, missing important information")
    print("0.35 - 0.50       | Good, captures main points")
    print("0.50 - 0.65       | Very good, comprehensive coverage")
    print("> 0.65            | Excellent, near-complete coverage")
    print("\nNote:")
    print("- ROUGE-1: Measures content selection (what information)")
    print("- ROUGE-2: Measures phrasing quality (how information)")
    print("- ROUGE-L: Measures fluency and structure")
    print("- Higher recall = better coverage, higher precision = less redundancy")
