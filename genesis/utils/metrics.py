"""Evaluation metrics for Genesis experiments."""

from typing import Any, Optional
import torch
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def compute_perplexity(
    loss: torch.Tensor,
    base: float = np.e,
) -> float:
    """
    Compute perplexity from loss.

    Args:
        loss: Cross-entropy loss
        base: Base for exponential (e for natural, 2 for bits)

    Returns:
        Perplexity value
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return base ** loss


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute accuracy between predictions and labels.

    Args:
        predictions: Predicted class indices
        labels: Ground truth labels
        ignore_index: Index to ignore in calculation

    Returns:
        Accuracy (0-1)
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0

    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()

    return (correct / total).item()


def compute_bleu(
    predictions: list[str],
    references: list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """
    Compute BLEU score.

    Args:
        predictions: List of predicted strings
        references: List of reference strings
        max_n: Maximum n-gram size
        smooth: Whether to apply smoothing

    Returns:
        Dictionary with BLEU scores
    """
    from collections import Counter

    def get_ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    total_matches = defaultdict(int)
    total_counts = defaultdict(int)
    total_ref_length = 0
    total_pred_length = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        total_ref_length += len(ref_tokens)
        total_pred_length += len(pred_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            for ngram, count in pred_ngrams.items():
                total_matches[n] += min(count, ref_ngrams.get(ngram, 0))
                total_counts[n] += count

    # Compute precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            precision = 0.0
        else:
            precision = total_matches[n] / total_counts[n]
            if smooth and precision == 0:
                precision = 1 / (total_counts[n] + 1)
        precisions.append(precision)

    # Compute brevity penalty
    if total_pred_length == 0:
        bp = 0.0
    elif total_pred_length >= total_ref_length:
        bp = 1.0
    else:
        bp = np.exp(1 - total_ref_length / total_pred_length)

    # Compute BLEU scores
    results = {}
    for n in range(1, max_n + 1):
        if any(p == 0 for p in precisions[:n]):
            bleu_n = 0.0
        else:
            log_precision = sum(np.log(p) for p in precisions[:n]) / n
            bleu_n = bp * np.exp(log_precision)
        results[f"bleu_{n}"] = bleu_n

    results["brevity_penalty"] = bp

    return results


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary with ROUGE scores
    """
    from collections import Counter

    def get_lcs_length(a: list[str], b: list[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    rouge_1_f1 = []
    rouge_2_f1 = []
    rouge_l_f1 = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # ROUGE-1 (unigrams)
        pred_unigrams = Counter(pred_tokens)
        ref_unigrams = Counter(ref_tokens)
        overlap = sum((pred_unigrams & ref_unigrams).values())

        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            f1 = 0.0
        rouge_1_f1.append(f1)

        # ROUGE-2 (bigrams)
        pred_bigrams = Counter(
            tuple(pred_tokens[i : i + 2]) for i in range(len(pred_tokens) - 1)
        )
        ref_bigrams = Counter(
            tuple(ref_tokens[i : i + 2]) for i in range(len(ref_tokens) - 1)
        )
        overlap = sum((pred_bigrams & ref_bigrams).values())

        if len(pred_bigrams) > 0 and len(ref_bigrams) > 0:
            precision = overlap / len(pred_bigrams)
            recall = overlap / len(ref_bigrams)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            f1 = 0.0
        rouge_2_f1.append(f1)

        # ROUGE-L (LCS)
        lcs_len = get_lcs_length(pred_tokens, ref_tokens)
        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            f1 = 0.0
        rouge_l_f1.append(f1)

    return {
        "rouge_1": np.mean(rouge_1_f1),
        "rouge_2": np.mean(rouge_2_f1),
        "rouge_l": np.mean(rouge_l_f1),
    }


class MetricsTracker:
    """
    Track and aggregate metrics during training/evaluation.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._step = 0

    def update(self, metrics: dict[str, float]) -> None:
        """
        Update with new metric values.

        Args:
            metrics: Dictionary of metric names and values
        """
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._metrics[name].append(value)

        self._step += 1

    def get_average(self, name: str, window: Optional[int] = None) -> float:
        """
        Get average of a metric.

        Args:
            name: Metric name
            window: Optional window size (uses all values if None)

        Returns:
            Average value
        """
        if name not in self._metrics:
            return 0.0

        values = self._metrics[name]
        if window:
            values = values[-window:]

        return np.mean(values) if values else 0.0

    def get_latest(self, name: str) -> float:
        """Get most recent value of a metric."""
        if name not in self._metrics or not self._metrics[name]:
            return 0.0
        return self._metrics[name][-1]

    def get_all(self, name: str) -> list[float]:
        """Get all values of a metric."""
        return self._metrics.get(name, []).copy()

    def get_summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dictionary with stats for each metric
        """
        summary = {}
        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                    "count": len(values),
                }
        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._step = 0

    @property
    def step(self) -> int:
        """Current step count."""
        return self._step


class FitnessTracker:
    """
    Track fitness statistics for evolutionary algorithms.
    """

    def __init__(self):
        """Initialize fitness tracker."""
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.diversity_history: list[float] = []
        self.generation_history: list[int] = []

    def update(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: Optional[float] = None,
    ) -> None:
        """
        Update with generation statistics.

        Args:
            generation: Generation number
            best_fitness: Best fitness in population
            avg_fitness: Average fitness
            diversity: Population diversity
        """
        self.generation_history.append(generation)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        if diversity is not None:
            self.diversity_history.append(diversity)

    def get_improvement(self, window: int = 10) -> float:
        """
        Get fitness improvement over window.

        Args:
            window: Number of generations to consider

        Returns:
            Improvement (positive = improving)
        """
        if len(self.best_fitness_history) < window:
            return 0.0

        recent = self.best_fitness_history[-window:]
        return recent[-1] - recent[0]

    def is_converged(
        self,
        threshold: float = 1e-4,
        window: int = 10,
    ) -> bool:
        """
        Check if evolution has converged.

        Args:
            threshold: Improvement threshold
            window: Window size to check

        Returns:
            True if converged
        """
        if len(self.best_fitness_history) < window:
            return False

        improvement = abs(self.get_improvement(window))
        return improvement < threshold

    def get_summary(self) -> dict[str, Any]:
        """Get summary of fitness tracking."""
        return {
            "generations": len(self.generation_history),
            "best_fitness": max(self.best_fitness_history) if self.best_fitness_history else 0,
            "final_best": self.best_fitness_history[-1] if self.best_fitness_history else 0,
            "final_avg": self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
            "improvement": self.get_improvement(),
            "converged": self.is_converged(),
        }
