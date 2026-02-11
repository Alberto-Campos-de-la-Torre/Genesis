"""Fitness evaluation functions for evolutionary optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitnessResult:
    """Container for fitness evaluation results."""

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return self.score


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluators."""

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        """
        Evaluate fitness of a model.

        Args:
            model: Model to evaluate
            state_dict: Optional state dict to load before evaluation

        Returns:
            FitnessResult containing score and metrics
        """
        pass

    def __call__(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        return self.evaluate(model, state_dict)


class PerplexityFitness(FitnessEvaluator):
    """Fitness based on language model perplexity (lower is better, converted to fitness)."""

    def __init__(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ):
        self.dataloader = dataloader
        self.device = device
        self.max_samples = max_samples

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)

        total_loss = 0.0
        total_tokens = 0
        samples_processed = 0

        for batch in self.dataloader:
            if self.max_samples and samples_processed >= self.max_samples:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            samples_processed += input_ids.size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Convert perplexity to fitness (lower perplexity = higher fitness)
        # Using negative log to convert to maximization problem
        fitness = 1.0 / (1.0 + perplexity)

        return FitnessResult(
            score=fitness,
            metrics={
                "perplexity": perplexity,
                "loss": avg_loss,
                "tokens_evaluated": total_tokens,
            },
        )


class AccuracyFitness(FitnessEvaluator):
    """Fitness based on classification accuracy."""

    def __init__(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ):
        self.dataloader = dataloader
        self.device = device
        self.max_samples = max_samples

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)

        correct = 0
        total = 0
        samples_processed = 0

        for batch in self.dataloader:
            if self.max_samples and samples_processed >= self.max_samples:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predictions
            predictions = logits.argmax(dim=-1)

            # For sequence classification, compare directly
            # For token classification, flatten
            if predictions.dim() > 1:
                mask = attention_mask.bool()
                predictions = predictions[mask]
                labels = labels[mask]

            correct += (predictions == labels).sum().item()
            total += labels.numel()
            samples_processed += input_ids.size(0)

        accuracy = correct / total if total > 0 else 0.0

        return FitnessResult(
            score=accuracy,
            metrics={
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            },
        )


class QAFitness(FitnessEvaluator):
    """Fitness for question-answering tasks."""

    def __init__(
        self,
        dataloader: DataLoader,
        tokenizer: Any,
        device: str = "cuda",
        max_samples: Optional[int] = None,
        max_new_tokens: int = 50,
    ):
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.eval()
        model.to(self.device)

        exact_match = 0
        f1_scores = []
        samples_processed = 0

        for batch in self.dataloader:
            if self.max_samples and samples_processed >= self.max_samples:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            target_texts = batch.get("target_text", [])

            # Generate answers
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            # Decode predictions
            predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pred, target in zip(predictions, target_texts):
                pred_normalized = self._normalize(pred)
                target_normalized = self._normalize(target)

                # Exact match
                if pred_normalized == target_normalized:
                    exact_match += 1

                # F1 score
                f1 = self._compute_f1(pred_normalized, target_normalized)
                f1_scores.append(f1)

            samples_processed += input_ids.size(0)

        em_score = exact_match / samples_processed if samples_processed > 0 else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # Combined fitness
        fitness = 0.5 * em_score + 0.5 * avg_f1

        return FitnessResult(
            score=fitness,
            metrics={
                "exact_match": em_score,
                "f1": avg_f1,
                "samples_evaluated": samples_processed,
            },
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    def _compute_f1(self, pred: str, target: str) -> float:
        """Compute F1 score between prediction and target."""
        pred_tokens = set(pred.split())
        target_tokens = set(target.split())

        if not pred_tokens or not target_tokens:
            return float(pred_tokens == target_tokens)

        common = pred_tokens & target_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(target_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)


class CompositeFitness(FitnessEvaluator):
    """Combine multiple fitness evaluators with weights."""

    def __init__(
        self,
        evaluators: list[FitnessEvaluator],
        weights: Optional[list[float]] = None,
    ):
        self.evaluators = evaluators
        self.weights = weights or [1.0 / len(evaluators)] * len(evaluators)
        assert len(self.weights) == len(self.evaluators)

    def evaluate(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        results = []
        for evaluator in self.evaluators:
            result = evaluator.evaluate(model, state_dict)
            results.append(result)

        # Weighted sum of scores
        total_score = sum(w * r.score for w, r in zip(self.weights, results))

        # Combine metrics
        combined_metrics = {}
        for i, result in enumerate(results):
            for key, value in result.metrics.items():
                combined_metrics[f"evaluator_{i}_{key}"] = value

        return FitnessResult(
            score=total_score,
            metrics=combined_metrics,
            metadata={"evaluator_scores": [r.score for r in results]},
        )


class CustomFitness(FitnessEvaluator):
    """Wrapper for custom fitness functions."""

    def __init__(self, fitness_fn: Callable[[nn.Module], float]):
        self.fitness_fn = fitness_fn

    def evaluate(
        self,
        model: nn.Module,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> FitnessResult:
        if state_dict is not None:
            model.load_state_dict(state_dict)

        score = self.fitness_fn(model)
        return FitnessResult(score=score)


def create_fitness_evaluator(
    fitness_type: str,
    dataloader: DataLoader,
    device: str = "cuda",
    **kwargs,
) -> FitnessEvaluator:
    """
    Factory function to create fitness evaluators.

    Args:
        fitness_type: Type of fitness ('perplexity', 'accuracy', 'qa')
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        **kwargs: Additional arguments for the evaluator

    Returns:
        FitnessEvaluator instance
    """
    evaluators = {
        "perplexity": PerplexityFitness,
        "accuracy": AccuracyFitness,
        "qa": QAFitness,
    }

    if fitness_type not in evaluators:
        raise ValueError(f"Unknown fitness type: {fitness_type}")

    return evaluators[fitness_type](dataloader=dataloader, device=device, **kwargs)
