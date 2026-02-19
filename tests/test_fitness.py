"""Tests for fitness evaluation module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from genesis.core.fitness import (
    FitnessEvaluator,
    FitnessResult,
    PerplexityFitness,
    AccuracyFitness,
    CompositeFitness,
    CustomFitness,
    create_fitness_evaluator,
)


class DummyModel(nn.Module):
    """Simple dummy model for testing."""

    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.linear(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        class Output:
            pass

        output = Output()
        output.logits = logits
        output.loss = loss

        return output


def create_dummy_dataloader(batch_size=4, seq_len=10, vocab_size=100, num_batches=2):
    """Create a dummy dataloader for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size)


class TestFitnessResult:
    """Tests for FitnessResult dataclass."""

    def test_fitness_result_creation(self):
        """Test FitnessResult creation."""
        result = FitnessResult(
            score=0.85,
            metrics={"accuracy": 0.85, "loss": 0.5},
        )

        assert result.score == 0.85
        assert result.metrics["accuracy"] == 0.85

    def test_fitness_result_float_conversion(self):
        """Test FitnessResult float conversion."""
        result = FitnessResult(score=0.75)

        assert float(result) == 0.75


class TestPerplexityFitness:
    """Tests for PerplexityFitness evaluator."""

    def test_perplexity_fitness_evaluation(self):
        """Test perplexity fitness evaluation."""
        model = DummyModel()
        dataloader = create_dummy_dataloader()

        # Wrap dataloader to return dicts
        def dict_dataloader():
            for batch in dataloader:
                yield {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                }

        evaluator = PerplexityFitness(
            dataloader=list(dict_dataloader()),
            device="cpu",
            max_samples=10,
        )

        result = evaluator.evaluate(model)

        assert isinstance(result, FitnessResult)
        assert 0 <= result.score <= 1  # Fitness should be bounded
        assert "perplexity" in result.metrics
        assert "loss" in result.metrics

    def test_perplexity_fitness_lower_perplexity_higher_fitness(self):
        """Test that lower perplexity gives higher fitness."""
        # Use labels that are all class-0 so we can control which model wins.
        dataloader = [
            {
                "input_ids": torch.zeros(4, 10, dtype=torch.long),
                "attention_mask": torch.ones(4, 10, dtype=torch.long),
                "labels": torch.zeros(4, 10, dtype=torch.long),  # all token 0
            }
        ]

        # good_model: zero embeddings + bias strongly predicts class 0 → low loss
        good_model = DummyModel()
        good_model.embedding.weight.data.zero_()
        good_model.linear.weight.data.zero_()
        good_model.linear.bias.data.zero_()
        good_model.linear.bias.data[0] = 20.0  # always predicts class 0

        # bad_model: same setup but strongly predicts class 1 (wrong) → high loss
        bad_model = DummyModel()
        bad_model.embedding.weight.data.zero_()
        bad_model.linear.weight.data.zero_()
        bad_model.linear.bias.data.zero_()
        bad_model.linear.bias.data[1] = 20.0  # always predicts class 1

        good_result = PerplexityFitness(dataloader=dataloader, device="cpu").evaluate(good_model)
        bad_result = PerplexityFitness(dataloader=dataloader, device="cpu").evaluate(bad_model)

        assert good_result.score > bad_result.score


class TestAccuracyFitness:
    """Tests for AccuracyFitness evaluator."""

    def test_accuracy_fitness_evaluation(self):
        """Test accuracy fitness evaluation."""

        class ClassificationModel(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.linear = nn.Linear(32, num_classes)

            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Dummy: just use input_ids sum as features
                features = input_ids.float().mean(dim=-1, keepdim=True).expand(-1, 32)
                logits = self.linear(features)

                class Output:
                    pass

                output = Output()
                output.logits = logits
                return output

        model = ClassificationModel()

        # Create dataloader with labels
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 10, (20,))

        def dict_dataloader():
            for i in range(0, 20, 4):
                yield {
                    "input_ids": input_ids[i : i + 4],
                    "attention_mask": torch.ones(4, 10),
                    "labels": labels[i : i + 4],
                }

        evaluator = AccuracyFitness(
            dataloader=list(dict_dataloader()),
            device="cpu",
        )

        result = evaluator.evaluate(model)

        assert isinstance(result, FitnessResult)
        assert 0 <= result.score <= 1
        assert "accuracy" in result.metrics


class TestCompositeFitness:
    """Tests for CompositeFitness evaluator."""

    def test_composite_fitness(self):
        """Test composite fitness evaluation."""

        class DummyEvaluator(FitnessEvaluator):
            def __init__(self, fixed_score):
                self.fixed_score = fixed_score

            def evaluate(self, model, state_dict=None):
                return FitnessResult(score=self.fixed_score)

        eval1 = DummyEvaluator(0.8)
        eval2 = DummyEvaluator(0.6)

        composite = CompositeFitness(
            evaluators=[eval1, eval2],
            weights=[0.5, 0.5],
        )

        result = composite.evaluate(nn.Linear(1, 1))

        # Weighted average: 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert abs(result.score - 0.7) < 1e-6

    def test_composite_fitness_unequal_weights(self):
        """Test composite fitness with unequal weights."""

        class DummyEvaluator(FitnessEvaluator):
            def __init__(self, fixed_score):
                self.fixed_score = fixed_score

            def evaluate(self, model, state_dict=None):
                return FitnessResult(score=self.fixed_score)

        eval1 = DummyEvaluator(1.0)
        eval2 = DummyEvaluator(0.0)

        composite = CompositeFitness(
            evaluators=[eval1, eval2],
            weights=[0.75, 0.25],
        )

        result = composite.evaluate(nn.Linear(1, 1))

        # Weighted average: 0.75 * 1.0 + 0.25 * 0.0 = 0.75
        assert abs(result.score - 0.75) < 1e-6


class TestCustomFitness:
    """Tests for CustomFitness wrapper."""

    def test_custom_fitness(self):
        """Test custom fitness function wrapper."""

        def my_fitness(model):
            # Count parameters as a dummy fitness
            return sum(p.numel() for p in model.parameters()) / 1000

        evaluator = CustomFitness(my_fitness)
        model = nn.Linear(10, 5)  # 55 parameters

        result = evaluator.evaluate(model)

        assert result.score == 0.055  # 55 / 1000

    def test_custom_fitness_with_state_dict(self):
        """Test custom fitness with state dict loading."""

        def my_fitness(model):
            return model.weight.mean().item()

        evaluator = CustomFitness(my_fitness)
        model = nn.Linear(5, 5, bias=False)

        state_dict = {"weight": torch.ones(5, 5) * 0.5}
        result = evaluator.evaluate(model, state_dict=state_dict)

        assert abs(result.score - 0.5) < 1e-6


class TestFitnessEvaluatorFactory:
    """Tests for fitness evaluator factory."""

    def test_create_perplexity_evaluator(self):
        """Test creating perplexity evaluator."""

        def dict_dataloader():
            yield {"input_ids": torch.randint(0, 100, (4, 10))}

        evaluator = create_fitness_evaluator(
            fitness_type="perplexity",
            dataloader=list(dict_dataloader()),
            device="cpu",
        )

        assert isinstance(evaluator, PerplexityFitness)

    def test_create_unknown_evaluator_raises(self):
        """Test that unknown type raises error."""
        with pytest.raises(ValueError):
            create_fitness_evaluator(
                fitness_type="unknown",
                dataloader=[],
            )
