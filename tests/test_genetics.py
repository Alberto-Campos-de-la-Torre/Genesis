"""Tests for genetics module."""

import pytest
import torch
import numpy as np

from genesis.core.genetics import (
    slerp,
    crossover,
    mutate,
    Genetics,
    blend_state_dicts,
)


class TestSLERP:
    """Tests for SLERP interpolation."""

    def test_slerp_basic(self):
        """Test basic SLERP interpolation."""
        v0 = torch.tensor([1.0, 0.0, 0.0])
        v1 = torch.tensor([0.0, 1.0, 0.0])

        # At t=0, should be close to v0
        result = slerp(0.0, v0, v1)
        assert torch.allclose(result, v0, atol=1e-5)

        # At t=1, should be close to v1
        result = slerp(1.0, v0, v1)
        assert torch.allclose(result, v1, atol=1e-5)

    def test_slerp_midpoint(self):
        """Test SLERP at midpoint."""
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])

        result = slerp(0.5, v0, v1)

        # At midpoint, magnitude should be preserved
        expected_magnitude = (v0.norm() + v1.norm()) / 2
        assert abs(result.norm().item() - expected_magnitude) < 0.1

    def test_slerp_preserves_shape(self):
        """Test that SLERP preserves tensor shape."""
        v0 = torch.randn(10, 20)
        v1 = torch.randn(10, 20)

        result = slerp(0.5, v0, v1)
        assert result.shape == v0.shape

    def test_slerp_parallel_vectors(self):
        """Test SLERP with nearly parallel vectors."""
        v0 = torch.tensor([1.0, 0.0, 0.0])
        v1 = torch.tensor([1.0 + 1e-9, 0.0, 0.0])

        # Should not crash with parallel vectors
        result = slerp(0.5, v0, v1)
        assert result.shape == v0.shape


class TestCrossover:
    """Tests for crossover operations."""

    def test_crossover_slerp(self):
        """Test SLERP crossover method."""
        parent1 = {"weight": torch.randn(10, 10)}
        parent2 = {"weight": torch.randn(10, 10)}

        child = crossover(parent1, parent2, crossover_rate=1.0, method="slerp")

        assert "weight" in child
        assert child["weight"].shape == parent1["weight"].shape

    def test_crossover_uniform(self):
        """Test uniform crossover method."""
        parent1 = {"w1": torch.ones(5), "w2": torch.ones(5)}
        parent2 = {"w1": torch.zeros(5), "w2": torch.zeros(5)}

        child = crossover(parent1, parent2, crossover_rate=1.0, method="uniform")

        assert "w1" in child
        assert "w2" in child

    def test_crossover_single_point(self):
        """Test single-point crossover method."""
        parent1 = {"a": torch.ones(5), "b": torch.ones(5), "c": torch.ones(5)}
        parent2 = {"a": torch.zeros(5), "b": torch.zeros(5), "c": torch.zeros(5)}

        child = crossover(parent1, parent2, crossover_rate=1.0, method="single_point")

        assert len(child) == 3

    def test_crossover_no_crossover(self):
        """Test that crossover rate 0 returns copy of parent1."""
        parent1 = {"weight": torch.ones(5)}
        parent2 = {"weight": torch.zeros(5)}

        child = crossover(parent1, parent2, crossover_rate=0.0)

        assert torch.allclose(child["weight"], parent1["weight"])


class TestMutation:
    """Tests for mutation operations."""

    def test_mutate_gaussian(self):
        """Test Gaussian mutation."""
        state_dict = {"weight": torch.zeros(100)}

        mutated = mutate(
            state_dict,
            mutation_rate=1.0,
            mutation_scale=0.1,
            mutation_prob_per_weight=1.0,
            method="gaussian",
        )

        # With all weights mutated, should not be all zeros
        assert not torch.allclose(mutated["weight"], torch.zeros(100))

    def test_mutate_uniform(self):
        """Test uniform mutation."""
        state_dict = {"weight": torch.zeros(100)}

        mutated = mutate(
            state_dict,
            mutation_rate=1.0,
            mutation_scale=0.1,
            mutation_prob_per_weight=1.0,
            method="uniform",
        )

        assert not torch.allclose(mutated["weight"], torch.zeros(100))

    def test_mutate_adaptive(self):
        """Test adaptive mutation."""
        state_dict = {"weight": torch.ones(100)}

        mutated = mutate(
            state_dict,
            mutation_rate=1.0,
            mutation_scale=0.1,
            mutation_prob_per_weight=1.0,
            method="adaptive",
        )

        assert mutated["weight"].shape == state_dict["weight"].shape

    def test_mutate_no_mutation(self):
        """Test that mutation rate 0 returns copy."""
        state_dict = {"weight": torch.ones(5)}

        mutated = mutate(state_dict, mutation_rate=0.0)

        assert torch.allclose(mutated["weight"], state_dict["weight"])

    def test_mutate_preserves_shape(self):
        """Test that mutation preserves tensor shapes."""
        state_dict = {
            "layer1": torch.randn(10, 20),
            "layer2": torch.randn(20, 30),
        }

        mutated = mutate(state_dict, mutation_rate=1.0)

        assert mutated["layer1"].shape == state_dict["layer1"].shape
        assert mutated["layer2"].shape == state_dict["layer2"].shape


class TestGenetics:
    """Tests for Genetics class."""

    def test_genetics_initialization(self):
        """Test Genetics class initialization."""
        genetics = Genetics(
            crossover_rate=0.7,
            mutation_rate=0.1,
            adaptive_mutation=True,
        )

        assert genetics.crossover_rate == 0.7
        assert genetics.mutation_rate == 0.1
        assert genetics.generation == 0

    def test_genetics_create_offspring(self):
        """Test offspring creation."""
        genetics = Genetics()

        parent1 = {"weight": torch.randn(10)}
        parent2 = {"weight": torch.randn(10)}

        child = genetics.create_offspring(parent1, parent2)

        assert "weight" in child
        assert child["weight"].shape == parent1["weight"].shape

    def test_genetics_generation_step(self):
        """Test generation stepping."""
        genetics = Genetics(adaptive_mutation=True)

        initial_gen = genetics.generation
        genetics.step_generation()

        assert genetics.generation == initial_gen + 1

    def test_genetics_adaptive_mutation_decay(self):
        """Test that adaptive mutation decays correctly."""
        genetics = Genetics(
            mutation_rate=0.1,
            adaptive_mutation=True,
            mutation_decay=0.9,
            min_mutation_rate=0.01,
        )

        rate1 = genetics._get_current_mutation_rate()
        genetics.step_generation()
        rate2 = genetics._get_current_mutation_rate()

        assert rate2 < rate1

    def test_genetics_reset(self):
        """Test genetics reset."""
        genetics = Genetics()
        genetics.step_generation()
        genetics.step_generation()

        genetics.reset()

        assert genetics.generation == 0


class TestBlendStateDicts:
    """Tests for blend_state_dicts function."""

    def test_blend_two_dicts(self):
        """Test blending two state dicts."""
        sd1 = {"weight": torch.ones(5)}
        sd2 = {"weight": torch.zeros(5)}

        blended = blend_state_dicts([sd1, sd2])

        # Equal weights should give average
        expected = torch.full((5,), 0.5)
        assert torch.allclose(blended["weight"], expected)

    def test_blend_with_weights(self):
        """Test blending with custom weights."""
        sd1 = {"weight": torch.ones(5)}
        sd2 = {"weight": torch.zeros(5)}

        blended = blend_state_dicts([sd1, sd2], weights=[0.8, 0.2])

        expected = torch.full((5,), 0.8)
        assert torch.allclose(blended["weight"], expected)

    def test_blend_single_dict(self):
        """Test blending single dict returns copy."""
        sd1 = {"weight": torch.ones(5)}

        blended = blend_state_dicts([sd1])

        assert torch.allclose(blended["weight"], sd1["weight"])

    def test_blend_empty_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError):
            blend_state_dicts([])
