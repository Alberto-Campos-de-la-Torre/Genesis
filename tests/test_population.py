"""Tests for population module."""

import pytest
import torch
import torch.nn as nn

from genesis.core.population import Population, Individual
from genesis.core.genetics import Genetics


class TestIndividual:
    """Tests for Individual class."""

    def test_individual_creation(self):
        """Test individual creation."""
        state_dict = {"weight": torch.randn(10)}
        individual = Individual(state_dict=state_dict, fitness=0.5, generation=1)

        assert individual.fitness == 0.5
        assert individual.generation == 1
        assert "weight" in individual.state_dict

    def test_individual_clone(self):
        """Test individual cloning."""
        state_dict = {"weight": torch.randn(10)}
        individual = Individual(state_dict=state_dict, fitness=0.8)

        clone = individual.clone()

        assert clone.id != individual.id
        assert clone.fitness == individual.fitness
        assert torch.allclose(clone.state_dict["weight"], individual.state_dict["weight"])

    def test_individual_clone_independence(self):
        """Test that cloned individuals are independent."""
        state_dict = {"weight": torch.randn(10)}
        individual = Individual(state_dict=state_dict)

        clone = individual.clone()
        clone.state_dict["weight"][0] = 999.0

        assert individual.state_dict["weight"][0] != 999.0

    def test_individual_repr(self):
        """Test individual string representation."""
        individual = Individual(fitness=0.75, generation=5)
        repr_str = repr(individual)

        assert "0.75" in repr_str
        assert "5" in repr_str


class TestPopulation:
    """Tests for Population class."""

    def test_population_initialization(self):
        """Test population initialization."""
        population = Population(size=10, elite_size=2)

        assert population.size == 10
        assert population.elite_size == 2
        assert len(population) == 0  # Not initialized yet

    def test_initialize_from_model(self):
        """Test population initialization from model."""
        model = nn.Linear(10, 5)
        population = Population(size=5)

        population.initialize_from_model(model, perturbation_scale=0.01)

        assert len(population) == 5
        # First individual should be unperturbed
        assert population[0].metadata.get("origin") == "base"

    def test_population_evaluation(self):
        """Test population fitness evaluation."""
        population = Population(size=5)
        population.initialize_from_model(nn.Linear(5, 5))

        # Simple fitness function
        def fitness_fn(state_dict):
            return torch.mean(state_dict["weight"]).item()

        population.evaluate(fitness_fn)

        # All individuals should have fitness set
        for ind in population:
            assert ind.fitness != 0.0

    def test_population_best(self):
        """Test getting best individual."""
        population = Population(size=3)
        population._individuals = [
            Individual(state_dict={}, fitness=0.5),
            Individual(state_dict={}, fitness=0.9),
            Individual(state_dict={}, fitness=0.3),
        ]

        best = population.best
        assert best.fitness == 0.9

    def test_population_worst(self):
        """Test getting worst individual."""
        population = Population(size=3)
        population._individuals = [
            Individual(state_dict={}, fitness=0.5),
            Individual(state_dict={}, fitness=0.9),
            Individual(state_dict={}, fitness=0.3),
        ]

        worst = population.worst
        assert worst.fitness == 0.3

    def test_population_average_fitness(self):
        """Test average fitness calculation."""
        population = Population(size=3)
        population._individuals = [
            Individual(state_dict={}, fitness=0.3),
            Individual(state_dict={}, fitness=0.6),
            Individual(state_dict={}, fitness=0.9),
        ]

        avg = population.average_fitness
        assert abs(avg - 0.6) < 1e-6

    def test_population_evolution(self):
        """Test population evolution."""
        model = nn.Linear(5, 5)
        population = Population(size=5, elite_size=1)
        population.initialize_from_model(model)

        # Set some fitnesses
        for i, ind in enumerate(population):
            ind.fitness = float(i) / len(population)

        initial_gen = population.generation
        population.evolve()

        assert population.generation == initial_gen + 1
        assert len(population) == 5

    def test_population_elitism(self):
        """Test that elites are preserved."""
        population = Population(size=5, elite_size=2)

        # Create individuals with known fitnesses
        population._individuals = [
            Individual(id="best", state_dict={"w": torch.ones(5)}, fitness=1.0),
            Individual(id="second", state_dict={"w": torch.ones(5) * 2}, fitness=0.8),
            Individual(id="third", state_dict={"w": torch.zeros(5)}, fitness=0.5),
            Individual(id="fourth", state_dict={"w": torch.zeros(5)}, fitness=0.3),
            Individual(id="fifth", state_dict={"w": torch.zeros(5)}, fitness=0.1),
        ]

        population.evolve()

        # Best individuals should have elite origin
        elite_count = sum(1 for ind in population if ind.metadata.get("origin") == "elite")
        assert elite_count >= 1

    def test_population_state_save_load(self):
        """Test saving and loading population state."""
        model = nn.Linear(5, 5)
        population = Population(size=3)
        population.initialize_from_model(model)

        # Set some state
        population._individuals[0].fitness = 0.99
        population._generation = 5

        # Save state
        state = population.get_state()

        # Create new population and load
        new_population = Population(size=3)
        new_population.load_state(state)

        assert new_population.generation == 5
        assert new_population[0].fitness == 0.99

    def test_population_iteration(self):
        """Test population iteration."""
        population = Population(size=3)
        population._individuals = [
            Individual(state_dict={}, fitness=i)
            for i in range(3)
        ]

        fitnesses = [ind.fitness for ind in population]
        assert fitnesses == [0.0, 1.0, 2.0]

    def test_population_indexing(self):
        """Test population indexing."""
        population = Population(size=3)
        population._individuals = [
            Individual(id=f"ind_{i}", state_dict={})
            for i in range(3)
        ]

        assert population[0].id == "ind_0"
        assert population[2].id == "ind_2"


class TestPopulationDiversity:
    """Tests for population diversity measurement."""

    def test_diversity_calculation(self):
        """Test diversity calculation."""
        population = Population(size=3)

        # Create individuals with different weights
        population._individuals = [
            Individual(state_dict={"w": torch.zeros(10)}),
            Individual(state_dict={"w": torch.ones(10)}),
            Individual(state_dict={"w": torch.ones(10) * 2}),
        ]

        diversity = population.diversity
        assert diversity > 0

    def test_diversity_zero_for_identical(self):
        """Test diversity is zero for identical individuals."""
        population = Population(size=3)
        same_weight = torch.ones(10)

        population._individuals = [
            Individual(state_dict={"w": same_weight.clone()})
            for _ in range(3)
        ]

        diversity = population.diversity
        assert diversity < 1e-6

    def test_diversity_with_single_individual(self):
        """Test diversity with single individual."""
        population = Population(size=1)
        population._individuals = [Individual(state_dict={"w": torch.ones(10)})]

        diversity = population.diversity
        assert diversity == 0.0
