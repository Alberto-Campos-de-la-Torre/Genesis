"""Selection strategies for evolutionary optimization."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import logging

from genesis.core.population import Individual

logger = logging.getLogger(__name__)


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""

    @abstractmethod
    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        """
        Select individuals from the population.

        Args:
            population: List of individuals to select from
            num_to_select: Number of individuals to select

        Returns:
            List of selected individuals
        """
        pass

    def __call__(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        return self.select(population, num_to_select)


class ElitismSelection(SelectionStrategy):
    """
    Elitism selection: always select the top N individuals by fitness.

    This ensures the best solutions are preserved across generations.
    """

    def __init__(self, preserve_ratio: float = 0.1):
        """
        Args:
            preserve_ratio: Ratio of population to preserve as elites
        """
        self.preserve_ratio = preserve_ratio

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        # Sort by fitness descending
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Select top individuals
        return sorted_pop[:num_to_select]


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection: randomly sample k individuals and select the best.

    This provides a balance between selection pressure and diversity.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        selected = []

        for _ in range(num_to_select):
            # Randomly select tournament participants
            tournament_indices = np.random.choice(
                len(population),
                size=min(self.tournament_size, len(population)),
                replace=False,
            )
            tournament = [population[i] for i in tournament_indices]

            # Select winner (highest fitness)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected


class RouletteWheelSelection(SelectionStrategy):
    """
    Roulette wheel (fitness proportionate) selection.

    Probability of selection is proportional to fitness.
    """

    def __init__(self, scaling: str = "linear"):
        """
        Args:
            scaling: Fitness scaling method ('linear', 'rank', 'sigma')
        """
        self.scaling = scaling

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        fitnesses = np.array([ind.fitness for ind in population])

        # Handle negative fitness values
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min() + 1e-8

        # Apply scaling
        if self.scaling == "rank":
            # Rank-based scaling
            ranks = np.argsort(np.argsort(fitnesses)) + 1
            probabilities = ranks / ranks.sum()
        elif self.scaling == "sigma":
            # Sigma scaling
            mean = fitnesses.mean()
            std = fitnesses.std() + 1e-8
            scaled = 1 + (fitnesses - mean) / (2 * std)
            scaled = np.maximum(scaled, 0)
            probabilities = scaled / scaled.sum()
        else:
            # Linear scaling (default)
            probabilities = fitnesses / fitnesses.sum()

        # Select individuals
        indices = np.random.choice(
            len(population),
            size=num_to_select,
            replace=True,
            p=probabilities,
        )

        return [population[i] for i in indices]


class RankSelection(SelectionStrategy):
    """
    Rank-based selection: selection probability based on rank, not raw fitness.

    This reduces the dominance of very fit individuals.
    """

    def __init__(self, selection_pressure: float = 2.0):
        """
        Args:
            selection_pressure: Controls selection intensity (1.0 = uniform, 2.0 = linear)
        """
        self.selection_pressure = selection_pressure

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        n = len(population)

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Compute rank-based probabilities (linear ranking)
        sp = self.selection_pressure
        probabilities = np.array([
            (2 - sp + 2 * (sp - 1) * (n - 1 - i) / (n - 1)) / n
            for i in range(n)
        ])
        probabilities = probabilities / probabilities.sum()

        # Select individuals
        indices = np.random.choice(n, size=num_to_select, replace=True, p=probabilities)

        return [sorted_pop[i] for i in indices]


class TruncationSelection(SelectionStrategy):
    """
    Truncation selection: only the top fraction of the population reproduces.

    Simple but effective for strong selection pressure.
    """

    def __init__(self, truncation_rate: float = 0.5):
        """
        Args:
            truncation_rate: Fraction of population eligible for selection
        """
        self.truncation_rate = truncation_rate

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Truncate
        cutoff = max(1, int(len(sorted_pop) * self.truncation_rate))
        eligible = sorted_pop[:cutoff]

        # Randomly select from eligible
        indices = np.random.choice(len(eligible), size=num_to_select, replace=True)

        return [eligible[i] for i in indices]


class BoltzmannSelection(SelectionStrategy):
    """
    Boltzmann selection: uses temperature-controlled softmax probabilities.

    High temperature = more exploration, low temperature = more exploitation.
    """

    def __init__(
        self,
        initial_temperature: float = 10.0,
        min_temperature: float = 0.1,
        cooling_rate: float = 0.95,
    ):
        """
        Args:
            initial_temperature: Starting temperature
            min_temperature: Minimum temperature
            cooling_rate: Temperature decay per generation
        """
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.cooling_rate = cooling_rate

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        fitnesses = np.array([ind.fitness for ind in population])

        # Normalize fitness for numerical stability
        fitnesses_norm = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

        # Boltzmann probabilities
        exp_values = np.exp(fitnesses_norm / self.temperature)
        probabilities = exp_values / exp_values.sum()

        # Select individuals
        indices = np.random.choice(
            len(population),
            size=num_to_select,
            replace=True,
            p=probabilities,
        )

        return [population[i] for i in indices]

    def cool_down(self) -> None:
        """Reduce temperature for next generation."""
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.cooling_rate,
        )


class SteadyStateSelection(SelectionStrategy):
    """
    Steady-state selection: replace only a few individuals each generation.

    Maintains population diversity by making gradual changes.
    """

    def __init__(
        self,
        replacement_rate: float = 0.2,
        parent_selection: Optional[SelectionStrategy] = None,
    ):
        """
        Args:
            replacement_rate: Fraction of population to replace
            parent_selection: Strategy for selecting parents
        """
        self.replacement_rate = replacement_rate
        self.parent_selection = parent_selection or TournamentSelection()

    def select(
        self,
        population: list[Individual],
        num_to_select: int,
    ) -> list[Individual]:
        # Determine how many to replace
        num_to_replace = max(1, int(len(population) * self.replacement_rate))

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Keep the best individuals
        num_to_keep = len(population) - num_to_replace
        survivors = sorted_pop[:num_to_keep]

        # Select parents for new offspring
        parents = self.parent_selection.select(sorted_pop, num_to_replace)

        return survivors + parents


def create_selection_strategy(
    strategy_type: str,
    **kwargs,
) -> SelectionStrategy:
    """
    Factory function to create selection strategies.

    Args:
        strategy_type: Type of selection strategy
        **kwargs: Additional arguments for the strategy

    Returns:
        SelectionStrategy instance
    """
    strategies = {
        "elitism": ElitismSelection,
        "tournament": TournamentSelection,
        "roulette": RouletteWheelSelection,
        "rank": RankSelection,
        "truncation": TruncationSelection,
        "boltzmann": BoltzmannSelection,
        "steady_state": SteadyStateSelection,
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown selection strategy: {strategy_type}")

    return strategies[strategy_type](**kwargs)
