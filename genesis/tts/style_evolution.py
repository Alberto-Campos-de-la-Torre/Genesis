"""Style token evolution for TTS models."""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


@dataclass
class StyleToken:
    """Represents a learnable style token."""

    embedding: torch.Tensor
    name: str = ""
    fitness: float = 0.0
    metadata: dict = field(default_factory=dict)

    def clone(self) -> "StyleToken":
        """Create a deep copy."""
        return StyleToken(
            embedding=self.embedding.clone(),
            name=self.name,
            fitness=self.fitness,
            metadata=deepcopy(self.metadata),
        )


class StyleEvolution:
    """
    Evolutionary optimization of TTS style tokens.

    Evolves style embeddings to optimize for specific voice
    characteristics like expressiveness, naturalness, etc.
    """

    def __init__(
        self,
        style_dim: int = 128,
        num_tokens: int = 10,
        population_size: int = 20,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.05,
        crossover_rate: float = 0.7,
    ):
        """
        Initialize style evolution.

        Args:
            style_dim: Dimension of style embeddings
            num_tokens: Number of style tokens per individual
            population_size: Population size
            elite_size: Number of elites to preserve
            mutation_rate: Probability of mutation
            mutation_scale: Scale of mutation noise
            crossover_rate: Probability of crossover
        """
        self.style_dim = style_dim
        self.num_tokens = num_tokens
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.crossover_rate = crossover_rate

        self._population: list[torch.Tensor] = []
        self._fitnesses: list[float] = []
        self._generation = 0

    def initialize_population(
        self,
        base_tokens: Optional[torch.Tensor] = None,
        perturbation_scale: float = 0.1,
    ) -> None:
        """
        Initialize the population of style tokens.

        Args:
            base_tokens: Optional base tokens to initialize from
            perturbation_scale: Scale of initial perturbations
        """
        self._population = []
        self._fitnesses = []

        if base_tokens is not None:
            # First individual is the base tokens
            self._population.append(base_tokens.clone())

            # Rest are perturbed versions
            for _ in range(self.population_size - 1):
                perturbed = base_tokens + torch.randn_like(base_tokens) * perturbation_scale
                self._population.append(perturbed)
        else:
            # Random initialization
            for _ in range(self.population_size):
                tokens = torch.randn(self.num_tokens, self.style_dim)
                # Normalize to unit sphere
                tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-8)
                self._population.append(tokens)

        self._fitnesses = [0.0] * self.population_size
        logger.info(f"Initialized population with {self.population_size} individuals")

    def set_fitness(self, index: int, fitness: float) -> None:
        """Set fitness for individual at index."""
        if 0 <= index < len(self._fitnesses):
            self._fitnesses[index] = fitness

    def set_all_fitnesses(self, fitnesses: list[float]) -> None:
        """Set all fitness values."""
        assert len(fitnesses) == len(self._population)
        self._fitnesses = list(fitnesses)

    def evolve(self) -> None:
        """Perform one generation of evolution."""
        # Sort by fitness
        sorted_indices = np.argsort(self._fitnesses)[::-1]
        sorted_pop = [self._population[i] for i in sorted_indices]
        sorted_fit = [self._fitnesses[i] for i in sorted_indices]

        new_population = []

        # Elitism: keep top individuals
        for i in range(self.elite_size):
            new_population.append(sorted_pop[i].clone())

        # Generate rest through selection, crossover, mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1_idx = self._tournament_select(sorted_fit)
            parent2_idx = self._tournament_select(sorted_fit)

            parent1 = sorted_pop[parent1_idx]
            parent2 = sorted_pop[parent2_idx]

            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.clone()

            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        self._population = new_population
        self._fitnesses = [0.0] * self.population_size
        self._generation += 1

        logger.info(f"Generation {self._generation}: Best fitness = {sorted_fit[0]:.4f}")

    def _tournament_select(
        self,
        fitnesses: list[float],
        tournament_size: int = 3,
    ) -> int:
        """Tournament selection."""
        indices = np.random.choice(len(fitnesses), size=tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitnesses)]
        return winner_idx

    def _crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> torch.Tensor:
        """Perform crossover between two parents."""
        # SLERP-style interpolation
        ratio = np.random.uniform(0.3, 0.7)

        # Normalize parents
        p1_norm = parent1 / (parent1.norm(dim=-1, keepdim=True) + 1e-8)
        p2_norm = parent2 / (parent2.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute angle between vectors
        dot = (p1_norm * p2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
        theta = torch.acos(dot)

        # SLERP
        sin_theta = torch.sin(theta) + 1e-8
        child = (
            torch.sin((1 - ratio) * theta) / sin_theta * parent1
            + torch.sin(ratio * theta) / sin_theta * parent2
        )

        return child

    def _mutate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply mutation to style tokens."""
        # Create mutation mask
        mask = torch.rand_like(tokens) < 0.2  # Mutate 20% of elements
        noise = torch.randn_like(tokens) * self.mutation_scale

        mutated = tokens + noise * mask.float()

        # Optionally renormalize
        # mutated = mutated / (mutated.norm(dim=-1, keepdim=True) + 1e-8)

        return mutated

    def get_best(self) -> tuple[torch.Tensor, float]:
        """Get best individual and its fitness."""
        best_idx = np.argmax(self._fitnesses)
        return self._population[best_idx], self._fitnesses[best_idx]

    def get_population(self) -> list[torch.Tensor]:
        """Get entire population."""
        return self._population

    def get_individual(self, index: int) -> torch.Tensor:
        """Get individual at index."""
        return self._population[index]

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    @property
    def best_fitness(self) -> float:
        """Best fitness in current population."""
        return max(self._fitnesses) if self._fitnesses else 0.0

    @property
    def average_fitness(self) -> float:
        """Average fitness of population."""
        return sum(self._fitnesses) / len(self._fitnesses) if self._fitnesses else 0.0

    def save_state(self, path: str) -> None:
        """Save evolution state to file."""
        state = {
            "generation": self._generation,
            "population": self._population,
            "fitnesses": self._fitnesses,
            "config": {
                "style_dim": self.style_dim,
                "num_tokens": self.num_tokens,
                "population_size": self.population_size,
                "elite_size": self.elite_size,
                "mutation_rate": self.mutation_rate,
                "mutation_scale": self.mutation_scale,
                "crossover_rate": self.crossover_rate,
            },
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load evolution state from file."""
        state = torch.load(path)
        self._generation = state["generation"]
        self._population = state["population"]
        self._fitnesses = state["fitnesses"]

        config = state["config"]
        self.style_dim = config["style_dim"]
        self.num_tokens = config["num_tokens"]
        self.population_size = config["population_size"]
        self.elite_size = config["elite_size"]
        self.mutation_rate = config["mutation_rate"]
        self.mutation_scale = config["mutation_scale"]
        self.crossover_rate = config["crossover_rate"]


class MultiStyleEvolution:
    """
    Evolve multiple style aspects simultaneously.

    Handles prosody, emotion, speaker characteristics, etc.
    """

    def __init__(
        self,
        style_configs: dict[str, dict],
        population_size: int = 20,
    ):
        """
        Initialize multi-style evolution.

        Args:
            style_configs: Dictionary mapping style name to config
            population_size: Population size
        """
        self.evolutions: dict[str, StyleEvolution] = {}

        for name, config in style_configs.items():
            self.evolutions[name] = StyleEvolution(
                style_dim=config.get("dim", 128),
                num_tokens=config.get("num_tokens", 10),
                population_size=population_size,
                mutation_rate=config.get("mutation_rate", 0.1),
                mutation_scale=config.get("mutation_scale", 0.05),
            )

    def initialize_all(self, base_styles: Optional[dict[str, torch.Tensor]] = None) -> None:
        """Initialize all style evolutions."""
        for name, evolution in self.evolutions.items():
            base = base_styles.get(name) if base_styles else None
            evolution.initialize_population(base)

    def evolve_all(self) -> None:
        """Evolve all style populations."""
        for evolution in self.evolutions.values():
            evolution.evolve()

    def get_combined_style(self, indices: dict[str, int]) -> dict[str, torch.Tensor]:
        """Get combined style from multiple evolutions."""
        result = {}
        for name, idx in indices.items():
            if name in self.evolutions:
                result[name] = self.evolutions[name].get_individual(idx)
        return result
