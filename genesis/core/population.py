"""Population management for evolutionary optimization."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import uuid
import logging

from genesis.core.genetics import Genetics, crossover, mutate

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents an individual in the population."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    fitness: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

    def clone(self) -> "Individual":
        """Create a deep copy of this individual."""
        return Individual(
            id=str(uuid.uuid4())[:8],
            state_dict=deepcopy(self.state_dict),
            fitness=self.fitness,
            metadata=deepcopy(self.metadata),
            generation=self.generation,
            parent_ids=[self.id],
        )

    def __repr__(self) -> str:
        return f"Individual(id={self.id}, fitness={self.fitness:.4f}, gen={self.generation})"


class Population:
    """Manages a population of individuals for evolutionary optimization."""

    def __init__(
        self,
        size: int,
        genetics: Optional[Genetics] = None,
        elite_size: int = 2,
    ):
        """
        Initialize population.

        Args:
            size: Population size
            genetics: Genetics instance for genetic operations
            elite_size: Number of elite individuals to preserve
        """
        self.size = size
        self.genetics = genetics or Genetics()
        self.elite_size = elite_size
        self._individuals: list[Individual] = []
        self._generation = 0
        self._best_fitness_history: list[float] = []
        self._avg_fitness_history: list[float] = []

    def initialize_from_model(
        self,
        model: nn.Module,
        perturbation_scale: float = 0.01,
    ) -> None:
        """
        Initialize population from a base model.

        Args:
            model: Base model to create population from
            perturbation_scale: Scale of random perturbations
        """
        base_state = deepcopy(model.state_dict())
        self._individuals = []

        for i in range(self.size):
            if i == 0:
                # First individual is unperturbed base model
                individual = Individual(
                    state_dict=deepcopy(base_state),
                    generation=0,
                    metadata={"origin": "base"},
                )
            else:
                # Add random perturbations
                perturbed_state = mutate(
                    base_state,
                    mutation_rate=1.0,  # Always mutate for initialization
                    mutation_scale=perturbation_scale,
                    mutation_prob_per_weight=0.5,
                )
                individual = Individual(
                    state_dict=perturbed_state,
                    generation=0,
                    metadata={"origin": "perturbed"},
                )
            self._individuals.append(individual)

        logger.info(f"Initialized population with {self.size} individuals")

    def initialize_from_state_dicts(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
    ) -> None:
        """
        Initialize population from existing state dictionaries.

        Args:
            state_dicts: List of state dictionaries
        """
        self._individuals = []
        for i, state_dict in enumerate(state_dicts):
            individual = Individual(
                state_dict=deepcopy(state_dict),
                generation=0,
                metadata={"origin": f"provided_{i}"},
            )
            self._individuals.append(individual)

        # Fill remaining slots with mutations if needed
        while len(self._individuals) < self.size:
            base = np.random.choice(self._individuals)
            mutated_state = mutate(
                base.state_dict,
                mutation_rate=1.0,
                mutation_scale=0.01,
            )
            individual = Individual(
                state_dict=mutated_state,
                generation=0,
                parent_ids=[base.id],
                metadata={"origin": "mutation"},
            )
            self._individuals.append(individual)

    def evaluate(
        self,
        fitness_fn: Callable[[dict[str, torch.Tensor]], float],
    ) -> None:
        """
        Evaluate fitness of all individuals.

        Args:
            fitness_fn: Function that takes state_dict and returns fitness score
        """
        for individual in self._individuals:
            individual.fitness = fitness_fn(individual.state_dict)

        # Sort by fitness (descending)
        self._individuals.sort(key=lambda x: x.fitness, reverse=True)

        # Update history
        self._best_fitness_history.append(self.best.fitness)
        self._avg_fitness_history.append(self.average_fitness)

        logger.info(
            f"Generation {self._generation}: "
            f"Best={self.best.fitness:.4f}, "
            f"Avg={self.average_fitness:.4f}"
        )

    def evolve(self) -> None:
        """
        Create next generation through selection, crossover, and mutation.
        """
        new_individuals = []

        # Elitism: preserve top individuals
        elites = self._individuals[: self.elite_size]
        for elite in elites:
            elite_copy = elite.clone()
            elite_copy.generation = self._generation + 1
            elite_copy.metadata["origin"] = "elite"
            new_individuals.append(elite_copy)

        # Create offspring for remaining slots
        while len(new_individuals) < self.size:
            # Tournament selection for parents
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Create offspring
            child_state = self.genetics.create_offspring(
                parent1.state_dict,
                parent2.state_dict,
            )

            child = Individual(
                state_dict=child_state,
                generation=self._generation + 1,
                parent_ids=[parent1.id, parent2.id],
                metadata={"origin": "offspring"},
            )
            new_individuals.append(child)

        self._individuals = new_individuals
        self._generation += 1
        self.genetics.step_generation()

    def _tournament_select(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        actual_size = min(tournament_size, len(self._individuals))
        tournament = np.random.choice(self._individuals, size=actual_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)

    @property
    def individuals(self) -> list[Individual]:
        """Return all individuals."""
        return self._individuals

    @property
    def best(self) -> Individual:
        """Return best individual by fitness."""
        return max(self._individuals, key=lambda x: x.fitness)

    @property
    def worst(self) -> Individual:
        """Return worst individual by fitness."""
        return min(self._individuals, key=lambda x: x.fitness)

    @property
    def average_fitness(self) -> float:
        """Return average fitness of population."""
        return sum(ind.fitness for ind in self._individuals) / len(self._individuals)

    @property
    def fitness_std(self) -> float:
        """Return standard deviation of fitness."""
        fitnesses = [ind.fitness for ind in self._individuals]
        return float(np.std(fitnesses))

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    @property
    def diversity(self) -> float:
        """
        Compute population diversity based on parameter variance.

        Returns:
            Average standard deviation across all parameters
        """
        if len(self._individuals) < 2:
            return 0.0

        keys = list(self._individuals[0].state_dict.keys())
        total_std = 0.0
        count = 0

        for key in keys:
            params = torch.stack([ind.state_dict[key].float() for ind in self._individuals])
            total_std += params.std(dim=0).mean().item()
            count += 1

        return total_std / count if count > 0 else 0.0

    def get_state(self) -> dict[str, Any]:
        """Get population state for checkpointing."""
        return {
            "generation": self._generation,
            "size": self.size,
            "elite_size": self.elite_size,
            "individuals": [
                {
                    "id": ind.id,
                    "state_dict": ind.state_dict,
                    "fitness": ind.fitness,
                    "metadata": ind.metadata,
                    "generation": ind.generation,
                    "parent_ids": ind.parent_ids,
                }
                for ind in self._individuals
            ],
            "best_fitness_history": self._best_fitness_history,
            "avg_fitness_history": self._avg_fitness_history,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load population state from checkpoint."""
        self._generation = state["generation"]
        self.size = state["size"]
        self.elite_size = state["elite_size"]
        self._best_fitness_history = state["best_fitness_history"]
        self._avg_fitness_history = state["avg_fitness_history"]

        self._individuals = []
        for ind_state in state["individuals"]:
            individual = Individual(
                id=ind_state["id"],
                state_dict=ind_state["state_dict"],
                fitness=ind_state["fitness"],
                metadata=ind_state["metadata"],
                generation=ind_state["generation"],
                parent_ids=ind_state["parent_ids"],
            )
            self._individuals.append(individual)

    def __len__(self) -> int:
        return len(self._individuals)

    def __getitem__(self, idx: int) -> Individual:
        return self._individuals[idx]

    def __iter__(self):
        return iter(self._individuals)
