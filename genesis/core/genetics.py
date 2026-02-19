"""Genetic operations for evolutionary optimization."""

from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two tensors.

    SLERP interpolates along the shortest arc on a hypersphere,
    making it ideal for blending model weights.

    Args:
        t: Interpolation factor (0 = v0, 1 = v1)
        v0: First tensor
        v1: Second tensor
        epsilon: Small value to prevent division by zero

    Returns:
        Interpolated tensor
    """
    # Flatten tensors
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    # Normalize
    v0_norm = v0_flat / (torch.norm(v0_flat) + epsilon)
    v1_norm = v1_flat / (torch.norm(v1_flat) + epsilon)

    # Compute angle between vectors
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)

    # Handle edge cases
    if theta.abs() < epsilon:
        # Vectors are nearly parallel, use linear interpolation
        result = (1 - t) * v0_flat + t * v1_flat
    else:
        sin_theta = torch.sin(theta)
        result = (
            torch.sin((1 - t) * theta) / sin_theta * v0_flat
            + torch.sin(t * theta) / sin_theta * v1_flat
        )

    # Preserve original magnitude (average of both)
    original_magnitude = (torch.norm(v0_flat) + torch.norm(v1_flat)) / 2
    result = result / (torch.norm(result) + epsilon) * original_magnitude

    return result.view(v0.shape).to(v0.dtype)


def crossover(
    parent1_state: dict[str, torch.Tensor],
    parent2_state: dict[str, torch.Tensor],
    crossover_rate: float = 0.7,
    method: str = "slerp",
    slerp_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Perform crossover between two parent state dictionaries.

    Args:
        parent1_state: State dict of first parent
        parent2_state: State dict of second parent
        crossover_rate: Probability of crossover occurring
        method: Crossover method ('slerp', 'uniform', 'single_point')
        slerp_ratio: Interpolation ratio for SLERP (0.5 = equal blend)

    Returns:
        Child state dictionary
    """
    child_state = {}

    # Check if crossover should occur
    if np.random.random() > crossover_rate:
        # No crossover, return copy of first parent
        return deepcopy(parent1_state)

    keys = list(parent1_state.keys())

    if method == "slerp":
        # SLERP-based crossover
        for key in keys:
            if key in parent2_state:
                p1 = parent1_state[key]
                p2 = parent2_state[key]
                if p1.shape == p2.shape:
                    # Add small random variation to slerp_ratio
                    t = slerp_ratio + np.random.uniform(-0.1, 0.1)
                    t = np.clip(t, 0.0, 1.0)
                    child_state[key] = slerp(t, p1, p2)
                else:
                    child_state[key] = p1.clone()
            else:
                child_state[key] = parent1_state[key].clone()

    elif method == "uniform":
        # Uniform crossover: randomly select each parameter from either parent
        for key in keys:
            if key in parent2_state and np.random.random() < 0.5:
                child_state[key] = parent2_state[key].clone()
            else:
                child_state[key] = parent1_state[key].clone()

    elif method == "single_point":
        # Single-point crossover: split at random point
        crossover_point = np.random.randint(0, len(keys))
        for i, key in enumerate(keys):
            if i < crossover_point:
                child_state[key] = parent1_state[key].clone()
            else:
                if key in parent2_state:
                    child_state[key] = parent2_state[key].clone()
                else:
                    child_state[key] = parent1_state[key].clone()

    else:
        raise ValueError(f"Unknown crossover method: {method}")

    return child_state


def mutate(
    state_dict: dict[str, torch.Tensor],
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.01,
    mutation_prob_per_weight: float = 0.1,
    method: str = "gaussian",
) -> dict[str, torch.Tensor]:
    """
    Apply mutation to a state dictionary.

    Args:
        state_dict: Model state dictionary
        mutation_rate: Overall probability of mutation occurring
        mutation_scale: Scale of Gaussian noise
        mutation_prob_per_weight: Probability of mutating each weight
        method: Mutation method ('gaussian', 'uniform', 'adaptive')

    Returns:
        Mutated state dictionary
    """
    mutated_state = {}

    # Check if mutation should occur at all
    if np.random.random() > mutation_rate:
        return deepcopy(state_dict)

    _float_dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    for key, param in state_dict.items():
        if param.dtype in _float_dtypes:
            if method == "gaussian":
                # Create mutation mask
                mask = torch.rand_like(param.float()) < mutation_prob_per_weight
                noise = torch.randn_like(param.float()) * mutation_scale

                # Apply mutation only where mask is True
                mutated_param = param.float() + noise * mask.float()
                mutated_state[key] = mutated_param.to(param.dtype)

            elif method == "uniform":
                mask = torch.rand_like(param.float()) < mutation_prob_per_weight
                noise = (torch.rand_like(param.float()) - 0.5) * 2 * mutation_scale
                mutated_param = param.float() + noise * mask.float()
                mutated_state[key] = mutated_param.to(param.dtype)

            elif method == "adaptive":
                # Scale mutation by parameter magnitude
                param_scale = torch.abs(param.float()).mean() + 1e-8
                mask = torch.rand_like(param.float()) < mutation_prob_per_weight
                noise = torch.randn_like(param.float()) * mutation_scale * param_scale
                mutated_param = param.float() + noise * mask.float()
                mutated_state[key] = mutated_param.to(param.dtype)

            else:
                raise ValueError(f"Unknown mutation method: {method}")
        else:
            mutated_state[key] = param.clone()

    return mutated_state


class Genetics:
    """Manager class for genetic operations."""

    def __init__(
        self,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.01,
        slerp_ratio: float = 0.5,
        crossover_method: str = "slerp",
        mutation_method: str = "gaussian",
        adaptive_mutation: bool = True,
        mutation_decay: float = 0.95,
        min_mutation_rate: float = 0.01,
    ):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.slerp_ratio = slerp_ratio
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.adaptive_mutation = adaptive_mutation
        self.mutation_decay = mutation_decay
        self.min_mutation_rate = min_mutation_rate

        self._generation = 0

    def create_offspring(
        self,
        parent1: Union[nn.Module, dict[str, torch.Tensor]],
        parent2: Union[nn.Module, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Create offspring from two parents using crossover and mutation.

        Args:
            parent1: First parent model or state dict
            parent2: Second parent model or state dict

        Returns:
            Child state dictionary
        """
        # Extract state dicts if needed
        p1_state = parent1.state_dict() if isinstance(parent1, nn.Module) else parent1
        p2_state = parent2.state_dict() if isinstance(parent2, nn.Module) else parent2

        # Perform crossover
        child_state = crossover(
            p1_state,
            p2_state,
            crossover_rate=self.crossover_rate,
            method=self.crossover_method,
            slerp_ratio=self.slerp_ratio,
        )

        # Apply mutation
        current_mutation_rate = self._get_current_mutation_rate()
        child_state = mutate(
            child_state,
            mutation_rate=current_mutation_rate,
            mutation_scale=self.mutation_scale,
            method=self.mutation_method,
        )

        return child_state

    def _get_current_mutation_rate(self) -> float:
        """Get mutation rate for current generation (with decay)."""
        if not self.adaptive_mutation:
            return self.mutation_rate

        decayed_rate = self.mutation_rate * (self.mutation_decay**self._generation)
        return max(decayed_rate, self.min_mutation_rate)

    def step_generation(self) -> None:
        """Advance to next generation (for adaptive mutation)."""
        self._generation += 1

    def reset(self) -> None:
        """Reset genetics state."""
        self._generation = 0

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation


def blend_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: Optional[list[float]] = None,
) -> dict[str, torch.Tensor]:
    """
    Blend multiple state dictionaries with optional weights.

    Args:
        state_dicts: List of state dictionaries to blend
        weights: Optional weights for each state dict (must sum to 1)

    Returns:
        Blended state dictionary
    """
    if len(state_dicts) == 0:
        raise ValueError("At least one state dict required")

    if len(state_dicts) == 1:
        return deepcopy(state_dicts[0])

    # Default to equal weights
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    assert len(weights) == len(state_dicts), "Weights must match number of state dicts"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    blended_state = {}
    keys = list(state_dicts[0].keys())

    for key in keys:
        tensors = [sd[key] for sd in state_dicts if key in sd]
        if len(tensors) == len(state_dicts):
            blended = sum(w * t.float() for w, t in zip(weights, tensors))
            blended_state[key] = blended.to(state_dicts[0][key].dtype)
        else:
            blended_state[key] = state_dicts[0][key].clone()

    return blended_state
