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

    # Handle both nearly-identical (dot → +1) and antiparallel (dot → -1) vectors.
    # In both cases sin(theta) → 0, causing division by zero in the SLERP formula.
    # Fall back to LERP, which is the correct limit in both situations.
    if torch.abs(dot) > 0.9995:
        result = (1.0 - t) * v0_flat + t * v1_flat
    else:
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        result = (
            torch.sin((1.0 - t) * theta) / sin_theta * v0_flat
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
    ties_density: float = 0.2,
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

    if method == "ties":
        return ties_crossover(parent1_state, parent2_state, density=ties_density)

    elif method == "slerp":
        # SLERP-based crossover — performed on CPU to avoid GPU OOM
        for key in keys:
            if key in parent2_state:
                p1 = parent1_state[key].cpu()
                p2 = parent2_state[key].cpu()
                if p1.shape == p2.shape:
                    # Add small random variation to slerp_ratio
                    t = slerp_ratio + np.random.uniform(-0.1, 0.1)
                    t = np.clip(t, 0.0, 1.0)
                    child_state[key] = slerp(t, p1, p2)
                else:
                    child_state[key] = p1.clone()
            else:
                child_state[key] = parent1_state[key].cpu().clone()

    elif method == "uniform":
        # Uniform crossover: randomly select each parameter from either parent
        for key in keys:
            if key in parent2_state and np.random.random() < 0.5:
                child_state[key] = parent2_state[key].cpu().clone()
            else:
                child_state[key] = parent1_state[key].cpu().clone()

    elif method == "single_point":
        # Single-point crossover: split at random point
        crossover_point = np.random.randint(0, len(keys))
        for i, key in enumerate(keys):
            if i < crossover_point:
                child_state[key] = parent1_state[key].cpu().clone()
            else:
                if key in parent2_state:
                    child_state[key] = parent2_state[key].cpu().clone()
                else:
                    child_state[key] = parent1_state[key].cpu().clone()

    else:
        raise ValueError(f"Unknown crossover method: {method} (choices: ties, slerp, uniform, single_point)")

    return child_state


def ties_crossover(
    parent1_state: dict[str, torch.Tensor],
    parent2_state: dict[str, torch.Tensor],
    density: float = 0.2,
) -> dict[str, torch.Tensor]:
    """
    TIES-Merging crossover (Yadav et al., 2023).

    Plain weight averaging destroys knowledge because parameters from different
    fine-tunes often have conflicting signs (sign interference).  TIES resolves
    this in three steps for every tensor:

      1. TRIM   — zero out the (1-density) fraction of weights with the
                  smallest absolute value, keeping only the most salient ones.
      2. ELECT  — decide the dominant sign per element by majority vote
                  (sign of the trimmed sum).
      3. MERGE  — average only the weights that agree with the elected sign;
                  weights that conflict are treated as zero.

    Args:
        parent1_state: LoRA state dict of the first parent (CPU tensors).
        parent2_state: LoRA state dict of the second parent (CPU tensors).
        density: Fraction of weights to keep after trimming (0.2 = top 20%).

    Returns:
        Child state dict (CPU tensors, same dtypes as parent1).
    """
    child_state: dict[str, torch.Tensor] = {}

    for key in parent1_state:
        p1 = parent1_state[key].cpu()
        if key not in parent2_state:
            child_state[key] = p1.clone()
            continue

        p2 = parent2_state[key].cpu()
        if p1.shape != p2.shape:
            child_state[key] = p1.clone()
            continue

        orig_dtype = p1.dtype
        p1f, p2f = p1.float(), p2.float()

        # ── 1. TRIM ──────────────────────────────────────────────────────────
        # Keep top-`density` fraction by absolute magnitude for each parent.
        n = p1f.numel()
        k1 = max(1, int(n * density))
        k2 = max(1, int(n * density))

        flat1, flat2 = p1f.flatten(), p2f.flatten()
        # kthvalue gives the k-th *smallest*, so numel-k+1 gives the k-th largest.
        thresh1 = torch.kthvalue(flat1.abs(), n - k1 + 1).values
        thresh2 = torch.kthvalue(flat2.abs(), n - k2 + 1).values

        p1_trim = torch.where(p1f.abs() >= thresh1, p1f, torch.zeros_like(p1f))
        p2_trim = torch.where(p2f.abs() >= thresh2, p2f, torch.zeros_like(p2f))

        # ── 2. ELECT SIGN ────────────────────────────────────────────────────
        # The elected sign is the sign of the element-wise sum of trimmed weights.
        # sign(0) = 0, but we need a non-zero direction; use p1_trim as tiebreak.
        elected = torch.sign(p1_trim + p2_trim)
        # Where the sum is exactly zero, fall back to parent-1's sign.
        tiebreak = torch.sign(p1_trim)
        elected = torch.where(elected == 0, tiebreak, elected)

        # ── 3. DISJOINT MERGE ────────────────────────────────────────────────
        # Average only the weights that agree with the elected sign.
        p1_aligned = torch.where(torch.sign(p1_trim) == elected, p1_trim, torch.zeros_like(p1f))
        p2_aligned = torch.where(torch.sign(p2_trim) == elected, p2_trim, torch.zeros_like(p2f))

        count = (p1_aligned != 0).float() + (p2_aligned != 0).float()
        merged = (p1_aligned + p2_aligned) / count.clamp(min=1.0)

        child_state[key] = merged.to(orig_dtype)

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
        # Always operate on CPU to avoid GPU OOM during population evolution
        cpu_param = param.cpu()
        if cpu_param.dtype in _float_dtypes:
            if method == "gaussian":
                # Create mutation mask
                mask = torch.rand_like(cpu_param.float()) < mutation_prob_per_weight
                noise = torch.randn_like(cpu_param.float()) * mutation_scale

                # Apply mutation only where mask is True
                mutated_param = cpu_param.float() + noise * mask.float()
                mutated_state[key] = mutated_param.to(cpu_param.dtype)

            elif method == "uniform":
                mask = torch.rand_like(cpu_param.float()) < mutation_prob_per_weight
                noise = (torch.rand_like(cpu_param.float()) - 0.5) * 2 * mutation_scale
                mutated_param = cpu_param.float() + noise * mask.float()
                mutated_state[key] = mutated_param.to(cpu_param.dtype)

            elif method == "adaptive":
                # Scale mutation by parameter magnitude
                param_scale = torch.abs(cpu_param.float()).mean() + 1e-8
                mask = torch.rand_like(cpu_param.float()) < mutation_prob_per_weight
                noise = torch.randn_like(cpu_param.float()) * mutation_scale * param_scale
                mutated_param = cpu_param.float() + noise * mask.float()
                mutated_state[key] = mutated_param.to(cpu_param.dtype)

            else:
                raise ValueError(f"Unknown mutation method: {method}")
        else:
            mutated_state[key] = cpu_param.clone()

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
        ties_density: float = 0.2,
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
        self.ties_density = ties_density

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
            ties_density=self.ties_density,
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
