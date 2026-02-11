"""Pruning strategies for model compression."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from genesis.pruning.saliency import SaliencyCalculator, compute_weight_importance

logger = logging.getLogger(__name__)


class PruningStrategy(Enum):
    """Pruning strategy types."""

    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"
    TAYLOR = "taylor"
    RANDOM = "random"
    STRUCTURED = "structured"


@dataclass
class PruningConfig:
    """Configuration for model pruning."""

    target_sparsity: float = 0.3
    pruning_method: str = "magnitude"
    structured: bool = False
    granularity: str = "element"  # element, row, column, block
    block_size: int = 4
    iterative_steps: int = 1

    # Gradual pruning
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.3
    pruning_schedule: str = "cubic"  # linear, cubic, exponential

    # Layer-wise settings
    skip_layers: list[str] = field(default_factory=list)
    layer_sparsity_overrides: dict[str, float] = field(default_factory=dict)


class Pruner:
    """
    Model pruner supporting various pruning strategies.

    Handles both unstructured and structured pruning with
    configurable sparsity schedules.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[PruningConfig] = None,
        dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        """
        Initialize pruner.

        Args:
            model: Model to prune
            config: Pruning configuration
            dataloader: DataLoader for importance calculation
            device: Device to run on
        """
        self.model = model
        self.config = config or PruningConfig()
        self.dataloader = dataloader
        self.device = device

        self._masks: dict[str, torch.Tensor] = {}
        self._original_weights: dict[str, torch.Tensor] = {}
        self._current_sparsity = 0.0
        self._pruning_step = 0

        self._saliency_calculator = SaliencyCalculator(
            model=model,
            method=self.config.pruning_method,
            dataloader=dataloader,
            device=device,
        )

    def compute_target_sparsity(self) -> float:
        """Compute target sparsity for current step based on schedule."""
        if self.config.iterative_steps <= 1:
            return self.config.final_sparsity

        progress = self._pruning_step / (self.config.iterative_steps - 1)
        progress = min(1.0, max(0.0, progress))

        initial = self.config.initial_sparsity
        final = self.config.final_sparsity

        if self.config.pruning_schedule == "linear":
            return initial + (final - initial) * progress
        elif self.config.pruning_schedule == "cubic":
            return initial + (final - initial) * (progress ** 3)
        elif self.config.pruning_schedule == "exponential":
            return initial + (final - initial) * (1 - (1 - progress) ** 3)
        else:
            return final

    def prune(self, sparsity: Optional[float] = None) -> dict[str, float]:
        """
        Prune the model to target sparsity.

        Args:
            sparsity: Target sparsity (overrides config if provided)

        Returns:
            Dictionary with pruning statistics
        """
        target_sparsity = sparsity or self.compute_target_sparsity()

        if self.config.structured:
            stats = self._structured_prune(target_sparsity)
        else:
            stats = self._unstructured_prune(target_sparsity)

        self._current_sparsity = target_sparsity
        self._pruning_step += 1

        logger.info(f"Pruning complete. Sparsity: {stats['actual_sparsity']:.2%}")
        return stats

    def _unstructured_prune(self, target_sparsity: float) -> dict[str, float]:
        """Apply unstructured (element-wise) pruning."""
        # Compute importance scores
        importance = self._saliency_calculator.compute()

        # Get masks
        self._masks = self._saliency_calculator.get_pruning_mask(
            sparsity=target_sparsity,
            per_layer=True,
        )

        # Apply masks
        total_params = 0
        pruned_params = 0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._masks and name not in self.config.skip_layers:
                    # Apply layer-specific sparsity if specified
                    if name in self.config.layer_sparsity_overrides:
                        layer_mask = self._saliency_calculator.get_pruning_mask(
                            sparsity=self.config.layer_sparsity_overrides[name],
                            per_layer=True,
                        )[name]
                    else:
                        layer_mask = self._masks[name]

                    # Store original weights
                    if name not in self._original_weights:
                        self._original_weights[name] = param.data.clone()

                    # Apply mask
                    param.data *= layer_mask.to(param.device)

                    total_params += param.numel()
                    pruned_params += (layer_mask == 0).sum().item()

        actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0

        return {
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sparsity,
            "total_params": total_params,
            "pruned_params": pruned_params,
        }

    def _structured_prune(self, target_sparsity: float) -> dict[str, float]:
        """Apply structured pruning (row/column/block)."""
        importance = self._saliency_calculator.compute()

        total_structures = 0
        pruned_structures = 0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in importance or name in self.config.skip_layers:
                    continue

                if param.dim() < 2:
                    continue  # Skip 1D parameters

                scores = importance[name]

                if self.config.granularity == "row":
                    # Compute row importance (sum along columns)
                    row_importance = scores.abs().sum(dim=-1)
                    num_rows = row_importance.numel()
                    num_to_prune = int(num_rows * target_sparsity)

                    # Get indices of rows to prune
                    _, indices = torch.topk(row_importance, num_to_prune, largest=False)

                    # Create mask
                    mask = torch.ones_like(param)
                    for idx in indices:
                        mask[idx] = 0

                    total_structures += num_rows
                    pruned_structures += num_to_prune

                elif self.config.granularity == "column":
                    # Compute column importance
                    col_importance = scores.abs().sum(dim=0)
                    num_cols = col_importance.numel()
                    num_to_prune = int(num_cols * target_sparsity)

                    _, indices = torch.topk(col_importance, num_to_prune, largest=False)

                    mask = torch.ones_like(param)
                    for idx in indices:
                        mask[:, idx] = 0

                    total_structures += num_cols
                    pruned_structures += num_to_prune

                elif self.config.granularity == "block":
                    # Block-wise pruning
                    mask = self._block_prune(param, scores, target_sparsity)
                    block_size = self.config.block_size
                    num_blocks = (param.numel() // (block_size ** 2))
                    total_structures += num_blocks
                    pruned_structures += int(num_blocks * target_sparsity)

                else:
                    continue

                # Store original and apply mask
                if name not in self._original_weights:
                    self._original_weights[name] = param.data.clone()

                self._masks[name] = mask
                param.data *= mask.to(param.device)

        actual_sparsity = pruned_structures / total_structures if total_structures > 0 else 0.0

        return {
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sparsity,
            "total_structures": total_structures,
            "pruned_structures": pruned_structures,
            "granularity": self.config.granularity,
        }

    def _block_prune(
        self,
        param: torch.Tensor,
        scores: torch.Tensor,
        target_sparsity: float,
    ) -> torch.Tensor:
        """Apply block-wise pruning."""
        block_size = self.config.block_size
        mask = torch.ones_like(param)

        if param.dim() != 2:
            return mask

        rows, cols = param.shape
        block_rows = rows // block_size
        block_cols = cols // block_size

        # Compute block importance
        block_importance = []
        block_indices = []

        for i in range(block_rows):
            for j in range(block_cols):
                block = scores[
                    i * block_size: (i + 1) * block_size,
                    j * block_size: (j + 1) * block_size,
                ]
                block_importance.append(block.abs().sum().item())
                block_indices.append((i, j))

        # Sort by importance and prune
        sorted_indices = sorted(
            range(len(block_importance)),
            key=lambda x: block_importance[x],
        )

        num_to_prune = int(len(sorted_indices) * target_sparsity)

        for idx in sorted_indices[:num_to_prune]:
            i, j = block_indices[idx]
            mask[
                i * block_size: (i + 1) * block_size,
                j * block_size: (j + 1) * block_size,
            ] = 0

        return mask

    def restore_weights(self) -> None:
        """Restore original unpruned weights."""
        with torch.no_grad():
            for name, original in self._original_weights.items():
                for param_name, param in self.model.named_parameters():
                    if param_name == name:
                        param.data.copy_(original)
                        break

        self._masks.clear()
        self._current_sparsity = 0.0
        logger.info("Weights restored to original values")

    def apply_masks(self) -> None:
        """Re-apply stored masks (useful after optimizer step)."""
        with torch.no_grad():
            for name, mask in self._masks.items():
                for param_name, param in self.model.named_parameters():
                    if param_name == name:
                        param.data *= mask.to(param.device)
                        break

    def get_sparsity_stats(self) -> dict[str, float]:
        """Get current sparsity statistics."""
        total_params = 0
        zero_params = 0
        layer_stats = {}

        for name, param in self.model.named_parameters():
            layer_total = param.numel()
            layer_zeros = (param == 0).sum().item()

            total_params += layer_total
            zero_params += layer_zeros

            layer_stats[name] = {
                "total": layer_total,
                "zeros": layer_zeros,
                "sparsity": layer_zeros / layer_total if layer_total > 0 else 0,
            }

        return {
            "global_sparsity": zero_params / total_params if total_params > 0 else 0,
            "total_params": total_params,
            "zero_params": zero_params,
            "layer_stats": layer_stats,
        }

    def make_pruning_permanent(self) -> None:
        """Make pruning permanent by removing mask references."""
        self._original_weights.clear()
        logger.info("Pruning made permanent")

    @property
    def current_sparsity(self) -> float:
        """Get current sparsity level."""
        return self._current_sparsity

    @property
    def masks(self) -> dict[str, torch.Tensor]:
        """Get current pruning masks."""
        return self._masks


def iterative_prune(
    model: nn.Module,
    config: PruningConfig,
    dataloader: Optional[DataLoader] = None,
    fine_tune_fn: Optional[callable] = None,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Perform iterative pruning with optional fine-tuning between steps.

    Args:
        model: Model to prune
        config: Pruning configuration
        dataloader: DataLoader for importance calculation
        fine_tune_fn: Optional fine-tuning function called after each step
        device: Device to run on

    Returns:
        Final pruning statistics
    """
    pruner = Pruner(model, config, dataloader, device)

    for step in range(config.iterative_steps):
        logger.info(f"Pruning step {step + 1}/{config.iterative_steps}")

        stats = pruner.prune()

        if fine_tune_fn is not None and step < config.iterative_steps - 1:
            logger.info("Fine-tuning after pruning step...")
            fine_tune_fn(model)

            # Re-apply masks after fine-tuning
            pruner.apply_masks()

    pruner.make_pruning_permanent()
    return pruner.get_sparsity_stats()
