"""Weight importance calculation for pruning."""

from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def compute_weight_importance(
    model: nn.Module,
    method: str = "magnitude",
    dataloader: Optional[DataLoader] = None,
    device: str = "cuda",
    num_samples: int = 100,
) -> dict[str, torch.Tensor]:
    """
    Compute importance scores for all model weights.

    Args:
        model: Model to compute importance for
        method: Importance method ('magnitude', 'gradient', 'taylor', 'fisher')
        dataloader: DataLoader for gradient-based methods
        device: Device to run on
        num_samples: Number of samples for gradient-based methods

    Returns:
        Dictionary mapping parameter names to importance scores
    """
    if method == "magnitude":
        return _magnitude_importance(model)
    elif method == "gradient":
        if dataloader is None:
            raise ValueError("DataLoader required for gradient importance")
        return _gradient_importance(model, dataloader, device, num_samples)
    elif method == "taylor":
        if dataloader is None:
            raise ValueError("DataLoader required for Taylor importance")
        return _taylor_importance(model, dataloader, device, num_samples)
    elif method == "fisher":
        if dataloader is None:
            raise ValueError("DataLoader required for Fisher importance")
        return _fisher_importance(model, dataloader, device, num_samples)
    else:
        raise ValueError(f"Unknown importance method: {method}")


def _magnitude_importance(model: nn.Module) -> dict[str, torch.Tensor]:
    """Compute importance based on weight magnitude."""
    importance = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            importance[name] = torch.abs(param.data)
    return importance


def _gradient_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    """Compute importance based on gradient magnitude."""
    model.to(device)
    model.train()

    # Accumulate gradients
    gradient_sums = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    samples_processed = 0

    for batch in dataloader:
        if samples_processed >= num_samples:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
        labels = batch.get("labels", input_ids).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_sums[name] += torch.abs(param.grad)

        samples_processed += input_ids.size(0)

    # Average and compute importance
    importance = {}
    for name, grad_sum in gradient_sums.items():
        importance[name] = grad_sum / max(1, samples_processed)

    return importance


def _taylor_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    """
    Compute importance using first-order Taylor expansion.

    Importance = |weight * gradient|
    """
    model.to(device)
    model.train()

    taylor_sums = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    samples_processed = 0

    for batch in dataloader:
        if samples_processed >= num_samples:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
        labels = batch.get("labels", input_ids).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                taylor_sums[name] += torch.abs(param.data * param.grad)

        samples_processed += input_ids.size(0)

    # Average
    importance = {}
    for name, taylor_sum in taylor_sums.items():
        importance[name] = taylor_sum / max(1, samples_processed)

    return importance


def _fisher_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    """
    Compute importance using Fisher information.

    Approximates the diagonal of the Fisher information matrix.
    """
    model.to(device)
    model.train()

    fisher_sums = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    samples_processed = 0

    for batch in dataloader:
        if samples_processed >= num_samples:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
        labels = batch.get("labels", input_ids).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_sums[name] += param.grad ** 2

        samples_processed += input_ids.size(0)

    # Average (Fisher is expectation of squared gradients)
    importance = {}
    for name, fisher_sum in fisher_sums.items():
        importance[name] = fisher_sum / max(1, samples_processed)

    return importance


class SaliencyCalculator:
    """
    Calculator for weight saliency/importance scores.

    Supports multiple importance methods and caching.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "magnitude",
        dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        """
        Initialize saliency calculator.

        Args:
            model: Model to calculate saliency for
            method: Importance calculation method
            dataloader: DataLoader for gradient-based methods
            device: Device to run on
        """
        self.model = model
        self.method = method
        self.dataloader = dataloader
        self.device = device
        self._importance_cache: Optional[dict[str, torch.Tensor]] = None

    def compute(self, num_samples: int = 100, force_recompute: bool = False) -> dict[str, torch.Tensor]:
        """
        Compute importance scores.

        Args:
            num_samples: Number of samples for gradient-based methods
            force_recompute: Force recomputation even if cached

        Returns:
            Dictionary of importance scores
        """
        if self._importance_cache is not None and not force_recompute:
            return self._importance_cache

        self._importance_cache = compute_weight_importance(
            self.model,
            method=self.method,
            dataloader=self.dataloader,
            device=self.device,
            num_samples=num_samples,
        )

        return self._importance_cache

    def get_top_k_mask(
        self,
        k: float,
        per_layer: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Get mask keeping top k% most important weights.

        Args:
            k: Percentage of weights to keep (0-1)
            per_layer: Whether to apply threshold per layer

        Returns:
            Dictionary of binary masks
        """
        importance = self.compute()
        masks = {}

        if per_layer:
            for name, scores in importance.items():
                threshold = torch.quantile(scores.flatten(), 1 - k)
                masks[name] = (scores >= threshold).float()
        else:
            # Global threshold
            all_scores = torch.cat([s.flatten() for s in importance.values()])
            threshold = torch.quantile(all_scores, 1 - k)

            for name, scores in importance.items():
                masks[name] = (scores >= threshold).float()

        return masks

    def get_pruning_mask(
        self,
        sparsity: float,
        per_layer: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Get mask for achieving target sparsity.

        Args:
            sparsity: Target sparsity (0-1, percentage of weights to prune)
            per_layer: Whether to apply threshold per layer

        Returns:
            Dictionary of binary masks (1 = keep, 0 = prune)
        """
        return self.get_top_k_mask(k=1 - sparsity, per_layer=per_layer)

    def get_importance_ranking(self) -> list[tuple[str, int, float]]:
        """
        Get global ranking of all weights by importance.

        Returns:
            List of (param_name, flat_index, importance_score)
        """
        importance = self.compute()
        rankings = []

        for name, scores in importance.items():
            flat_scores = scores.flatten()
            for idx, score in enumerate(flat_scores):
                rankings.append((name, idx, score.item()))

        # Sort by importance (ascending for pruning)
        rankings.sort(key=lambda x: x[2])
        return rankings

    def clear_cache(self) -> None:
        """Clear cached importance scores."""
        self._importance_cache = None
