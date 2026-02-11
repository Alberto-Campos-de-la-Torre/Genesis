"""Model checkpointing utilities for Genesis."""

from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
import json
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: float = 0.0,
    config: Optional[dict] = None,
    **kwargs,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Save path
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        best_metric: Best metric value
        config: Configuration dictionary
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "timestamp": datetime.now().isoformat(),
    }

    if model is not None:
        checkpoint["model_state_dict"] = model.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config

    checkpoint.update(kwargs)

    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load onto
        strict: Strict loading for model state dict

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("Model state loaded")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state loaded")

    logger.info(f"Checkpoint loaded from {path}")
    return checkpoint


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        checkpoint_prefix: str = "checkpoint",
        metric_name: str = "loss",
        mode: str = "min",
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_prefix: Prefix for checkpoint filenames
            metric_name: Name of metric to track
            mode: 'min' or 'max' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_prefix = checkpoint_prefix
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self._checkpoints: list[dict] = []

        # Load existing checkpoints info
        self._load_checkpoint_info()

    def _load_checkpoint_info(self) -> None:
        """Load information about existing checkpoints."""
        info_path = self.checkpoint_dir / "checkpoints.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                data = json.load(f)
                self._checkpoints = data.get("checkpoints", [])
                self.best_metric = data.get("best_metric", self.best_metric)

    def _save_checkpoint_info(self) -> None:
        """Save checkpoint information to JSON."""
        info_path = self.checkpoint_dir / "checkpoints.json"
        with open(info_path, "w") as f:
            json.dump(
                {
                    "checkpoints": self._checkpoints,
                    "best_metric": self.best_metric,
                },
                f,
                indent=2,
            )

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metric: Optional[float] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            epoch: Current epoch
            step: Current step
            metric: Current metric value
            **kwargs: Additional items to save

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        # Determine checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.checkpoint_prefix}_step{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Check if this is the best checkpoint
        is_best = False
        if metric is not None:
            if self.mode == "min" and metric < self.best_metric:
                self.best_metric = metric
                is_best = True
            elif self.mode == "max" and metric > self.best_metric:
                self.best_metric = metric
                is_best = True

        # Save checkpoint
        save_checkpoint(
            path=str(checkpoint_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            best_metric=self.best_metric,
            **kwargs,
        )

        # Update checkpoint list
        self._checkpoints.append(
            {
                "path": str(checkpoint_path),
                "step": step,
                "epoch": epoch,
                "metric": metric,
                "timestamp": timestamp,
                "is_best": is_best,
            }
        )

        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"New best checkpoint! {self.metric_name}={metric:.4f}")

        # Cleanup old checkpoints
        self._cleanup()

        # Save checkpoint info
        self._save_checkpoint_info()

        return str(checkpoint_path)

    def _cleanup(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        # Keep best checkpoint and most recent ones
        if len(self._checkpoints) <= self.max_checkpoints:
            return

        # Sort by step (most recent first)
        sorted_checkpoints = sorted(
            self._checkpoints,
            key=lambda x: x["step"],
            reverse=True,
        )

        # Keep best and most recent
        to_keep = set()
        for cp in sorted_checkpoints:
            if cp["is_best"]:
                to_keep.add(cp["path"])

        for cp in sorted_checkpoints[: self.max_checkpoints - 1]:
            to_keep.add(cp["path"])

        # Remove old checkpoints
        new_checkpoints = []
        for cp in self._checkpoints:
            if cp["path"] in to_keep:
                new_checkpoints.append(cp)
            else:
                # Delete file
                path = Path(cp["path"])
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed old checkpoint: {cp['path']}")

        self._checkpoints = new_checkpoints

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> dict:
        """
        Load the best checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            device: Device to load onto

        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found")

        return load_checkpoint(
            path=str(best_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> dict:
        """
        Load the most recent checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            device: Device to load onto

        Returns:
            Checkpoint dictionary
        """
        if not self._checkpoints:
            raise FileNotFoundError("No checkpoints found")

        latest = max(self._checkpoints, key=lambda x: x["step"])

        return load_checkpoint(
            path=latest["path"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def get_checkpoint_list(self) -> list[dict]:
        """Get list of all checkpoints."""
        return self._checkpoints.copy()

    @property
    def latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self._checkpoints:
            return None
        return max(self._checkpoints, key=lambda x: x["step"])["path"]

    @property
    def best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pt"
        return str(best_path) if best_path.exists() else None
