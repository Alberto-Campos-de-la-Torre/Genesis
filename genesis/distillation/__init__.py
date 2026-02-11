"""Knowledge distillation module for Genesis."""

from genesis.distillation.kd_loss import (
    KDLoss,
    kl_divergence_loss,
    soft_target_loss,
    feature_distillation_loss,
)
from genesis.distillation.trainer import DistillationTrainer, TrainingConfig

__all__ = [
    "KDLoss",
    "kl_divergence_loss",
    "soft_target_loss",
    "feature_distillation_loss",
    "DistillationTrainer",
    "TrainingConfig",
]
