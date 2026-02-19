"""Knowledge distillation module for Genesis."""

from genesis.distillation.kd_loss import (
    KDLoss,
    kl_divergence_loss,
    soft_target_loss,
    feature_distillation_loss,
)

__all__ = [
    "KDLoss",
    "kl_divergence_loss",
    "soft_target_loss",
    "feature_distillation_loss",
    "DistillationTrainer",
    "TrainingConfig",
]


def __getattr__(name):
    if name in ("DistillationTrainer", "TrainingConfig"):
        from genesis.distillation.trainer import DistillationTrainer, TrainingConfig
        if name == "DistillationTrainer":
            return DistillationTrainer
        return TrainingConfig
    raise AttributeError(f"module 'genesis.distillation' has no attribute {name!r}")
