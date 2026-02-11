"""Knowledge distillation loss functions."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    Compute KL divergence loss between student and teacher logits.

    Args:
        student_logits: Student model logits [batch, seq_len, vocab]
        teacher_logits: Teacher model logits [batch, seq_len, vocab]
        temperature: Temperature for softmax scaling
        reduction: Reduction method ('batchmean', 'sum', 'mean', 'none')

    Returns:
        KL divergence loss
    """
    # Scale logits by temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction=reduction)

    # Scale by temperature squared (as per original KD paper)
    return kl_loss * (temperature**2)


def soft_target_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute soft target cross-entropy loss.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature for softmax
        attention_mask: Optional mask for valid positions

    Returns:
        Soft target loss
    """
    # Get soft targets from teacher
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Log softmax of student
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # Cross entropy with soft targets
    loss = -(teacher_probs * student_log_probs).sum(dim=-1)

    # Apply mask if provided
    if attention_mask is not None:
        loss = loss * attention_mask
        loss = loss.sum() / attention_mask.sum()
    else:
        loss = loss.mean()

    return loss * (temperature**2)


def feature_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection: Optional[nn.Module] = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute feature distillation loss between hidden states.

    Args:
        student_hidden: Student hidden states [batch, seq_len, hidden_dim]
        teacher_hidden: Teacher hidden states [batch, seq_len, hidden_dim]
        projection: Optional projection layer for dimension mismatch
        loss_type: Loss type ('mse', 'cosine', 'l1')

    Returns:
        Feature distillation loss
    """
    # Project student features if needed
    if projection is not None:
        student_hidden = projection(student_hidden)

    # Normalize hidden states
    student_norm = F.normalize(student_hidden, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)

    if loss_type == "mse":
        loss = F.mse_loss(student_norm, teacher_norm)
    elif loss_type == "cosine":
        loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=-1).mean()
    elif loss_type == "l1":
        loss = F.l1_loss(student_norm, teacher_norm)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


def attention_distillation_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention map distillation loss.

    Args:
        student_attention: Student attention weights [batch, heads, seq, seq]
        teacher_attention: Teacher attention weights [batch, heads, seq, seq]
        attention_mask: Optional attention mask

    Returns:
        Attention distillation loss
    """
    # KL divergence between attention distributions
    student_attn = F.log_softmax(student_attention, dim=-1)
    teacher_attn = F.softmax(teacher_attention, dim=-1)

    kl_loss = F.kl_div(student_attn, teacher_attn, reduction="none")

    if attention_mask is not None:
        # Expand mask for attention heads
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        kl_loss = kl_loss * mask
        return kl_loss.sum() / mask.sum()

    return kl_loss.mean()


class KDLoss(nn.Module):
    """
    Comprehensive knowledge distillation loss module.

    Combines multiple distillation objectives with configurable weights.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        use_feature_distillation: bool = False,
        feature_weight: float = 0.1,
        feature_layers: Optional[list[int]] = None,
        use_attention_distillation: bool = False,
        attention_weight: float = 0.1,
        student_hidden_dim: Optional[int] = None,
        teacher_hidden_dim: Optional[int] = None,
    ):
        """
        Initialize KD loss module.

        Args:
            temperature: Temperature for soft targets
            alpha: Weight for distillation loss vs hard label loss
            use_feature_distillation: Whether to use feature distillation
            feature_weight: Weight for feature distillation loss
            feature_layers: Layer indices for feature distillation
            use_attention_distillation: Whether to use attention distillation
            attention_weight: Weight for attention distillation loss
            student_hidden_dim: Student hidden dimension (for projection)
            teacher_hidden_dim: Teacher hidden dimension (for projection)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_feature_distillation = use_feature_distillation
        self.feature_weight = feature_weight
        self.feature_layers = feature_layers or [-1]
        self.use_attention_distillation = use_attention_distillation
        self.attention_weight = attention_weight

        # Create projection layer if dimensions differ
        self.projection = None
        if (
            use_feature_distillation
            and student_hidden_dim is not None
            and teacher_hidden_dim is not None
            and student_hidden_dim != teacher_hidden_dim
        ):
            self.projection = nn.Linear(student_hidden_dim, teacher_hidden_dim)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: Optional[torch.Tensor] = None,
        student_hidden_states: Optional[tuple[torch.Tensor, ...]] = None,
        teacher_hidden_states: Optional[tuple[torch.Tensor, ...]] = None,
        student_attentions: Optional[tuple[torch.Tensor, ...]] = None,
        teacher_attentions: Optional[tuple[torch.Tensor, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            hard_labels: Optional hard labels
            student_hidden_states: Optional student hidden states
            teacher_hidden_states: Optional teacher hidden states
            student_attentions: Optional student attention weights
            teacher_attentions: Optional teacher attention weights
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing total loss and component losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # KL divergence loss (main distillation objective)
        kd_loss = kl_divergence_loss(
            student_logits,
            teacher_logits,
            temperature=self.temperature,
        )
        losses["kd_loss"] = kd_loss
        total_loss = total_loss + self.alpha * kd_loss

        # Hard label loss
        if hard_labels is not None:
            # Shift for causal LM
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = hard_labels[..., 1:].contiguous()

            hard_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            losses["hard_loss"] = hard_loss
            total_loss = total_loss + (1 - self.alpha) * hard_loss

        # Feature distillation loss
        if (
            self.use_feature_distillation
            and student_hidden_states is not None
            and teacher_hidden_states is not None
        ):
            feature_loss = torch.tensor(0.0, device=student_logits.device)

            for layer_idx in self.feature_layers:
                student_hidden = student_hidden_states[layer_idx]
                teacher_hidden = teacher_hidden_states[layer_idx]

                layer_loss = feature_distillation_loss(
                    student_hidden,
                    teacher_hidden,
                    projection=self.projection,
                )
                feature_loss = feature_loss + layer_loss

            feature_loss = feature_loss / len(self.feature_layers)
            losses["feature_loss"] = feature_loss
            total_loss = total_loss + self.feature_weight * feature_loss

        # Attention distillation loss
        if (
            self.use_attention_distillation
            and student_attentions is not None
            and teacher_attentions is not None
        ):
            attn_loss = torch.tensor(0.0, device=student_logits.device)

            for student_attn, teacher_attn in zip(student_attentions, teacher_attentions):
                layer_loss = attention_distillation_loss(
                    student_attn,
                    teacher_attn,
                    attention_mask,
                )
                attn_loss = attn_loss + layer_loss

            attn_loss = attn_loss / len(student_attentions)
            losses["attention_loss"] = attn_loss
            total_loss = total_loss + self.attention_weight * attn_loss

        losses["total_loss"] = total_loss
        return losses


class ProgressiveKDLoss(KDLoss):
    """
    Progressive knowledge distillation with curriculum.

    Starts with high temperature and alpha, gradually reducing.
    """

    def __init__(
        self,
        initial_temperature: float = 10.0,
        final_temperature: float = 1.0,
        initial_alpha: float = 0.9,
        final_alpha: float = 0.1,
        total_steps: int = 10000,
        **kwargs,
    ):
        super().__init__(temperature=initial_temperature, alpha=initial_alpha, **kwargs)
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self._current_step = 0

    def step(self) -> None:
        """Update temperature and alpha based on current step."""
        progress = min(1.0, self._current_step / self.total_steps)

        # Linear interpolation
        self.temperature = (
            self.initial_temperature
            + (self.final_temperature - self.initial_temperature) * progress
        )
        self.alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress

        self._current_step += 1

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_step = 0
        self.temperature = self.initial_temperature
        self.alpha = self.initial_alpha
