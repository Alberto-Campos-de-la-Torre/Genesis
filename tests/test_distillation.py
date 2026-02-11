"""Tests for distillation module."""

import pytest
import torch
import torch.nn as nn

from genesis.distillation.kd_loss import (
    kl_divergence_loss,
    soft_target_loss,
    feature_distillation_loss,
    KDLoss,
    ProgressiveKDLoss,
)


class TestKLDivergenceLoss:
    """Tests for KL divergence loss."""

    def test_kl_divergence_basic(self):
        """Test basic KL divergence computation."""
        student_logits = torch.randn(2, 10, 100)  # batch, seq, vocab
        teacher_logits = torch.randn(2, 10, 100)

        loss = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)

        assert loss.dim() == 0  # Scalar
        assert loss >= 0  # KL divergence is non-negative

    def test_kl_divergence_same_logits(self):
        """Test KL divergence with identical logits."""
        logits = torch.randn(2, 10, 100)

        loss = kl_divergence_loss(logits, logits.clone(), temperature=1.0)

        # KL divergence should be ~0 for identical distributions
        assert loss < 1e-5

    def test_kl_divergence_temperature_scaling(self):
        """Test that higher temperature produces different loss."""
        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)

        loss_t1 = kl_divergence_loss(student_logits, teacher_logits, temperature=1.0)
        loss_t4 = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)

        # Losses should be different with different temperatures
        assert not torch.allclose(loss_t1, loss_t4)


class TestSoftTargetLoss:
    """Tests for soft target loss."""

    def test_soft_target_basic(self):
        """Test basic soft target loss."""
        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)

        loss = soft_target_loss(student_logits, teacher_logits, temperature=4.0)

        assert loss.dim() == 0
        assert loss >= 0

    def test_soft_target_with_mask(self):
        """Test soft target loss with attention mask."""
        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Mask out last 5 positions

        loss = soft_target_loss(
            student_logits, teacher_logits, temperature=4.0, attention_mask=mask
        )

        assert loss.dim() == 0


class TestFeatureDistillationLoss:
    """Tests for feature distillation loss."""

    def test_feature_loss_mse(self):
        """Test MSE feature distillation loss."""
        student_hidden = torch.randn(2, 10, 256)
        teacher_hidden = torch.randn(2, 10, 256)

        loss = feature_distillation_loss(
            student_hidden, teacher_hidden, loss_type="mse"
        )

        assert loss.dim() == 0
        assert loss >= 0

    def test_feature_loss_cosine(self):
        """Test cosine feature distillation loss."""
        student_hidden = torch.randn(2, 10, 256)
        teacher_hidden = torch.randn(2, 10, 256)

        loss = feature_distillation_loss(
            student_hidden, teacher_hidden, loss_type="cosine"
        )

        assert loss.dim() == 0
        assert 0 <= loss <= 2  # Cosine loss range

    def test_feature_loss_with_projection(self):
        """Test feature loss with dimension projection."""
        student_hidden = torch.randn(2, 10, 128)  # Smaller dimension
        teacher_hidden = torch.randn(2, 10, 256)  # Larger dimension

        projection = nn.Linear(128, 256)

        loss = feature_distillation_loss(
            student_hidden, teacher_hidden, projection=projection
        )

        assert loss.dim() == 0


class TestKDLoss:
    """Tests for KDLoss class."""

    def test_kdloss_basic(self):
        """Test basic KDLoss forward pass."""
        kd_loss = KDLoss(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)

        losses = kd_loss(student_logits, teacher_logits)

        assert "total_loss" in losses
        assert "kd_loss" in losses

    def test_kdloss_with_hard_labels(self):
        """Test KDLoss with hard labels."""
        kd_loss = KDLoss(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        losses = kd_loss(student_logits, teacher_logits, hard_labels=labels)

        assert "hard_loss" in losses
        assert "total_loss" in losses

    def test_kdloss_with_feature_distillation(self):
        """Test KDLoss with feature distillation."""
        kd_loss = KDLoss(
            temperature=4.0,
            alpha=0.5,
            use_feature_distillation=True,
            feature_weight=0.1,
            feature_layers=[-1],
        )

        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)
        student_hidden = (torch.randn(2, 10, 256),)
        teacher_hidden = (torch.randn(2, 10, 256),)

        losses = kd_loss(
            student_logits,
            teacher_logits,
            student_hidden_states=student_hidden,
            teacher_hidden_states=teacher_hidden,
        )

        assert "feature_loss" in losses

    def test_kdloss_alpha_weighting(self):
        """Test that alpha correctly weights losses."""
        # All KD loss
        kd_loss_all_kd = KDLoss(alpha=1.0)
        # All hard loss
        kd_loss_all_hard = KDLoss(alpha=0.0)

        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        losses_all_kd = kd_loss_all_kd(
            student_logits, teacher_logits, hard_labels=labels
        )
        losses_all_hard = kd_loss_all_hard(
            student_logits, teacher_logits, hard_labels=labels
        )

        # Total loss should equal KD loss when alpha=1
        assert torch.allclose(
            losses_all_kd["total_loss"], losses_all_kd["kd_loss"], atol=1e-5
        )


class TestProgressiveKDLoss:
    """Tests for ProgressiveKDLoss."""

    def test_progressive_initialization(self):
        """Test progressive KD loss initialization."""
        loss = ProgressiveKDLoss(
            initial_temperature=10.0,
            final_temperature=1.0,
            initial_alpha=0.9,
            final_alpha=0.1,
            total_steps=100,
        )

        assert loss.temperature == 10.0
        assert loss.alpha == 0.9

    def test_progressive_step(self):
        """Test progressive stepping."""
        loss = ProgressiveKDLoss(
            initial_temperature=10.0,
            final_temperature=1.0,
            initial_alpha=0.9,
            final_alpha=0.1,
            total_steps=100,
        )

        initial_temp = loss.temperature
        initial_alpha = loss.alpha

        for _ in range(50):
            loss.step()

        # Should be halfway through progression
        assert loss.temperature < initial_temp
        assert loss.alpha < initial_alpha

    def test_progressive_full_progression(self):
        """Test full progression to final values."""
        loss = ProgressiveKDLoss(
            initial_temperature=10.0,
            final_temperature=1.0,
            initial_alpha=0.9,
            final_alpha=0.1,
            total_steps=100,
        )

        for _ in range(100):
            loss.step()

        assert abs(loss.temperature - 1.0) < 0.1
        assert abs(loss.alpha - 0.1) < 0.1

    def test_progressive_reset(self):
        """Test progressive reset."""
        loss = ProgressiveKDLoss(
            initial_temperature=10.0,
            final_temperature=1.0,
            total_steps=100,
        )

        for _ in range(50):
            loss.step()

        loss.reset()

        assert loss.temperature == 10.0
        assert loss._current_step == 0
