"""Tests for LoRAManager and LoRAConfig."""

import pytest
import torch
import torch.nn as nn

from genesis.models.lora_manager import LoRAManager, LoRAConfig


# ── Mock model with LoRA-named parameters ───────────────────────────────────────

class MockLoRALayer(nn.Module):
    """A layer that exposes lora_A / lora_B weights — discovered by LoRAManager."""

    def __init__(self, in_features: int = 10, out_features: int = 10, r: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))


class MockLoRAModel(nn.Module):
    """Minimal model with LoRA layers."""

    def __init__(self, in_features: int = 10, out_features: int = 10, r: int = 4):
        super().__init__()
        self.q_proj = MockLoRALayer(in_features, out_features, r)
        self.v_proj = MockLoRALayer(in_features, out_features, r)


def _make_manager(r: int = 4) -> tuple[MockLoRAModel, LoRAManager]:
    model = MockLoRAModel(r=r)
    # Use empty target_modules so only 'lora_' name matching is used
    config = LoRAConfig(r=r, target_modules=[])
    manager = LoRAManager(model, config)
    return model, manager


# ── LoRAConfig ──────────────────────────────────────────────────────────────────

class TestLoRAConfig:

    def test_default_construction(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.modules_to_save is None

    def test_custom_construction(self):
        cfg = LoRAConfig(r=8, lora_alpha=16, modules_to_save=["embed_tokens"])
        assert cfg.r == 8
        assert cfg.modules_to_save == ["embed_tokens"]

    def test_settings_loraconfig_has_modules_to_save(self):
        """settings.LoRAConfig must now also expose modules_to_save (bug fix check)."""
        from genesis.config.settings import LoRAConfig as SettingsLoRAConfig
        cfg = SettingsLoRAConfig()
        assert hasattr(cfg, "modules_to_save")
        assert cfg.modules_to_save is None


# ── LoRAManager discovery ────────────────────────────────────────────────────────

class TestLoRAManagerDiscovery:

    def test_discovers_lora_modules(self):
        _, manager = _make_manager()
        # Should discover at least the two MockLoRALayer modules
        assert len(manager._lora_modules) > 0

    def test_get_lora_state_dict_contains_lora_keys(self):
        _, manager = _make_manager()
        state = manager.get_lora_state_dict()
        assert len(state) > 0
        for key in state:
            assert "lora_" in key.lower()

    def test_get_lora_state_dict_excludes_base_weights(self):
        _, manager = _make_manager()
        state = manager.get_lora_state_dict()
        for key in state:
            assert "weight" not in key or "lora_" in key.lower()

    def test_clone_lora_weights_is_independent(self):
        _, manager = _make_manager()
        original = manager.get_lora_state_dict()
        cloned = manager.clone_lora_weights()

        # Mutate cloned; original should not change
        for key in cloned:
            cloned[key].fill_(999.0)

        original_after = manager.get_lora_state_dict()
        for key in original:
            assert not torch.allclose(original_after[key],
                                      torch.full_like(original_after[key], 999.0))


# ── LoRAManager operations ───────────────────────────────────────────────────────

class TestLoRAManagerInterpolation:

    def test_interpolate_at_zero_returns_first(self):
        _, manager = _make_manager()
        sd1 = manager.get_lora_state_dict()
        sd2 = {k: torch.zeros_like(v) for k, v in sd1.items()}
        result = manager.interpolate_lora(sd1, sd2, ratio=0.0)
        for key in sd1:
            assert torch.allclose(result[key], sd1[key])

    def test_interpolate_at_one_returns_second(self):
        _, manager = _make_manager()
        sd1 = manager.get_lora_state_dict()
        sd2 = {k: torch.zeros_like(v) for k, v in sd1.items()}
        result = manager.interpolate_lora(sd1, sd2, ratio=1.0)
        for key in sd2:
            assert torch.allclose(result[key], sd2[key])

    def test_interpolate_midpoint(self):
        _, manager = _make_manager()
        sd1 = {k: torch.ones_like(v) for k, v in manager.get_lora_state_dict().items()}
        sd2 = {k: torch.zeros_like(v) for k, v in sd1.items()}
        result = manager.interpolate_lora(sd1, sd2, ratio=0.5)
        for key in result:
            assert torch.allclose(result[key], torch.full_like(result[key], 0.5))


class TestLoRAManagerMerge:

    def test_merge_multiple_equal_weights(self):
        _, manager = _make_manager()
        sd1 = {k: torch.ones_like(v) for k, v in manager.get_lora_state_dict().items()}
        sd2 = {k: torch.zeros_like(v) for k, v in sd1.items()}
        merged = manager.merge_multiple_lora([sd1, sd2])
        # Equal weights → average
        for key in merged:
            assert torch.allclose(merged[key], torch.full_like(merged[key], 0.5))

    def test_merge_multiple_custom_weights(self):
        _, manager = _make_manager()
        base = manager.get_lora_state_dict()
        sd1 = {k: torch.ones_like(v) for k, v in base.items()}
        sd2 = {k: torch.zeros_like(v) for k, v in sd1.items()}
        merged = manager.merge_multiple_lora([sd1, sd2], weights=[0.8, 0.2])
        for key in merged:
            assert torch.allclose(merged[key], torch.full_like(merged[key], 0.8),
                                  atol=1e-5)

    def test_merge_empty_raises(self):
        _, manager = _make_manager()
        with pytest.raises(ValueError):
            manager.merge_multiple_lora([])


class TestLoRAManagerSimilarity:

    def test_similarity_identical_dicts(self):
        _, manager = _make_manager()
        sd = manager.get_lora_state_dict()
        sim = manager.compute_lora_similarity(sd, sd)
        assert abs(sim - 1.0) < 1e-4  # cosine sim of identical vectors = 1

    def test_similarity_opposite_dicts(self):
        _, manager = _make_manager()
        sd1 = {k: torch.ones_like(v) for k, v in manager.get_lora_state_dict().items()}
        sd2 = {k: -torch.ones_like(v) for k, v in sd1.items()}
        sim = manager.compute_lora_similarity(sd1, sd2)
        assert sim < -0.5  # should be close to -1

    def test_similarity_range(self):
        _, manager = _make_manager()
        sd1 = manager.get_lora_state_dict()
        sd2 = {k: torch.randn_like(v) for k, v in sd1.items()}
        sim = manager.compute_lora_similarity(sd1, sd2)
        assert -1.1 <= sim <= 1.1


class TestLoRAManagerMagnitudeAndNoise:

    def test_magnitude_is_non_negative(self):
        _, manager = _make_manager()
        sd = manager.get_lora_state_dict()
        mag = manager.get_lora_magnitude(sd)
        assert mag >= 0.0

    def test_magnitude_zero_for_zero_weights(self):
        _, manager = _make_manager()
        sd = {k: torch.zeros_like(v) for k, v in manager.get_lora_state_dict().items()}
        assert manager.get_lora_magnitude(sd) == 0.0

    def test_scale_weights(self):
        _, manager = _make_manager()
        sd = {k: torch.ones_like(v) for k, v in manager.get_lora_state_dict().items()}
        scaled = manager.scale_lora_weights(sd, scale=2.0)
        for key in scaled:
            assert torch.allclose(scaled[key], torch.full_like(scaled[key], 2.0))

    def test_add_noise_changes_weights(self):
        _, manager = _make_manager()
        sd = {k: torch.zeros_like(v) for k, v in manager.get_lora_state_dict().items()}
        noisy = manager.add_noise_to_lora(sd, noise_scale=0.1)
        changed = any(not torch.allclose(noisy[k], sd[k]) for k in sd)
        assert changed

    def test_add_noise_preserves_shape(self):
        _, manager = _make_manager()
        sd = manager.get_lora_state_dict()
        noisy = manager.add_noise_to_lora(sd, noise_scale=0.01)
        for key in sd:
            assert noisy[key].shape == sd[key].shape


class TestLoRAManagerFreeze:

    def test_freeze_base_model(self):
        model, manager = _make_manager()
        manager.freeze_base_model()
        for name, param in model.named_parameters():
            if "lora_" not in name.lower():
                assert not param.requires_grad

    def test_unfreeze_all(self):
        model, manager = _make_manager()
        manager.freeze_base_model()
        manager.unfreeze_all()
        for param in model.parameters():
            assert param.requires_grad

    def test_trainable_param_count(self):
        model, manager = _make_manager()
        trainable, total = manager.get_trainable_param_count()
        assert trainable > 0
        assert total >= trainable


class TestLoRAManagerSetState:

    def test_set_lora_state_dict_updates_weights(self):
        model, manager = _make_manager()
        new_sd = {k: torch.zeros_like(v)
                  for k, v in manager.get_lora_state_dict().items()}
        manager.set_lora_state_dict(new_sd, strict=False)
        current = manager.get_lora_state_dict()
        for key in new_sd:
            assert torch.allclose(current[key], torch.zeros_like(current[key]))

    def test_set_lora_state_dict_unknown_key_strict_raises(self):
        _, manager = _make_manager()
        bad_sd = {"nonexistent_key": torch.zeros(1)}
        with pytest.raises(KeyError):
            manager.set_lora_state_dict(bad_sd, strict=True)
