"""Tests for GenesisConfig, sub-configs, and HardwareConfig."""

import pytest
import torch
import yaml

from genesis.config.settings import (
    GenesisConfig,
    GeneticConfig,
    DistillationConfig,
    PruningConfig,
    LoRAConfig,
)
from genesis.config.hardware import HardwareConfig


# ── GeneticConfig ────────────────────────────────────────────────────────────────

class TestGeneticConfig:

    def test_defaults(self):
        cfg = GeneticConfig()
        assert cfg.population_size == 20
        assert cfg.generations == 50
        assert 0.0 < cfg.mutation_rate < 1.0
        assert cfg.elite_size >= 1

    def test_custom_values(self):
        cfg = GeneticConfig(population_size=10, generations=5, mutation_rate=0.2)
        assert cfg.population_size == 10
        assert cfg.generations == 5
        assert cfg.mutation_rate == 0.2

    def test_adaptive_mutation_fields(self):
        cfg = GeneticConfig()
        assert hasattr(cfg, "adaptive_mutation")
        assert hasattr(cfg, "mutation_decay")
        assert hasattr(cfg, "min_mutation_rate")


# ── DistillationConfig ───────────────────────────────────────────────────────────

class TestDistillationConfig:

    def test_defaults(self):
        cfg = DistillationConfig()
        assert cfg.temperature > 1.0
        assert 0.0 < cfg.alpha <= 1.0

    def test_custom_temperature(self):
        cfg = DistillationConfig(temperature=2.0, alpha=0.3)
        assert cfg.temperature == 2.0
        assert cfg.alpha == 0.3


# ── PruningConfig ────────────────────────────────────────────────────────────────

class TestPruningConfig:

    def test_defaults(self):
        cfg = PruningConfig()
        assert 0.0 < cfg.target_sparsity < 1.0
        assert cfg.pruning_method in {"magnitude", "gradient", "taylor"}

    def test_has_all_pruner_fields(self):
        """settings.PruningConfig must be field-compatible with pruner.PruningConfig."""
        cfg = PruningConfig()
        # Fields added during the high-severity bug fix
        assert hasattr(cfg, "block_size")
        assert hasattr(cfg, "skip_layers")
        assert hasattr(cfg, "layer_sparsity_overrides")
        assert isinstance(cfg.skip_layers, list)
        assert isinstance(cfg.layer_sparsity_overrides, dict)


# ── LoRAConfig ───────────────────────────────────────────────────────────────────

class TestLoRAConfig:

    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r > 0
        assert cfg.lora_alpha > 0
        assert isinstance(cfg.target_modules, list)

    def test_has_modules_to_save(self):
        """Bug-fix check: modules_to_save must exist."""
        cfg = LoRAConfig()
        assert hasattr(cfg, "modules_to_save")
        assert cfg.modules_to_save is None

    def test_custom_modules_to_save(self):
        cfg = LoRAConfig(modules_to_save=["embed_tokens", "lm_head"])
        assert cfg.modules_to_save == ["embed_tokens", "lm_head"]


# ── GenesisConfig ────────────────────────────────────────────────────────────────

class TestGenesisConfig:

    def test_default_construction(self, tmp_path):
        cfg = GenesisConfig(
            output_dir=str(tmp_path / "outputs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )
        assert cfg.project_name == "genesis_experiment"
        assert isinstance(cfg.genetic, GeneticConfig)
        assert isinstance(cfg.distillation, DistillationConfig)
        assert isinstance(cfg.pruning, PruningConfig)
        assert isinstance(cfg.lora, LoRAConfig)

    def test_post_init_creates_directories(self, tmp_path):
        out = tmp_path / "out"
        ckpt = tmp_path / "ckpt"
        logs = tmp_path / "logs"
        GenesisConfig(
            output_dir=str(out),
            checkpoint_dir=str(ckpt),
            log_dir=str(logs),
        )
        assert out.exists()
        assert ckpt.exists()
        assert logs.exists()

    def test_from_dict_basic(self, tmp_path):
        d = {
            "project_name": "test_proj",
            "output_dir": str(tmp_path / "out"),
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
        }
        cfg = GenesisConfig.from_dict(d)
        assert cfg.project_name == "test_proj"

    def test_from_dict_with_sub_configs(self, tmp_path):
        d = {
            "output_dir": str(tmp_path / "out"),
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
            "genetic": {"population_size": 8, "generations": 3},
            "distillation": {"temperature": 2.0},
            "pruning": {},
            "lora": {},
        }
        cfg = GenesisConfig.from_dict(d)
        assert cfg.genetic.population_size == 8
        assert cfg.genetic.generations == 3
        assert cfg.distillation.temperature == 2.0

    def test_to_dict_roundtrip(self, tmp_path):
        cfg = GenesisConfig(
            project_name="roundtrip_test",
            output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        )
        d = cfg.to_dict()
        assert d["project_name"] == "roundtrip_test"
        assert "genetic" in d
        assert "distillation" in d
        assert "pruning" in d
        assert "lora" in d

    def test_to_dict_contains_all_pruning_fields(self, tmp_path):
        """Bug-fix check: to_dict() must include the new pruning fields."""
        cfg = GenesisConfig(
            output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        )
        pruning_dict = cfg.to_dict()["pruning"]
        assert "block_size" in pruning_dict
        assert "skip_layers" in pruning_dict
        assert "layer_sparsity_overrides" in pruning_dict

    def test_to_dict_slerp_ratio_from_genetic(self, tmp_path):
        """Bug-fix check: slerp_ratio must come from genetic, no dead hasattr branch."""
        cfg = GenesisConfig(
            output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        )
        d = cfg.to_dict()
        assert d["genetic"]["slerp_ratio"] == cfg.genetic.slerp_ratio

    def test_yaml_roundtrip(self, tmp_path):
        cfg = GenesisConfig(
            project_name="yaml_test",
            output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        )
        cfg.genetic.population_size = 7
        cfg.distillation.temperature = 3.5

        yaml_path = str(tmp_path / "config.yaml")
        cfg.to_yaml(yaml_path)

        loaded = GenesisConfig.from_yaml(yaml_path)
        assert loaded.project_name == "yaml_test"
        assert loaded.genetic.population_size == 7
        assert abs(loaded.distillation.temperature - 3.5) < 1e-6

    def test_lora_dict_includes_modules_to_save(self, tmp_path):
        cfg = GenesisConfig(
            output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        )
        lora_dict = cfg.to_dict()["lora"]
        assert "modules_to_save" in lora_dict


# ── HardwareConfig ───────────────────────────────────────────────────────────────

class TestHardwareConfig:

    def test_default_construction(self):
        hw = HardwareConfig()
        assert hw.teacher_device is not None
        assert hw.student_device is not None

    def test_memory_summary_returns_string(self):
        hw = HardwareConfig()
        summary = hw.memory_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_cpu_fallback_when_no_cuda(self):
        """If CUDA is unavailable, devices must fall back to cpu."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available — fallback not triggered")
        hw = HardwareConfig(teacher_device="cuda:0", student_device="cuda:1")
        assert "cpu" in hw.teacher_device or "cpu" in hw.student_device

    def test_explicit_cpu_devices(self):
        hw = HardwareConfig(teacher_device="cpu", student_device="cpu")
        assert hw.teacher_device == "cpu"
        assert hw.student_device == "cpu"
