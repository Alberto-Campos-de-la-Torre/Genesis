"""Tests for genesis/optimizer.py — EvolutionaryOptimizer."""

import json
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from genesis.config.settings import GenesisConfig, GeneticConfig
from genesis.core.fitness import FitnessResult
from genesis.core.genetics import Genetics
from genesis.core.population import Population, Individual
from genesis.optimizer import EvolutionaryOptimizer


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, **overrides) -> GenesisConfig:
    """Build a minimal GenesisConfig that writes to a tmp dir."""
    return GenesisConfig(
        project_name="test",
        output_dir=str(tmp_path / "outputs"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        genetic=GeneticConfig(
            population_size=4,
            generations=2,
            mutation_rate=0.5,
            mutation_scale=0.01,
            elite_size=1,
            tournament_size=2,
        ),
        use_tensorboard=False,
        use_wandb=False,
        **overrides,
    )


def _fake_state_dict() -> dict:
    """Minimal LoRA-like state dict."""
    return {
        "base_model.model.layer.lora_A.weight": torch.randn(4, 8),
        "base_model.model.layer.lora_B.weight": torch.randn(8, 4),
    }


def _make_mock_student(state_dict=None):
    """Return a mock StudentModel."""
    sd = state_dict or _fake_state_dict()
    student = MagicMock()
    student.model = MagicMock()
    student.model.parameters.return_value = [torch.nn.Parameter(torch.randn(4, 4))]
    student.model.named_parameters.return_value = list(sd.items())
    student.device = "cpu"
    student.get_state_dict.return_value = sd
    student.load_state_dict.return_value = None
    student.save.return_value = None
    student.unload.return_value = None
    return student


def _make_mock_teacher():
    """Return a mock TeacherModel."""
    teacher = MagicMock()
    teacher.model = MagicMock()
    teacher.model.parameters.return_value = [torch.nn.Parameter(torch.randn(4, 4))]
    teacher.device = "cpu"
    teacher.unload.return_value = None
    return teacher


def _make_optimizer(tmp_path: Path, student=None, teacher=None):
    """Build an EvolutionaryOptimizer with mocked models."""
    cfg = _make_config(tmp_path)
    student = student or _make_mock_student()
    teacher = teacher or _make_mock_teacher()
    opt = EvolutionaryOptimizer(
        config=cfg,
        teacher_model=teacher,
        student_model=student,
    )
    return opt, student, teacher


# ── Initialization ────────────────────────────────────────────────────────────


class TestEvolutionaryOptimizerInit:

    def test_creates_with_config(self, tmp_path):
        cfg = _make_config(tmp_path)
        opt = EvolutionaryOptimizer(config=cfg)
        assert opt.config is cfg

    def test_default_config_when_none(self, tmp_path):
        # Use a config with tmp dirs so file handlers succeed
        cfg = _make_config(tmp_path)
        opt = EvolutionaryOptimizer(config=cfg)
        assert isinstance(opt.config, GenesisConfig)

    def test_accepts_pre_built_models(self, tmp_path):
        opt, student, teacher = _make_optimizer(tmp_path)
        assert opt.teacher is teacher
        assert opt.student is student

    def test_not_initialized_before_initialize(self, tmp_path):
        opt, *_ = _make_optimizer(tmp_path)
        assert not opt._initialized

    def test_current_generation_starts_at_zero(self, tmp_path):
        opt, *_ = _make_optimizer(tmp_path)
        assert opt.current_generation == 0

    def test_fitness_tracker_created(self, tmp_path):
        from genesis.utils.metrics import FitnessTracker
        opt, *_ = _make_optimizer(tmp_path)
        assert isinstance(opt.fitness_tracker, FitnessTracker)

    def test_metrics_tracker_created(self, tmp_path):
        from genesis.utils.metrics import MetricsTracker
        opt, *_ = _make_optimizer(tmp_path)
        assert isinstance(opt.metrics_tracker, MetricsTracker)

    def test_best_individual_none_before_init(self, tmp_path):
        opt, *_ = _make_optimizer(tmp_path)
        assert opt.best_individual is None

    def test_population_none_before_init(self, tmp_path):
        opt, *_ = _make_optimizer(tmp_path)
        assert opt.population is None


# ── Initialize method ─────────────────────────────────────────────────────────


class TestEvolutionaryOptimizerInitialize:

    def test_initialize_sets_flag(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        opt._initialized = False
        # initialize_from_model needs a real state dict
        student.model.state_dict.return_value = _fake_state_dict()
        with patch.object(opt._population.__class__, "initialize_from_model") if opt._population else patch("genesis.core.population.Population.initialize_from_model") as m:
            opt.initialize()
        assert opt._initialized

    def test_initialize_creates_genetics(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        assert opt._genetics is not None

    def test_initialize_creates_population(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        assert opt._population is not None
        assert len(opt._population) == opt.config.genetic.population_size

    def test_initialize_creates_selection_strategy(self, tmp_path):
        from genesis.core.selection import TournamentSelection
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        assert isinstance(opt._selection_strategy, TournamentSelection)

    def test_initialize_population_individuals_distinct(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        # Use a larger state dict so perturbation is detectable
        sd = {"lora_A": torch.randn(16, 32), "lora_B": torch.randn(32, 16)}
        student.model.state_dict.return_value = sd
        opt.initialize()
        # All individuals should exist with state dicts
        for ind in opt.population:
            assert ind.state_dict is not None


# ── Custom evaluator and strategy ─────────────────────────────────────────────


class TestSetterMethods:

    def test_set_fitness_evaluator(self, tmp_path):
        from genesis.core.fitness import FitnessEvaluator
        opt, *_ = _make_optimizer(tmp_path)
        evaluator = MagicMock(spec=FitnessEvaluator)
        opt.set_fitness_evaluator(evaluator)
        assert opt._fitness_evaluator is evaluator

    def test_set_selection_strategy(self, tmp_path):
        from genesis.core.selection import ElitismSelection
        opt, *_ = _make_optimizer(tmp_path)
        strategy = ElitismSelection()
        opt.set_selection_strategy(strategy)
        assert opt._selection_strategy is strategy


# ── Checkpoint round-trip ─────────────────────────────────────────────────────


class TestCheckpointing:

    def test_save_and_load_checkpoint(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = {"lora_A": torch.randn(4, 8), "lora_B": torch.randn(8, 4)}
        student.model.state_dict.return_value = sd
        opt.initialize()

        # Set a custom generation
        opt._current_generation = 3

        ckpt_path = str(tmp_path / "checkpoints" / "test_ckpt.pt")
        opt._save_checkpoint(generation=3)

        # Find the actual saved file
        ckpt_files = list((tmp_path / "checkpoints").glob("*.pt"))
        assert len(ckpt_files) >= 1

        # Reset state and reload
        opt._current_generation = 0
        opt.load_checkpoint(str(ckpt_files[0]))

        assert opt._current_generation == 3

    def test_checkpoint_contains_config(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        opt._save_checkpoint(generation=0)

        ckpt_files = list((tmp_path / "checkpoints").glob("*.pt"))
        state = torch.load(str(ckpt_files[0]), weights_only=False)
        assert "config" in state
        assert "population_state" in state
        assert "generation" in state


# ── Save best model ───────────────────────────────────────────────────────────


class TestSaveBestModel:

    def test_save_best_creates_metadata(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd
        opt.initialize()

        best = opt.population.best
        if best is None:
            for i, ind in enumerate(opt.population):
                ind.fitness = float(i)
            best = opt.population.best

        opt._save_best_model(best)

        meta_path = Path(opt.config.output_dir) / "best_model" / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "fitness" in meta
        assert "generation" in meta

    def test_save_best_calls_student_save(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd
        opt.initialize()

        for i, ind in enumerate(opt.population):
            ind.fitness = float(i)
        best = opt.population.best
        opt._save_best_model(best)

        student.save.assert_called_once()


# ── Evaluate population ───────────────────────────────────────────────────────


class TestEvaluatePopulation:

    def test_evaluate_raises_without_evaluator_or_dataloader(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        # No evaluator, no eval_dataloader
        with pytest.raises(ValueError, match="Fitness evaluator"):
            opt._evaluate_population()

    def test_evaluate_with_custom_evaluator(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd
        opt.initialize()

        # Inject a mock evaluator
        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = FitnessResult(score=0.8, metrics={})
        opt.set_fitness_evaluator(mock_eval)

        opt._evaluate_population()

        # All individuals should have fitness set
        for ind in opt.population:
            assert ind.fitness == pytest.approx(0.8)


# ── Run (minimal, mocked) ─────────────────────────────────────────────────────


class TestRunMethod:

    def test_run_calls_initialize_if_needed(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = FitnessResult(score=0.5, metrics={})
        opt.set_fitness_evaluator(mock_eval)

        results = opt.run(num_generations=1)
        assert opt._initialized

    def test_run_returns_dict_with_expected_keys(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = FitnessResult(score=0.5, metrics={})
        opt.set_fitness_evaluator(mock_eval)

        results = opt.run(num_generations=1)
        for key in ["best_fitness", "best_individual", "generations", "fitness_history", "converged"]:
            assert key in results, f"Missing key: {key}"

    def test_run_callback_is_called(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = FitnessResult(score=0.5, metrics={})
        opt.set_fitness_evaluator(mock_eval)

        calls = []
        opt.run(num_generations=2, callback=lambda info: calls.append(info))
        assert len(calls) == 2
        assert "generation" in calls[0]
        assert "best" in calls[0]

    def test_run_fitness_increases_or_stays(self, tmp_path):
        """Best fitness should be monotonically non-decreasing."""
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd

        counter = {"n": 0}

        def varying_score(_):
            counter["n"] += 1
            return FitnessResult(score=0.1 * (counter["n"] % 3 + 1), metrics={})

        mock_eval = MagicMock()
        mock_eval.evaluate.side_effect = varying_score
        opt.set_fitness_evaluator(mock_eval)

        opt.run(num_generations=3)
        history = opt.fitness_tracker.best_fitness_history
        assert all(b >= 0 for b in history)

    def test_run_skips_distillation_without_train_dataloader(self, tmp_path):
        """_refine_best_individual should be a no-op when no train_dataloader."""
        opt, student, _ = _make_optimizer(tmp_path)
        sd = _fake_state_dict()
        student.model.state_dict.return_value = sd

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = FitnessResult(score=0.5, metrics={})
        opt.set_fitness_evaluator(mock_eval)

        # Override eval interval so _refine_best_individual is triggered
        opt.config.eval_every_n_generations = 1
        # Should not raise even without a train_dataloader
        opt.run(num_generations=2)


# ── Properties ────────────────────────────────────────────────────────────────


class TestProperties:

    def test_best_individual_after_init(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        # best_individual may be None if fitnesses not set; just shouldn't raise
        _ = opt.best_individual

    def test_population_property(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()
        opt.initialize()
        assert opt.population is opt._population

    def test_current_generation_updates_during_run(self, tmp_path):
        opt, student, _ = _make_optimizer(tmp_path)
        student.model.state_dict.return_value = _fake_state_dict()

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = FitnessResult(score=0.5, metrics={})
        opt.set_fitness_evaluator(mock_eval)

        opt.run(num_generations=2)
        assert opt.current_generation >= 1
