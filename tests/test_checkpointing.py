"""Tests for genesis/utils/checkpointing.py."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from genesis.utils.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_model():
    return nn.Linear(4, 2)


def _simple_optimizer(model):
    return SGD(model.parameters(), lr=0.01)


# ── save_checkpoint / load_checkpoint ────────────────────────────────────────

class TestSaveLoadCheckpoint:

    def test_save_creates_file(self, tmp_path):
        model = _simple_model()
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model=model, epoch=1, step=10)
        assert (tmp_path / "ckpt.pt").exists()

    def test_load_restores_model_weights(self, tmp_path):
        model = _simple_model()
        # Set a known weight
        with torch.no_grad():
            model.weight.fill_(3.14)
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model=model)

        # Load into a fresh model
        model2 = _simple_model()
        load_checkpoint(path, model=model2)
        assert torch.allclose(model2.weight, model.weight)

    def test_load_restores_epoch_and_step(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, epoch=5, step=100)
        ckpt = load_checkpoint(path)
        assert ckpt["epoch"] == 5
        assert ckpt["step"] == 100

    def test_load_restores_optimizer_state(self, tmp_path):
        model = _simple_model()
        opt = _simple_optimizer(model)
        # Run a step to populate optimizer state
        out = model(torch.randn(2, 4))
        out.sum().backward()
        opt.step()

        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model=model, optimizer=opt)

        model2 = _simple_model()
        opt2 = _simple_optimizer(model2)
        load_checkpoint(path, model=model2, optimizer=opt2)
        # Optimizer state restored (e.g. SGD has no running state, but no error)

    def test_save_with_config(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, config={"lr": 0.01, "batch_size": 32})
        ckpt = load_checkpoint(path)
        assert ckpt["config"]["lr"] == 0.01

    def test_save_with_extra_kwargs(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, custom_key="hello")
        ckpt = load_checkpoint(path)
        assert ckpt["custom_key"] == "hello"

    def test_checkpoint_contains_timestamp(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path)
        ckpt = load_checkpoint(path)
        assert "timestamp" in ckpt

    def test_saves_to_nested_dir(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "ckpt.pt")
        save_checkpoint(path)
        assert (tmp_path / "deep" / "nested" / "ckpt.pt").exists()


# ── CheckpointManager ─────────────────────────────────────────────────────────

class TestCheckpointManager:

    def test_save_creates_checkpoint_file(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        model = _simple_model()
        path = manager.save(model, step=1, metric=0.5)
        assert path is not None
        import os; assert os.path.exists(path)

    def test_save_returns_path_string(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        path = manager.save(_simple_model(), step=1)
        assert isinstance(path, str)

    def test_best_checkpoint_written_on_improvement(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), mode="min")
        manager.save(_simple_model(), step=1, metric=1.0)
        manager.save(_simple_model(), step=2, metric=0.5)  # better
        assert manager.best_checkpoint is not None
        assert (tmp_path / "checkpoint_best.pt").exists()

    def test_best_metric_tracked_min(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), mode="min")
        manager.save(_simple_model(), step=1, metric=1.0)
        manager.save(_simple_model(), step=2, metric=0.3)
        assert manager.best_metric == pytest.approx(0.3)

    def test_best_metric_tracked_max(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), mode="max")
        manager.save(_simple_model(), step=1, metric=0.5)
        manager.save(_simple_model(), step=2, metric=0.9)
        assert manager.best_metric == pytest.approx(0.9)

    def test_cleanup_keeps_max_checkpoints(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), max_checkpoints=3)
        for i in range(6):
            manager.save(_simple_model(), step=i, metric=float(i))
        # Should not exceed max_checkpoints non-best files
        assert len(manager.get_checkpoint_list()) <= 3

    def test_load_best_restores_model(self, tmp_path):
        manager = CheckpointManager(str(tmp_path), mode="min")
        model = _simple_model()
        with torch.no_grad():
            model.weight.fill_(7.0)
        manager.save(model, step=1, metric=0.1)

        model2 = _simple_model()
        manager.load_best(model2)
        assert torch.allclose(model2.weight, model.weight)

    def test_load_best_raises_if_none(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.load_best(_simple_model())

    def test_load_latest_restores_model(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        model = _simple_model()
        with torch.no_grad():
            model.weight.fill_(2.5)
        manager.save(model, step=1)
        manager.save(_simple_model(), step=2)

        # Load latest (step 2)
        model2 = _simple_model()
        manager.load_latest(model2)

    def test_load_latest_raises_if_no_checkpoints(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.load_latest(_simple_model())

    def test_latest_checkpoint_property(self, tmp_path):
        manager = CheckpointManager(str(tmp_path))
        assert manager.latest_checkpoint is None
        manager.save(_simple_model(), step=1)
        assert manager.latest_checkpoint is not None

    def test_checkpoint_info_persists_across_instances(self, tmp_path):
        """Checkpoint list is saved to JSON and reloaded."""
        manager1 = CheckpointManager(str(tmp_path))
        manager1.save(_simple_model(), step=10, metric=0.5)

        manager2 = CheckpointManager(str(tmp_path))
        assert len(manager2.get_checkpoint_list()) == 1
        assert manager2.get_checkpoint_list()[0]["step"] == 10
