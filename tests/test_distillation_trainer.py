"""Tests for DistillationTrainer training loop."""

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from genesis.distillation.trainer import DistillationTrainer, TrainingConfig
from genesis.distillation.kd_loss import KDLoss


# ── Mock teacher / student ───────────────────────────────────────────────────────

VOCAB = 50
SEQ = 8
HIDDEN = 32


class _StudentNN(nn.Module):
    """Underlying nn.Module for the mock student."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.proj = nn.Linear(HIDDEN, VOCAB)


class MockStudent:
    """Thin wrapper that matches the StudentModel interface used by the trainer."""

    def __init__(self):
        self._model = _StudentNN()
        self.device = "cpu"

    @property
    def model(self) -> nn.Module:
        return self._model

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                labels=None, **_):
        input_ids = input_ids.to(self.device)
        hidden = self._model.embed(input_ids)
        logits = self._model.proj(hidden)
        result: dict = {"logits": logits}
        if output_hidden_states:
            result["hidden_states"] = (hidden,)
        return result

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(path, "model.pt"))


class MockTeacher:
    """Thin wrapper that matches the TeacherModel interface used by the trainer."""

    def __init__(self):
        self.model = nn.Linear(1, 1)  # only needs .eval()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                **_):
        batch, seq = input_ids.shape[:2]
        logits = torch.randn(batch, seq, VOCAB)
        result: dict = {"logits": logits}
        if output_hidden_states:
            result["hidden_states"] = (torch.randn(batch, seq, HIDDEN),)
        return result


def _make_dataloader(num_samples: int = 8, batch_size: int = 4) -> DataLoader:
    ids = torch.randint(0, VOCAB, (num_samples, SEQ))
    mask = torch.ones_like(ids)
    ds = TensorDataset(ids, mask, ids.clone())  # labels = input_ids

    def collate(batch):
        ids_b, mask_b, labels_b = zip(*batch)
        return {
            "input_ids": torch.stack(ids_b),
            "attention_mask": torch.stack(mask_b),
            "labels": torch.stack(labels_b),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


def _fp32_config(**kwargs) -> TrainingConfig:
    """TrainingConfig with fp32 mixed precision (avoids GradScaler on CPU)."""
    defaults = dict(
        learning_rate=1e-3,
        max_steps=4,
        warmup_steps=1,
        gradient_accumulation_steps=1,
        logging_steps=2,
        eval_steps=4,
        mixed_precision="fp32",
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


# ── Initialisation ───────────────────────────────────────────────────────────────

class TestDistillationTrainerInit:

    def test_creates_optimizer(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(),
        )
        assert trainer.optimizer is not None

    def test_creates_scheduler(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(),
        )
        assert trainer.scheduler is not None

    def test_scaler_none_for_fp32(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(mixed_precision="fp32"),
        )
        assert trainer.scaler is None

    def test_default_kd_loss_created(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(),
        )
        assert isinstance(trainer.kd_loss, KDLoss)

    def test_custom_kd_loss_accepted(self):
        custom_loss = KDLoss(temperature=2.0, alpha=0.3)
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(),
            kd_loss=custom_loss,
        )
        assert trainer.kd_loss is custom_loss

    def test_initial_global_step_zero(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(),
        )
        assert trainer.global_step == 0


# ── Training loop ────────────────────────────────────────────────────────────────

class TestDistillationTrainerLoop:

    def _trainer(self, **cfg_kwargs):
        return DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(num_samples=16, batch_size=4),
            config=_fp32_config(**cfg_kwargs),
        )

    def test_train_advances_global_step(self):
        trainer = self._trainer(max_steps=4, gradient_accumulation_steps=1)
        results = trainer.train()
        assert results["global_step"] == 4

    def test_train_returns_dict(self):
        trainer = self._trainer(max_steps=2)
        results = trainer.train()
        assert "global_step" in results
        assert "best_eval_loss" in results
        assert "training_logs" in results

    def test_train_logs_are_recorded(self):
        trainer = self._trainer(max_steps=4, logging_steps=2,
                                gradient_accumulation_steps=1)
        results = trainer.train()
        # Should have logged at steps 2 and 4
        assert len(results["training_logs"]) == 2

    def test_log_entry_has_required_keys(self):
        trainer = self._trainer(max_steps=2, logging_steps=1,
                                gradient_accumulation_steps=1)
        results = trainer.train()
        log = results["training_logs"][0]
        assert "step" in log
        assert "loss" in log
        assert "lr" in log

    def test_logged_loss_is_finite(self):
        trainer = self._trainer(max_steps=4, logging_steps=2,
                                gradient_accumulation_steps=1)
        results = trainer.train()
        for entry in results["training_logs"]:
            assert torch.isfinite(torch.tensor(entry["loss"]))

    def test_gradient_accumulation_reduces_steps(self):
        trainer = self._trainer(max_steps=4, gradient_accumulation_steps=2)
        results = trainer.train()
        # 4 optimizer steps — each needs 2 forward passes
        assert results["global_step"] == 4

    def test_num_steps_override(self):
        trainer = self._trainer(max_steps=100)
        results = trainer.train(num_steps=3)
        assert results["global_step"] == 3

    def test_callback_is_called(self):
        calls = []
        trainer = self._trainer(max_steps=3, gradient_accumulation_steps=1)
        trainer.train(callback=lambda r: calls.append(r))
        assert len(calls) == 3


# ── Log window loss correctness ──────────────────────────────────────────────────

class TestLogWindowLoss:
    """
    Verify the accumulated_loss fix: the logged average must reflect only the
    logging window, not all steps since training started.
    """

    def test_log_loss_resets_between_windows(self):
        """Log entries at different windows must differ (not ever-growing)."""
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(num_samples=32, batch_size=4),
            config=_fp32_config(
                max_steps=6,
                logging_steps=2,
                gradient_accumulation_steps=1,
            ),
        )
        results = trainer.train()
        logs = results["training_logs"]
        assert len(logs) == 3

        # If loss were accumulating without reset, each entry's loss would be
        # monotonically larger (roughly). Check that they are distinct and not
        # simply N × previous.
        losses = [e["loss"] for e in logs]
        # The second loss should not be ≈ 2× the first (accumulation symptom)
        assert not (abs(losses[1] - 2 * losses[0]) < 0.01 * losses[0])


# ── Evaluation ───────────────────────────────────────────────────────────────────

class TestDistillationTrainerEval:

    def test_evaluate_returns_loss(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            eval_dataloader=_make_dataloader(num_samples=8, batch_size=4),
            config=_fp32_config(max_steps=1),
        )
        results = trainer.evaluate()
        assert "loss" in results
        assert results["loss"] > 0

    def test_evaluate_empty_without_dataloader(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(),
        )
        results = trainer.evaluate()
        assert results == {}

    def test_get_training_state(self):
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(max_steps=2),
        )
        trainer.train()
        state = trainer.get_training_state()
        assert "global_step" in state
        assert "epoch" in state
        assert "best_eval_loss" in state
        assert "current_lr" in state


# ── Checkpoint save / load ────────────────────────────────────────────────────────

class TestDistillationTrainerCheckpoint:

    def test_save_checkpoint(self, tmp_path):
        cfg = _fp32_config(max_steps=3, warmup_steps=1, output_dir=str(tmp_path))
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=cfg,
        )
        trainer.train()
        trainer._save_checkpoint("test")
        ckpt_dir = tmp_path / "checkpoint-test"
        assert (ckpt_dir / "training_state.pt").exists()

    def test_load_checkpoint_restores_step(self, tmp_path):
        cfg = _fp32_config(max_steps=3, output_dir=str(tmp_path),
                           gradient_accumulation_steps=1)
        trainer = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(num_samples=16),
            config=cfg,
        )
        trainer.train()
        trainer._save_checkpoint("restore")

        # Create fresh trainer and load the checkpoint
        trainer2 = DistillationTrainer(
            teacher=MockTeacher(),
            student=MockStudent(),
            train_dataloader=_make_dataloader(),
            config=_fp32_config(output_dir=str(tmp_path)),
        )
        trainer2.load_checkpoint(str(tmp_path / "checkpoint-restore"))
        assert trainer2.global_step == 3
