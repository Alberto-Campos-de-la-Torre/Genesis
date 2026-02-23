"""Distillation training loop with dual-GPU support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import logging

from genesis.distillation.kd_loss import KDLoss
from genesis.config.hardware import synchronize, empty_cache

if TYPE_CHECKING:
    from genesis.models.teacher import TeacherModel
    from genesis.models.student import StudentModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for distillation training."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 1000
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    mixed_precision: str = "fp16"  # fp16, bf16, fp32

    # Distillation settings
    temperature: float = 4.0
    alpha: float = 0.5

    # Checkpointing
    output_dir: str = "./outputs"
    save_total_limit: int = 3


class DistillationTrainer:
    """
    Trainer for knowledge distillation with dual-GPU support.

    Manages teacher model on GPU 0 and student model on GPU 1,
    handling cross-device data transfer efficiently.
    """

    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        kd_loss: Optional[KDLoss] = None,
    ):
        """
        Initialize the distillation trainer.

        Args:
            teacher: Teacher model wrapper
            student: Student model wrapper
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            config: Training configuration
            kd_loss: Optional custom KD loss module
        """
        self.teacher = teacher
        self.student = student
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()

        # Initialize KD loss
        self.kd_loss = kd_loss or KDLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
        )

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = None
        if self.config.mixed_precision in ["fp16", "bf16"]:
            self.scaler = torch.amp.GradScaler("cuda")

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.training_logs: list[dict[str, Any]] = []

    def _create_optimizer(self) -> AdamW:
        """Create optimizer for student model."""
        # Get trainable parameters
        params = [p for p in self.student.model.parameters() if p.requires_grad]

        return AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=1e-7,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps],
        )

    def train(
        self,
        num_steps: Optional[int] = None,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Any]:
        """
        Run distillation training.

        Args:
            num_steps: Optional override for number of training steps
            callback: Optional callback function called after each step

        Returns:
            Training results dictionary
        """
        num_steps = num_steps or self.config.max_steps
        self.student.model.train()

        # Ensure teacher is in eval mode
        self.teacher.model.eval()

        progress_bar = tqdm(total=num_steps, desc="Training")
        accumulated_loss = 0.0
        accumulated_steps = 0
        log_window_loss = 0.0   # Sum of per-optimizer-step average losses since last log
        log_window_steps = 0    # Number of optimizer steps since last log

        data_iter = iter(self.train_dataloader)

        while self.global_step < num_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
                self.epoch += 1

            # Training step
            step_results = self._training_step(batch)
            accumulated_loss += step_results["loss"]
            accumulated_steps += 1

            # Gradient accumulation
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                self._optimization_step()
                self.global_step += 1

                # Record the average loss for this optimizer step, then reset
                log_window_loss += accumulated_loss / self.config.gradient_accumulation_steps
                log_window_steps += 1
                accumulated_loss = 0.0
                accumulated_steps = 0

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = log_window_loss / log_window_steps
                    log_entry = {
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        **step_results,
                    }
                    self.training_logs.append(log_entry)
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}")
                    log_window_loss = 0.0
                    log_window_steps = 0

                # Evaluation
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_results = self.evaluate()
                    if eval_results["loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_results["loss"]
                        self._save_checkpoint("best")

                # Callback
                if callback:
                    callback(step_results)

                progress_bar.update(1)

        progress_bar.close()

        return {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "training_logs": self.training_logs,
        }

    def _training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch: Input batch

        Returns:
            Step results including loss components
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids)

        # Get teacher outputs (no gradients needed)
        with torch.no_grad():
            teacher_outputs = self.teacher.forward(
                input_ids,
                attention_mask,
                output_hidden_states=self.kd_loss.use_feature_distillation,
            )
            teacher_logits = teacher_outputs["logits"]

        # Forward through student
        amp_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.scaler is not None):
            student_outputs = self.student.forward(
                input_ids,
                attention_mask,
                output_hidden_states=self.kd_loss.use_feature_distillation,
            )
            student_logits = student_outputs["logits"]

            # Move teacher logits to student device
            teacher_logits_student_device = teacher_logits.to(self.student.device)

            # Compute loss
            loss_dict = self.kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits_student_device,
                hard_labels=labels.to(self.student.device),
                student_hidden_states=student_outputs.get("hidden_states"),
                teacher_hidden_states=self._transfer_hidden_states(
                    teacher_outputs.get("hidden_states")
                ),
                attention_mask=attention_mask.to(self.student.device) if attention_mask is not None else None,
            )

            loss = loss_dict["total_loss"] / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "kd_loss": loss_dict.get("kd_loss", torch.tensor(0)).item(),
            "hard_loss": loss_dict.get("hard_loss", torch.tensor(0)).item(),
        }

    def _transfer_hidden_states(
        self,
        hidden_states: Optional[tuple[torch.Tensor, ...]],
    ) -> Optional[tuple[torch.Tensor, ...]]:
        """Transfer hidden states to student device."""
        if hidden_states is None:
            return None
        return tuple(h.to(self.student.device) for h in hidden_states)

    def _optimization_step(self) -> None:
        """Perform optimization step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student.model.parameters(),
                self.config.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.student.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the student model.

        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        self.student.model.eval()
        total_loss = 0.0
        total_kd_loss = 0.0
        total_hard_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels", input_ids)

            # Teacher forward
            teacher_outputs = self.teacher.forward(input_ids, attention_mask)
            teacher_logits = teacher_outputs["logits"].to(self.student.device)

            # Student forward
            student_outputs = self.student.forward(input_ids, attention_mask)
            student_logits = student_outputs["logits"]

            # Compute loss
            loss_dict = self.kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                hard_labels=labels.to(self.student.device),
            )

            total_loss += loss_dict["total_loss"].item()
            total_kd_loss += loss_dict.get("kd_loss", torch.tensor(0)).item()
            total_hard_loss += loss_dict.get("hard_loss", torch.tensor(0)).item()
            num_batches += 1

        self.student.model.train()

        results = {
            "loss": total_loss / num_batches,
            "kd_loss": total_kd_loss / num_batches,
            "hard_loss": total_hard_loss / num_batches,
        }

        logger.info(f"Evaluation: {results}")
        return results

    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        import os

        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save student model
        self.student.save(checkpoint_dir)

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_eval_loss": self.best_eval_loss,
            },
            os.path.join(checkpoint_dir, "training_state.pt"),
        )

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load training checkpoint."""
        import os

        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, weights_only=False)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.scheduler.load_state_dict(state["scheduler_state"])
            self.best_eval_loss = state["best_eval_loss"]

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

    def get_training_state(self) -> dict[str, Any]:
        """Get current training state."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "current_lr": self.scheduler.get_last_lr()[0],
        }
