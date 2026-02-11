# Distillation Module API Reference

The distillation module provides knowledge distillation functionality.

## KD Loss Functions

### `genesis.distillation.kd_loss`

#### `kl_divergence_loss(student_logits, teacher_logits, temperature=4.0, reduction="batchmean")`

Compute KL divergence loss between student and teacher logits.

```python
from genesis.distillation.kd_loss import kl_divergence_loss

loss = kl_divergence_loss(
    student_logits,      # [batch, seq_len, vocab]
    teacher_logits,      # [batch, seq_len, vocab]
    temperature=4.0,     # Softmax temperature
    reduction="batchmean"
)
```

#### `soft_target_loss(student_logits, teacher_logits, temperature=4.0, attention_mask=None)`

Compute soft target cross-entropy loss.

#### `feature_distillation_loss(student_hidden, teacher_hidden, projection=None, loss_type="mse")`

Compute feature distillation loss between hidden states.

```python
from genesis.distillation.kd_loss import feature_distillation_loss

loss = feature_distillation_loss(
    student_hidden,    # [batch, seq_len, hidden_dim]
    teacher_hidden,    # [batch, seq_len, hidden_dim]
    projection=None,   # Optional projection layer
    loss_type="mse"    # "mse", "cosine", or "l1"
)
```

### KDLoss Class

Comprehensive knowledge distillation loss module.

```python
from genesis.distillation import KDLoss

kd_loss = KDLoss(
    temperature=4.0,
    alpha=0.5,                        # Weight for KD vs hard loss
    use_feature_distillation=True,
    feature_weight=0.1,
    feature_layers=[-1, -2, -3],
    use_attention_distillation=False,
    attention_weight=0.1,
)

losses = kd_loss(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    hard_labels=labels,
    student_hidden_states=student_hidden,
    teacher_hidden_states=teacher_hidden,
    attention_mask=attention_mask,
)

total_loss = losses["total_loss"]
kd_loss_value = losses["kd_loss"]
hard_loss_value = losses["hard_loss"]
```

### ProgressiveKDLoss

Knowledge distillation with curriculum learning.

```python
from genesis.distillation.kd_loss import ProgressiveKDLoss

progressive_loss = ProgressiveKDLoss(
    initial_temperature=10.0,
    final_temperature=1.0,
    initial_alpha=0.9,
    final_alpha=0.1,
    total_steps=10000,
)

# Update temperature and alpha each step
progressive_loss.step()
```

## Distillation Trainer

### `genesis.distillation.trainer.DistillationTrainer`

Training loop with dual-GPU synchronization.

```python
from genesis.distillation import DistillationTrainer, TrainingConfig

config = TrainingConfig(
    learning_rate=2e-5,
    weight_decay=0.01,
    max_steps=1000,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    mixed_precision="fp16",
    temperature=4.0,
    alpha=0.5,
    output_dir="./outputs",
)

trainer = DistillationTrainer(
    teacher=teacher_model,
    student=student_model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    config=config,
    kd_loss=custom_kd_loss,  # Optional custom loss
)

# Run training
results = trainer.train(
    num_steps=1000,
    callback=lambda r: print(f"Step {r['step']}: loss={r['loss']:.4f}")
)

# Evaluate
eval_results = trainer.evaluate()

# Save/load checkpoint
trainer._save_checkpoint("best")
trainer.load_checkpoint("./checkpoints/checkpoint-best")

# Get training state
state = trainer.get_training_state()
```

### TrainingConfig

Configuration for distillation training.

```python
@dataclass
class TrainingConfig:
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
    temperature: float = 4.0
    alpha: float = 0.5
    output_dir: str = "./outputs"
    save_total_limit: int = 3
```
