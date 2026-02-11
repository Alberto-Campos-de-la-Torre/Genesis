# Models Module API Reference

The models module provides wrappers for teacher and student models.

## TeacherModel

### `genesis.models.teacher.TeacherModel`

Wrapper for teacher model in knowledge distillation.

```python
from genesis.models import TeacherModel

teacher = TeacherModel(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
    dtype=torch.float16,
    load_in_8bit=False,
    load_in_4bit=False,
)

# Load the model
teacher.load()

# Forward pass
outputs = teacher.forward(input_ids, attention_mask)
logits = outputs["logits"]

# Get soft targets
soft_targets = teacher.get_soft_targets(input_ids, temperature=4.0)

# Get hidden states
hidden_states = teacher.get_hidden_states(input_ids, layer_indices=[-1, -2])

# Generate text
generated = teacher.generate(input_ids, max_new_tokens=100)

# Cleanup
teacher.unload()
```

### Methods

#### `load()`
Load the model and tokenizer.

#### `forward(input_ids, attention_mask=None, output_hidden_states=False)`
Forward pass through the model.

#### `get_soft_targets(input_ids, attention_mask=None, temperature=1.0)`
Get soft probability targets for distillation.

#### `get_hidden_states(input_ids, attention_mask=None, layer_indices=None)`
Extract hidden states from specified layers.

#### `generate(input_ids, attention_mask=None, max_new_tokens=100, **kwargs)`
Generate text using the model.

## StudentModel

### `genesis.models.student.StudentModel`

Wrapper for student model with LoRA support.

```python
from genesis.models import StudentModel
from genesis.models.lora_manager import LoRAConfig

lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

student = StudentModel(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    device="cuda:1",
    dtype=torch.float16,
    use_lora=True,
    lora_config=lora_config,
)

student.load()

# Forward pass
outputs = student.forward(input_ids, attention_mask, labels=labels)
loss = outputs["loss"]

# Training step with distillation
losses = student.train_step(
    input_ids,
    attention_mask,
    teacher_logits=teacher_logits,
    labels=labels,
    temperature=4.0,
    alpha=0.5,
)

# Get/set state dict
state = student.get_state_dict(lora_only=True)
student.load_state_dict(state)

# Save model
student.save("./outputs/model", merge_lora=False)
```

## LoRAManager

### `genesis.models.lora_manager.LoRAManager`

Manager for LoRA adapters supporting evolutionary operations.

```python
from genesis.models import LoRAManager, LoRAConfig

config = LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

manager = LoRAManager(model, config)

# Get LoRA weights
lora_state = manager.get_lora_state_dict()

# Set LoRA weights
manager.set_lora_state_dict(lora_state)

# Interpolate between two LoRA states
blended = manager.interpolate_lora(state1, state2, ratio=0.5)

# Merge multiple LoRA states
merged = manager.merge_multiple_lora([state1, state2, state3], weights=[0.5, 0.3, 0.2])

# Compute similarity
similarity = manager.compute_lora_similarity(state1, state2)

# Add noise for mutation
noisy = manager.add_noise_to_lora(lora_state, noise_scale=0.01)

# Save/load LoRA weights
manager.save_lora("./lora_weights.pt")
manager.load_lora("./lora_weights.pt")
```

### LoRAConfig

Configuration for LoRA adapters.

```python
@dataclass
class LoRAConfig:
    r: int = 16                    # LoRA rank
    lora_alpha: int = 32           # LoRA alpha
    lora_dropout: float = 0.05     # Dropout probability
    target_modules: list = ["q_proj", "v_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
```

### Helper Function

```python
from genesis.models.lora_manager import create_lora_config_for_model

# Create optimized config for specific model types
config = create_lora_config_for_model("llama", r=16, alpha=32)
# Returns config with target_modules optimized for Llama architecture
```
