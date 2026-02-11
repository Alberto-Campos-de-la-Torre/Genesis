# Quick Start Guide

Get started with Genesis in just a few minutes.

## Basic Usage

### 1. Create Configuration

```python
from genesis import GenesisConfig

config = GenesisConfig(
    project_name="my_experiment",
    teacher_model="gpt2",  # Use a small model for testing
    use_lora=True,
    genetic=dict(
        population_size=10,
        generations=20,
        mutation_rate=0.1,
    ),
)
```

### 2. Prepare Data

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from genesis.data.datasets import DatasetLoader

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create dataloader
dataset_loader = DatasetLoader(
    dataset_name="wikitext",
    tokenizer=tokenizer,
    max_length=256,
)

train_loader = dataset_loader.get_dataloader(batch_size=8)
```

### 3. Run Evolution

```python
from genesis import EvolutionaryOptimizer

optimizer = EvolutionaryOptimizer(
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=train_loader,  # Use same for demo
)

# Run evolution
results = optimizer.run()

print(f"Best fitness: {results['best_fitness']:.4f}")
print(f"Generations: {results['generations']}")
```

### 4. Save and Use the Model

```python
# The best model is automatically saved
# Load it for inference
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./outputs/best_model")
```

## Configuration Options

### Genetic Algorithm

```python
config = GenesisConfig(
    genetic=dict(
        population_size=20,      # Number of individuals
        generations=50,          # Evolution iterations
        mutation_rate=0.1,       # Mutation probability
        crossover_rate=0.7,      # Crossover probability
        elite_size=2,            # Top individuals to preserve
        tournament_size=3,       # Tournament selection size
        slerp_ratio=0.5,         # SLERP interpolation ratio
    ),
)
```

### Knowledge Distillation

```python
config = GenesisConfig(
    distillation=dict(
        temperature=4.0,         # Softmax temperature
        alpha=0.5,               # Distillation vs hard loss weight
        learning_rate=2e-5,      # Learning rate
        max_steps=1000,          # Training steps
    ),
)
```

### Pruning

```python
config = GenesisConfig(
    pruning=dict(
        target_sparsity=0.3,     # Remove 30% of weights
        pruning_method="magnitude",  # magnitude, gradient, taylor
        structured=False,        # Element-wise vs structured
    ),
)
```

## Full Example

```python
import torch
from transformers import AutoTokenizer
from genesis import EvolutionaryOptimizer, GenesisConfig
from genesis.data.datasets import create_dataloader

# Configuration
config = GenesisConfig(
    project_name="quickstart_demo",
    teacher_model="gpt2",
    use_lora=True,
    teacher_device="cuda:0",
    student_device="cuda:0",  # Use same GPU if only one available
    genetic=dict(
        population_size=8,
        generations=10,
        mutation_rate=0.1,
    ),
    distillation=dict(
        temperature=4.0,
        max_steps=100,
    ),
)

# Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create dataloader
train_loader = create_dataloader(
    dataset_name="wikitext",
    tokenizer=tokenizer,
    batch_size=4,
    split="train",
    max_samples=500,
)

# Create and run optimizer
optimizer = EvolutionaryOptimizer(
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=train_loader,
)

results = optimizer.run()

# Optional: Prune the model
prune_stats = optimizer.prune_model(target_sparsity=0.2)

# Optional: Final distillation
distill_results = optimizer.distill(num_steps=100)

print("Evolution complete!")
print(f"Best fitness: {results['best_fitness']:.4f}")
print(f"Final sparsity: {prune_stats['actual_sparsity']:.2%}")
```

## Next Steps

- [Architecture Overview](architecture.md) - Deep dive into the system
- [Medical LLM Tutorial](tutorials/medical_llm.md) - Real-world example
- [API Reference](api/core.md) - Complete API documentation
