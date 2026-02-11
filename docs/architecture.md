# Architecture Overview

Genesis is built with a modular architecture that separates concerns and enables flexible experimentation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EvolutionaryOptimizer                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      Configuration                           │    │
│  │  GenesisConfig │ GeneticConfig │ DistillationConfig │ etc.  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│         ┌──────────────────────┼──────────────────────┐             │
│         ▼                      ▼                      ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │   Teacher   │      │ Population  │      │   Student   │         │
│  │   Model     │      │  Manager    │      │   Model     │         │
│  │  (GPU 0)    │      │             │      │  (GPU 1)    │         │
│  └─────────────┘      └─────────────┘      └─────────────┘         │
│         │                    │                    │                  │
│         │                    ▼                    │                  │
│         │           ┌─────────────┐               │                  │
│         │           │  Genetics   │               │                  │
│         │           │  - SLERP    │               │                  │
│         │           │  - Mutation │               │                  │
│         │           │  - Select   │               │                  │
│         │           └─────────────┘               │                  │
│         │                    │                    │                  │
│         └────────────────────┼────────────────────┘                  │
│                              ▼                                       │
│                    ┌─────────────────┐                              │
│                    │   Distillation  │                              │
│                    │    Trainer      │                              │
│                    └─────────────────┘                              │
│                              │                                       │
│                              ▼                                       │
│                    ┌─────────────────┐                              │
│                    │     Pruner      │                              │
│                    └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. EvolutionaryOptimizer

The main orchestrator that coordinates all components:

```python
class EvolutionaryOptimizer:
    def __init__(self, config, teacher_model, student_model, ...)
    def initialize(self)
    def run(self, num_generations) -> dict
    def prune_model(self, target_sparsity) -> dict
    def distill(self, num_steps) -> dict
```

### 2. Population Manager

Manages the population of model variants:

```python
class Population:
    def initialize_from_model(self, model)
    def evaluate(self, fitness_fn)
    def evolve(self)  # Selection, crossover, mutation

class Individual:
    state_dict: dict[str, Tensor]
    fitness: float
    generation: int
    parent_ids: list[str]
```

### 3. Genetics Module

Implements genetic operations:

```python
# SLERP interpolation for weight blending
def slerp(t, v0, v1) -> Tensor

# Crossover between parents
def crossover(parent1_state, parent2_state, method="slerp") -> dict

# Mutation with adaptive rates
def mutate(state_dict, mutation_rate, mutation_scale) -> dict
```

### 4. Model Wrappers

#### TeacherModel
- Runs on GPU 0 (by default)
- Frozen during training
- Provides soft targets for distillation

#### StudentModel
- Runs on GPU 1 (by default)
- Supports LoRA adapters
- Trainable weights evolved by genetic algorithm

### 5. Knowledge Distillation

```python
class KDLoss:
    def forward(self, student_logits, teacher_logits, hard_labels=None):
        # KL divergence + optional hard label loss
        # Optional feature distillation
        return {"total_loss": ..., "kd_loss": ..., "hard_loss": ...}

class DistillationTrainer:
    def train(self, num_steps)
    def evaluate(self) -> dict
```

### 6. Pruning Module

```python
class Pruner:
    def prune(self, sparsity) -> dict
    def apply_masks(self)  # Reapply after optimizer step

class SaliencyCalculator:
    def compute(self, method="magnitude") -> dict[str, Tensor]
```

## Data Flow

### Evolution Loop

```
1. Initialize population from base model
   │
2. For each generation:
   │
   ├─► Evaluate fitness of all individuals
   │   │
   │   └─► Load state → Forward pass → Compute metrics
   │
   ├─► Select best individuals (elitism)
   │
   ├─► Create offspring via crossover + mutation
   │   │
   │   ├─► SLERP interpolation of weights
   │   └─► Gaussian noise mutation
   │
   └─► Optional: Refine best with distillation

3. Save best evolved model
```

### Distillation Flow

```
Input Batch
    │
    ├──────────────────────┐
    ▼                      ▼
Teacher Model          Student Model
(GPU 0, frozen)        (GPU 1, training)
    │                      │
    ▼                      ▼
Teacher Logits         Student Logits
    │                      │
    └──────────┬───────────┘
               │
               ▼
         KD Loss
    (KL Div + Hard Loss)
               │
               ▼
      Backprop & Update
```

## Memory Management

### Dual-GPU Strategy

```
GPU 0 (Teacher):
├── Full model weights (frozen)
├── Inference only
└── No gradient computation

GPU 1 (Student):
├── Base model + LoRA adapters
├── Gradient computation
└── Optimizer states
```

### Memory Optimization

1. **LoRA**: Only train low-rank adapters (< 1% of parameters)
2. **Gradient Checkpointing**: Trade compute for memory
3. **Mixed Precision**: FP16/BF16 for reduced memory
4. **Quantization**: 8-bit teacher model option

## Extensibility

### Custom Fitness Evaluator

```python
from genesis.core.fitness import FitnessEvaluator, FitnessResult

class CustomFitness(FitnessEvaluator):
    def evaluate(self, model, state_dict=None):
        # Load state
        if state_dict:
            model.load_state_dict(state_dict)

        # Your evaluation logic
        score = compute_custom_score(model)

        return FitnessResult(
            score=score,
            metrics={"custom_metric": score}
        )
```

### Custom Selection Strategy

```python
from genesis.core.selection import SelectionStrategy

class CustomSelection(SelectionStrategy):
    def select(self, population, num_to_select):
        # Your selection logic
        return selected_individuals
```

## Configuration Hierarchy

```yaml
GenesisConfig
├── project_name, output_dir, seed
├── teacher_model, student_model
├── teacher_device, student_device
│
├── GeneticConfig
│   ├── population_size, generations
│   ├── mutation_rate, crossover_rate
│   └── elite_size, tournament_size
│
├── DistillationConfig
│   ├── temperature, alpha
│   ├── learning_rate, max_steps
│   └── use_feature_distillation
│
├── PruningConfig
│   ├── target_sparsity
│   ├── pruning_method
│   └── structured
│
└── LoRAConfig
    ├── r, lora_alpha
    └── target_modules
```
