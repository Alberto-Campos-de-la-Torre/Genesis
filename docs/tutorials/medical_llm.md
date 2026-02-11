# Medical LLM Evolution Tutorial

This tutorial walks you through evolving a medical question-answering model using Genesis.

## Overview

We'll create a specialized medical QA model by:
1. Using a large LLM as teacher (e.g., Llama-2-7B)
2. Evolving LoRA adapters for medical domain expertise
3. Applying knowledge distillation from teacher to student
4. Pruning the final model for efficiency

## Prerequisites

- 2 GPUs with at least 24GB VRAM each
- PubMedQA dataset (automatically downloaded)
- Llama-2-7B model access (HuggingFace token)

## Step 1: Configuration

Create a configuration file `config.yaml`:

```yaml
project_name: "medical_qa_evolution"
output_dir: "./outputs/medical"
checkpoint_dir: "./checkpoints/medical"
seed: 42

teacher_model: "meta-llama/Llama-2-7b-hf"
use_lora: true

teacher_device: "cuda:0"
student_device: "cuda:1"

genetic:
  population_size: 20
  generations: 50
  mutation_rate: 0.1
  crossover_rate: 0.7
  elite_size: 2

distillation:
  temperature: 4.0
  alpha: 0.5
  learning_rate: 2e-5
  max_steps: 1000

lora:
  r: 16
  lora_alpha: 32
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

dataset_name: "pubmed_qa"
max_samples: 5000
eval_samples: 500
```

## Step 2: Load Data

```python
from transformers import AutoTokenizer
from genesis.data.datasets import PubMedQADataset
from torch.utils.data import DataLoader

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Create datasets
train_dataset = PubMedQADataset(
    tokenizer=tokenizer,
    split="train",
    max_length=512,
    max_samples=5000,
)

eval_dataset = PubMedQADataset(
    tokenizer=tokenizer,
    split="train",
    max_length=512,
    max_samples=500,
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)
```

## Step 3: Initialize Models

```python
from genesis.models import TeacherModel, StudentModel
import torch

# Teacher model (frozen, provides soft targets)
teacher = TeacherModel(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
    dtype=torch.float16,
)
teacher.load()

# Student model (evolved with LoRA)
student = StudentModel(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    device="cuda:1",
    dtype=torch.float16,
    use_lora=True,
)
student.load()
```

## Step 4: Create Fitness Evaluator

```python
from genesis.core.fitness import QAFitness

fitness_evaluator = QAFitness(
    dataloader=eval_loader,
    tokenizer=tokenizer,
    device="cuda:1",
    max_samples=500,
    max_new_tokens=50,
)
```

## Step 5: Run Evolution

```python
from genesis import EvolutionaryOptimizer, GenesisConfig

# Load config
config = GenesisConfig.from_yaml("config.yaml")

# Create optimizer
optimizer = EvolutionaryOptimizer(
    config=config,
    teacher_model=teacher,
    student_model=student,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

# Set fitness evaluator
optimizer.set_fitness_evaluator(fitness_evaluator)

# Run evolution with callback
def on_generation(info):
    print(f"Generation {info['generation']}")
    print(f"  Best fitness: {info['best'].fitness:.4f}")
    print(f"  Avg fitness: {info['avg_fitness']:.4f}")

results = optimizer.run(callback=on_generation)
```

## Step 6: Post-Processing

### Pruning

```python
# Prune 30% of weights
prune_stats = optimizer.prune_model(target_sparsity=0.3)
print(f"Actual sparsity: {prune_stats['actual_sparsity']:.2%}")
```

### Final Distillation

```python
# Additional distillation to recover from pruning
distill_results = optimizer.distill(num_steps=500)
print(f"Final loss: {distill_results['best_eval_loss']:.4f}")
```

## Step 7: Evaluation

```python
# Load the best model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./outputs/medical/best_model")
model.to("cuda:0")

# Test with a medical question
question = "What are the common symptoms of diabetes?"
context = "Diabetes mellitus is a metabolic disease..."

input_text = f"Question: {question}\nContext: {context}\nAnswer:"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=100,
    temperature=0.7,
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Expected Results

After 50 generations of evolution:
- Best fitness: ~0.75 (combined exact match + F1)
- Model size reduction: 30% (with pruning)
- Inference speed improvement: ~40%

## Tips for Better Results

1. **Larger Population**: Increase `population_size` for more diversity
2. **Longer Evolution**: More generations allow finer optimization
3. **Adaptive Mutation**: Enable for automatic rate adjustment
4. **Feature Distillation**: Add hidden state matching for better transfer
5. **Multiple Runs**: Average results from multiple seeds

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Enable gradient checkpointing
- Use 8-bit quantization for teacher

### Slow Convergence

- Increase `mutation_rate`
- Use larger `elite_size`
- Check if fitness function is well-calibrated

### Fitness Plateau

- Increase `mutation_scale` temporarily
- Add more diverse data samples
- Try different selection strategies
