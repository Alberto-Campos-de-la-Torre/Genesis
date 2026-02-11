# Genesis Documentation

Welcome to the Genesis AI Evolution Laboratory documentation. Genesis is a framework for creating efficient AI models using evolutionary algorithms, knowledge distillation, and pruning.

## Overview

Genesis combines three powerful techniques to create optimized AI models:

1. **Evolutionary Algorithms**: Evolve populations of model weights using genetic operations like SLERP-based crossover and adaptive mutation.

2. **Knowledge Distillation**: Transfer knowledge from large teacher models to smaller, more efficient student models.

3. **Model Pruning**: Remove unnecessary weights while maintaining model performance.

## Getting Started

- [Installation](installation.md) - Set up Genesis on your system
- [Quick Start](quickstart.md) - Run your first evolution experiment
- [Architecture](architecture.md) - Understand the system design

## API Reference

- [Core Module](api/core.md) - Evolutionary components
- [Models](api/models.md) - Teacher, Student, and LoRA management
- [Distillation](api/distillation.md) - Knowledge distillation
- [TTS](api/tts.md) - Text-to-Speech evolution

## Tutorials

- [Medical LLM Evolution](tutorials/medical_llm.md) - Train a medical QA model
- [TTS Voice Evolution](tutorials/tts_evolution.md) - Evolve voice characteristics

## Key Features

### Dual-GPU Architecture

Genesis is designed for dual-GPU systems, running the teacher model on GPU 0 and the student model on GPU 1 for maximum efficiency.

```
GPU 0: Teacher Model (frozen, inference only)
GPU 1: Student Model (trained, evolved)
```

### SLERP Crossover

Unlike traditional linear interpolation, Genesis uses Spherical Linear Interpolation (SLERP) for weight crossover, which better preserves the magnitude of neural network weights.

### LoRA Integration

Full support for Low-Rank Adaptation (LoRA) enables efficient fine-tuning of large models with minimal memory overhead.

### Extensible Fitness Functions

Custom fitness evaluators can be implemented by extending the `FitnessEvaluator` base class:

```python
from genesis.core.fitness import FitnessEvaluator, FitnessResult

class MyFitness(FitnessEvaluator):
    def evaluate(self, model, state_dict=None):
        # Your evaluation logic
        return FitnessResult(score=0.95, metrics={"accuracy": 0.95})
```

## Support

- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share ideas

## License

Genesis is released under the MIT License.
