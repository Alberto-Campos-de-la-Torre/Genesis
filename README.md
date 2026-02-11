# Genesis AI Evolution Laboratory

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

Genesis is a cutting-edge framework for creating efficient AI models through evolutionary algorithms, knowledge distillation, and model pruning. Designed for dual-GPU acceleration, it evolves neural networks to achieve optimal performance with minimal computational resources.

## Features

- **Evolutionary Optimization**: SLERP-based genetic operations for smooth model weight interpolation
- **Knowledge Distillation**: Transfer knowledge from large teacher models to smaller students
- **Model Pruning**: Intelligent weight pruning with multiple saliency methods
- **Dual-GPU Support**: Parallel execution with teacher on GPU 0 and student on GPU 1
- **LoRA Integration**: Efficient fine-tuning with Low-Rank Adaptation
- **TTS Evolution**: Specialized support for Text-to-Speech style token evolution
- **Extensible Design**: Modular architecture for custom experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/genesis-ai/genesis.git
cd genesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Genesis in development mode
pip install -e .
```

## Quick Start

```python
from genesis import EvolutionaryOptimizer, GenesisConfig

# Create configuration
config = GenesisConfig(
    teacher_model="meta-llama/Llama-2-7b-hf",
    use_lora=True,
    genetic=dict(
        population_size=20,
        generations=50,
        mutation_rate=0.1,
    ),
)

# Create optimizer
optimizer = EvolutionaryOptimizer(
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

# Run evolution
results = optimizer.run()
print(f"Best fitness: {results['best_fitness']}")

# Optional: Prune and distill
optimizer.prune_model(target_sparsity=0.3)
optimizer.distill(num_steps=1000)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Genesis Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   Teacher    │     │  Population  │     │   Student    │ │
│  │   (GPU 0)    │────▶│   Manager    │────▶│   (GPU 1)    │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│         │                    │                    │          │
│         │                    ▼                    │          │
│         │           ┌──────────────┐              │          │
│         │           │   Genetics   │              │          │
│         │           │  (SLERP +    │              │          │
│         │           │  Mutation)   │              │          │
│         │           └──────────────┘              │          │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Knowledge Distillation                   │   │
│  │         (KL Divergence + Feature Matching)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Pruning                            │   │
│  │     (Magnitude / Gradient / Taylor / Structured)     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Experiments

### Medical LLM Evolution

Evolve a medical question-answering model on PubMedQA:

```bash
cd experiments/llm_medical
python run_evolution.py --config config.yaml
```

### TTS Voice Evolution

Evolve TTS style tokens for voice customization:

```bash
cd experiments/tts_voice
python run_evolution.py --config config.yaml
```

## Configuration

Genesis uses YAML configuration files. Key parameters:

```yaml
# Genetic Algorithm
genetic:
  population_size: 20
  generations: 50
  mutation_rate: 0.1
  crossover_rate: 0.7
  elite_size: 2

# Knowledge Distillation
distillation:
  temperature: 4.0
  alpha: 0.5  # Weight for soft vs hard targets

# Pruning
pruning:
  target_sparsity: 0.3
  pruning_method: "magnitude"
```

## Core Components

| Component | Description |
|-----------|-------------|
| `EvolutionaryOptimizer` | Main orchestrator combining all techniques |
| `Population` | Manages population of model variants |
| `Genetics` | SLERP crossover, mutation operations |
| `TeacherModel` | Large model providing soft targets |
| `StudentModel` | Smaller model being optimized |
| `LoRAManager` | Efficient LoRA adapter management |
| `KDLoss` | Knowledge distillation loss functions |
| `Pruner` | Model pruning with various strategies |

## Hardware Requirements

- **Minimum**: 1x GPU with 16GB VRAM
- **Recommended**: 2x GPUs (24GB+ each) for full dual-GPU mode
- **Optimal**: 2x A100/H100 GPUs

## Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## Citation

```bibtex
@software{genesis2024,
  title={Genesis: AI Evolution Laboratory},
  author={Genesis Team},
  year={2024},
  url={https://github.com/genesis-ai/genesis}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- HuggingFace Transformers team
- PyTorch team
- PEFT library developers
