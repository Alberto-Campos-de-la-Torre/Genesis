# Installation Guide

This guide walks you through installing Genesis and its dependencies.

## Requirements

### System Requirements

- **Python**: 3.9 or higher
- **CUDA**: 11.7 or higher (for GPU support)
- **Operating System**: Linux (recommended), macOS, or Windows

### Hardware Requirements

| Configuration | GPUs | VRAM | Use Case |
|--------------|------|------|----------|
| Minimum | 1x | 16GB | Small models, testing |
| Recommended | 2x | 24GB each | Full dual-GPU mode |
| Optimal | 2x A100/H100 | 40-80GB each | Large models |

## Installation Methods

### Method 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/genesis-ai/genesis.git
cd genesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Genesis in development mode
pip install -e .
```

### Method 2: Using pip

```bash
pip install genesis-ai
```

### Method 3: Using conda

```bash
conda create -n genesis python=3.10
conda activate genesis
pip install -r requirements.txt
pip install -e .
```

## Dependencies

### Core Dependencies

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
datasets>=2.16.0
```

### Optional Dependencies

For TTS experiments:
```bash
pip install librosa soundfile pyworld
```

For distributed training:
```bash
pip install ray[tune] deepspeed
```

For documentation:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

## Verifying Installation

Run the following to verify your installation:

```python
import genesis
print(f"Genesis version: {genesis.__version__}")

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Test basic import
from genesis import EvolutionaryOptimizer, GenesisConfig
print("Installation successful!")
```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce `batch_size` in config
2. Enable gradient checkpointing
3. Use smaller population size
4. Enable 8-bit quantization for teacher model

### Import Errors

Ensure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Permission Errors

On Linux, you may need to:
```bash
chmod +x scripts/*.sh
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first experiment
- [Architecture Overview](architecture.md) - Understand the system
