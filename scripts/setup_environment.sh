#!/bin/bash
# Genesis Environment Setup Script

set -e

echo "=========================================="
echo "  Genesis AI Evolution Laboratory Setup"
echo "=========================================="
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

echo "Python version: $python_version"

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 9 ]); then
    echo "Error: Python 3.9+ is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (detect CUDA)
echo "Detecting CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Genesis in development mode
echo "Installing Genesis..."
pip install -e .

# Install optional dependencies
read -p "Install TTS dependencies (librosa, soundfile)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install librosa soundfile pyworld
fi

read -p "Install development dependencies (pytest, black, ruff)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install pytest pytest-cov black ruff mypy
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p outputs checkpoints logs data

# Verify installation
echo
echo "Verifying installation..."
python -c "
import genesis
print(f'Genesis version: {genesis.__version__}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

from genesis import EvolutionaryOptimizer, GenesisConfig
print('All imports successful!')
"

echo
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo
echo "To run tests:"
echo "  pytest tests/"
echo
echo "To start an experiment:"
echo "  python experiments/llm_medical/run_evolution.py --dry-run"
