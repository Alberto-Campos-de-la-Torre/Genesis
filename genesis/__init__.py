"""
Genesis AI Evolution Laboratory

A framework for creating efficient AI models using evolutionary algorithms,
pruning, and knowledge distillation on dual GPUs.
"""

__version__ = "0.1.0"
__author__ = "Genesis Team"

from genesis.optimizer import EvolutionaryOptimizer
from genesis.config.settings import GenesisConfig
from genesis.config.hardware import HardwareConfig

__all__ = [
    "EvolutionaryOptimizer",
    "GenesisConfig",
    "HardwareConfig",
    "__version__",
]
