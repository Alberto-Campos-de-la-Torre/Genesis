"""
Genesis AI Evolution Laboratory

A framework for creating efficient AI models using evolutionary algorithms,
pruning, and knowledge distillation on dual GPUs.
"""

__version__ = "0.1.0"
__author__ = "Genesis Team"

__all__ = [
    "EvolutionaryOptimizer",
    "GenesisConfig",
    "HardwareConfig",
    "__version__",
]


def __getattr__(name):
    if name == "EvolutionaryOptimizer":
        from genesis.optimizer import EvolutionaryOptimizer
        return EvolutionaryOptimizer
    if name in ("GenesisConfig", "HardwareConfig"):
        if name == "GenesisConfig":
            from genesis.config.settings import GenesisConfig
            return GenesisConfig
        from genesis.config.hardware import HardwareConfig
        return HardwareConfig
    raise AttributeError(f"module 'genesis' has no attribute {name!r}")
