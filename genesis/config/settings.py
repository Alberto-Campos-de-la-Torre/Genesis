"""Global settings and configuration for Genesis."""

from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import yaml


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm parameters."""

    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    slerp_ratio: float = 0.5

    # Mutation parameters
    mutation_scale: float = 0.01
    mutation_prob_per_weight: float = 0.1

    # Adaptive mutation
    adaptive_mutation: bool = True
    mutation_decay: float = 0.95
    min_mutation_rate: float = 0.01


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 4.0
    alpha: float = 0.5  # Weight for distillation loss vs hard label loss
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Feature distillation
    use_feature_distillation: bool = False
    feature_layers: list = field(default_factory=lambda: [-1, -2, -3])
    feature_weight: float = 0.1


@dataclass
class PruningConfig:
    """Configuration for model pruning."""

    target_sparsity: float = 0.3
    pruning_method: str = "magnitude"  # magnitude, gradient, taylor
    structured: bool = False
    granularity: str = "element"  # element, row, column, block
    iterative_steps: int = 1

    # Gradual pruning
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.3
    pruning_schedule: str = "cubic"  # linear, cubic, exponential


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class GenesisConfig:
    """Main configuration class for Genesis."""

    # Project settings
    project_name: str = "genesis_experiment"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    seed: int = 42

    # Model settings
    teacher_model: str = "meta-llama/Llama-2-7b-hf"
    student_model: Optional[str] = None  # If None, derived from teacher
    use_lora: bool = True

    # Hardware settings
    teacher_device: str = "cuda:0"
    student_device: str = "cuda:1"
    mixed_precision: str = "fp16"  # fp16, bf16, fp32

    # Sub-configurations
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Experiment settings
    experiment_type: str = "llm"  # llm, tts
    dataset_name: str = "pubmed_qa"
    max_samples: Optional[int] = None
    eval_samples: int = 100

    # Logging
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "genesis"
    log_every_n_steps: int = 10
    eval_every_n_generations: int = 5
    save_every_n_generations: int = 10

    def __post_init__(self):
        """Validate and set up configuration."""
        # Create directories
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "GenesisConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "GenesisConfig":
        """Create configuration from dictionary."""
        # Extract sub-configs
        genetic_dict = config_dict.pop("genetic", {})
        distillation_dict = config_dict.pop("distillation", {})
        pruning_dict = config_dict.pop("pruning", {})
        lora_dict = config_dict.pop("lora", {})

        return cls(
            **config_dict,
            genetic=GeneticConfig(**genetic_dict),
            distillation=DistillationConfig(**distillation_dict),
            pruning=PruningConfig(**pruning_dict),
            lora=LoRAConfig(**lora_dict),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "seed": self.seed,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "use_lora": self.use_lora,
            "teacher_device": self.teacher_device,
            "student_device": self.student_device,
            "mixed_precision": self.mixed_precision,
            "experiment_type": self.experiment_type,
            "dataset_name": self.dataset_name,
            "max_samples": self.max_samples,
            "eval_samples": self.eval_samples,
            "use_tensorboard": self.use_tensorboard,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "log_every_n_steps": self.log_every_n_steps,
            "eval_every_n_generations": self.eval_every_n_generations,
            "save_every_n_generations": self.save_every_n_generations,
            "genetic": {
                "population_size": self.genetic.population_size,
                "generations": self.genetic.generations,
                "mutation_rate": self.genetic.mutation_rate,
                "crossover_rate": self.genetic.crossover_rate,
                "elite_size": self.genetic.elite_size,
                "tournament_size": self.genetic.tournament_size,
                "slerp_ratio": self.slerp_ratio if hasattr(self, "slerp_ratio") else self.genetic.slerp_ratio,
                "mutation_scale": self.genetic.mutation_scale,
                "mutation_prob_per_weight": self.genetic.mutation_prob_per_weight,
                "adaptive_mutation": self.genetic.adaptive_mutation,
                "mutation_decay": self.genetic.mutation_decay,
                "min_mutation_rate": self.genetic.min_mutation_rate,
            },
            "distillation": {
                "temperature": self.distillation.temperature,
                "alpha": self.distillation.alpha,
                "learning_rate": self.distillation.learning_rate,
                "warmup_steps": self.distillation.warmup_steps,
                "max_steps": self.distillation.max_steps,
                "batch_size": self.distillation.batch_size,
                "gradient_accumulation_steps": self.distillation.gradient_accumulation_steps,
                "use_feature_distillation": self.distillation.use_feature_distillation,
                "feature_layers": self.distillation.feature_layers,
                "feature_weight": self.distillation.feature_weight,
            },
            "pruning": {
                "target_sparsity": self.pruning.target_sparsity,
                "pruning_method": self.pruning.pruning_method,
                "structured": self.pruning.structured,
                "granularity": self.pruning.granularity,
                "iterative_steps": self.pruning.iterative_steps,
                "initial_sparsity": self.pruning.initial_sparsity,
                "final_sparsity": self.pruning.final_sparsity,
                "pruning_schedule": self.pruning.pruning_schedule,
            },
            "lora": {
                "r": self.lora.r,
                "lora_alpha": self.lora.lora_alpha,
                "lora_dropout": self.lora.lora_dropout,
                "target_modules": self.lora.target_modules,
                "bias": self.lora.bias,
                "task_type": self.lora.task_type,
            },
        }
