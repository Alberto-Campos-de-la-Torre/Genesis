"""Main EvolutionaryOptimizer class for Genesis."""

from typing import Any, Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from genesis.config.settings import GenesisConfig
from genesis.config.hardware import HardwareConfig
from genesis.core.genetics import Genetics
from genesis.core.population import Population, Individual
from genesis.core.fitness import FitnessEvaluator, create_fitness_evaluator
from genesis.core.selection import SelectionStrategy, TournamentSelection, ElitismSelection
from genesis.models.teacher import TeacherModel
from genesis.models.student import StudentModel
from genesis.distillation.trainer import DistillationTrainer, TrainingConfig
from genesis.distillation.kd_loss import KDLoss
from genesis.pruning.pruner import Pruner, PruningConfig
from genesis.utils.logging import setup_logging, ProgressLogger, TrainingLogger
from genesis.utils.checkpointing import CheckpointManager
from genesis.utils.metrics import FitnessTracker, MetricsTracker

logger = logging.getLogger(__name__)


class EvolutionaryOptimizer:
    """
    Main orchestrator for evolutionary optimization of AI models.

    Combines genetic algorithms, knowledge distillation, and pruning
    to create efficient AI models using dual-GPU acceleration.
    """

    def __init__(
        self,
        config: Optional[GenesisConfig] = None,
        teacher_model: Optional[TeacherModel] = None,
        student_model: Optional[StudentModel] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """
        Initialize the evolutionary optimizer.

        Args:
            config: Genesis configuration
            teacher_model: Pre-initialized teacher model
            student_model: Pre-initialized student model
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
        """
        self.config = config or GenesisConfig()

        # Set up logging
        self.logger = setup_logging(
            log_level="INFO",
            log_dir=self.config.log_dir,
            name="genesis",
        )

        # Hardware configuration
        self.hardware = HardwareConfig(
            teacher_device=self.config.teacher_device,
            student_device=self.config.student_device,
        )

        # Models
        self.teacher = teacher_model
        self.student = student_model

        # Data
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Core components (initialized lazily)
        self._genetics: Optional[Genetics] = None
        self._population: Optional[Population] = None
        self._fitness_evaluator: Optional[FitnessEvaluator] = None
        self._selection_strategy: Optional[SelectionStrategy] = None
        self._distillation_trainer: Optional[DistillationTrainer] = None
        self._pruner: Optional[Pruner] = None

        # Tracking
        self.fitness_tracker = FitnessTracker()
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_checkpoints=5,
        )

        # Training logger
        self.training_logger = TrainingLogger(
            log_dir=self.config.log_dir,
            use_tensorboard=self.config.use_tensorboard,
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project,
        )

        # State
        self._initialized = False
        self._current_generation = 0

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Genesis Evolutionary Optimizer...")

        # Log hardware info
        logger.info(self.hardware.memory_summary())

        # Initialize genetics
        self._genetics = Genetics(
            crossover_rate=self.config.genetic.crossover_rate,
            mutation_rate=self.config.genetic.mutation_rate,
            mutation_scale=self.config.genetic.mutation_scale,
            slerp_ratio=self.config.genetic.slerp_ratio,
            adaptive_mutation=self.config.genetic.adaptive_mutation,
            mutation_decay=self.config.genetic.mutation_decay,
            min_mutation_rate=self.config.genetic.min_mutation_rate,
        )

        # Initialize population
        self._population = Population(
            size=self.config.genetic.population_size,
            genetics=self._genetics,
            elite_size=self.config.genetic.elite_size,
        )

        # Initialize selection strategy
        self._selection_strategy = TournamentSelection(
            tournament_size=self.config.genetic.tournament_size,
        )

        # Load models if not provided
        if self.teacher is None:
            self.teacher = TeacherModel(
                model_name_or_path=self.config.teacher_model,
                device=self.hardware.teacher_device,
            )
            self.teacher.load()

        if self.student is None:
            student_model = self.config.student_model or self.config.teacher_model
            self.student = StudentModel(
                model_name_or_path=student_model,
                device=self.hardware.student_device,
                use_lora=self.config.use_lora,
            )
            self.student.load()

        # Initialize population from student model
        self._population.initialize_from_model(
            self.student.model,
            perturbation_scale=self.config.genetic.mutation_scale,
        )

        self._initialized = True
        logger.info("Initialization complete")

    def set_fitness_evaluator(self, evaluator: FitnessEvaluator) -> None:
        """Set custom fitness evaluator."""
        self._fitness_evaluator = evaluator

    def set_selection_strategy(self, strategy: SelectionStrategy) -> None:
        """Set custom selection strategy."""
        self._selection_strategy = strategy

    def run(
        self,
        num_generations: Optional[int] = None,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Any]:
        """
        Run the evolutionary optimization.

        Args:
            num_generations: Number of generations (overrides config)
            callback: Optional callback after each generation

        Returns:
            Dictionary with optimization results
        """
        if not self._initialized:
            self.initialize()

        num_generations = num_generations or self.config.genetic.generations
        progress_logger = ProgressLogger(num_generations)

        logger.info(f"Starting evolution for {num_generations} generations")

        for generation in range(num_generations):
            self._current_generation = generation

            # Evaluate population
            self._evaluate_population()

            # Log progress
            best = self._population.best
            avg_fitness = self._population.average_fitness
            diversity = self._population.diversity

            progress_logger.log_generation(
                generation=generation,
                best_fitness=best.fitness,
                avg_fitness=avg_fitness,
                diversity=diversity,
            )

            self.fitness_tracker.update(
                generation=generation,
                best_fitness=best.fitness,
                avg_fitness=avg_fitness,
                diversity=diversity,
            )

            # Log to tensorboard/wandb
            self.training_logger.log_metrics(
                {
                    "best_fitness": best.fitness,
                    "avg_fitness": avg_fitness,
                    "diversity": diversity,
                    "mutation_rate": self._genetics._get_current_mutation_rate(),
                },
                step=generation,
                prefix="evolution",
            )

            # Callback
            if callback:
                callback({
                    "generation": generation,
                    "best": best,
                    "avg_fitness": avg_fitness,
                    "diversity": diversity,
                })

            # Check for early stopping
            if self.fitness_tracker.is_converged():
                logger.info(f"Converged at generation {generation}")
                break

            # Checkpoint
            if generation % self.config.save_every_n_generations == 0:
                self._save_checkpoint(generation)

            # Evolve to next generation
            self._population.evolve()

            # Optional: Distillation refinement
            if generation % self.config.eval_every_n_generations == 0:
                self._refine_best_individual()

        # Final evaluation and save
        self._evaluate_population()
        best = self._population.best

        # Save best model
        self._save_best_model(best)

        # Close loggers
        self.training_logger.close()

        results = {
            "best_fitness": best.fitness,
            "best_individual": best,
            "generations": self._current_generation + 1,
            "fitness_history": self.fitness_tracker.get_summary(),
            "converged": self.fitness_tracker.is_converged(),
        }

        logger.info(f"Evolution complete. Best fitness: {best.fitness:.4f}")
        return results

    def _evaluate_population(self) -> None:
        """Evaluate fitness of all individuals."""
        if self._fitness_evaluator is None:
            # Create default fitness evaluator
            if self.eval_dataloader is not None:
                self._fitness_evaluator = create_fitness_evaluator(
                    fitness_type="perplexity",
                    dataloader=self.eval_dataloader,
                    device=self.hardware.student_device,
                    max_samples=self.config.eval_samples,
                )
            else:
                raise ValueError("Fitness evaluator or eval_dataloader required")

        def fitness_fn(state_dict: dict[str, torch.Tensor]) -> float:
            # Load state into student model
            self.student.load_state_dict(state_dict, strict=False)

            # Evaluate
            result = self._fitness_evaluator.evaluate(self.student.model)
            return result.score

        self._population.evaluate(fitness_fn)

    def _refine_best_individual(self) -> None:
        """Refine best individual with distillation."""
        if self.train_dataloader is None:
            return

        best = self._population.best
        logger.info("Refining best individual with distillation...")

        # Load best state into student
        self.student.load_state_dict(best.state_dict, strict=False)

        # Create distillation trainer
        trainer_config = TrainingConfig(
            learning_rate=self.config.distillation.learning_rate,
            max_steps=100,  # Short refinement
            temperature=self.config.distillation.temperature,
            alpha=self.config.distillation.alpha,
        )

        trainer = DistillationTrainer(
            teacher=self.teacher,
            student=self.student,
            train_dataloader=self.train_dataloader,
            config=trainer_config,
        )

        # Train briefly
        trainer.train(num_steps=100)

        # Update best individual's state using the full model state dict so it
        # stays consistent with how the rest of the population is stored.
        best.state_dict = self.student.get_state_dict(lora_only=False)

    def _save_checkpoint(self, generation: int) -> None:
        """Save evolution checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"evolution_gen{generation}.pt"

        state = {
            "generation": generation,
            "population_state": self._population.get_state(),
            "genetics_generation": self._genetics.generation,
            "fitness_tracker": {
                "best_history": self.fitness_tracker.best_fitness_history,
                "avg_history": self.fitness_tracker.avg_fitness_history,
            },
            "config": self.config.to_dict(),
        }

        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load evolution checkpoint."""
        state = torch.load(checkpoint_path)

        self._current_generation = state["generation"]
        self._population.load_state(state["population_state"])
        self._genetics._generation = state["genetics_generation"]

        self.fitness_tracker.best_fitness_history = state["fitness_tracker"]["best_history"]
        self.fitness_tracker.avg_fitness_history = state["fitness_tracker"]["avg_history"]

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def _save_best_model(self, individual: Individual) -> None:
        """Save the best evolved model."""
        output_path = Path(self.config.output_dir) / "best_model"
        output_path.mkdir(parents=True, exist_ok=True)

        # Load state into student and save
        self.student.load_state_dict(individual.state_dict, strict=False)
        self.student.save(str(output_path))

        # Save metadata
        metadata = {
            "fitness": individual.fitness,
            "generation": individual.generation,
            "parent_ids": individual.parent_ids,
            "config": self.config.to_dict(),
        }

        import json

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Best model saved to {output_path}")

    def prune_model(
        self,
        target_sparsity: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Prune the best evolved model.

        Args:
            target_sparsity: Target sparsity (overrides config)

        Returns:
            Pruning statistics
        """
        if self._pruner is None:
            pruning_config = PruningConfig(
                target_sparsity=target_sparsity or self.config.pruning.target_sparsity,
                pruning_method=self.config.pruning.pruning_method,
                structured=self.config.pruning.structured,
                granularity=self.config.pruning.granularity,
                block_size=self.config.pruning.block_size,
                iterative_steps=self.config.pruning.iterative_steps,
                initial_sparsity=self.config.pruning.initial_sparsity,
                final_sparsity=self.config.pruning.final_sparsity,
                pruning_schedule=self.config.pruning.pruning_schedule,
                skip_layers=self.config.pruning.skip_layers,
                layer_sparsity_overrides=self.config.pruning.layer_sparsity_overrides,
            )
            self._pruner = Pruner(
                model=self.student.model,
                config=pruning_config,
                dataloader=self.train_dataloader,
                device=self.hardware.student_device,
            )

        stats = self._pruner.prune()
        logger.info(f"Model pruned to {stats['actual_sparsity']:.2%} sparsity")
        return stats

    def distill(
        self,
        num_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Run full knowledge distillation training.

        Args:
            num_steps: Number of training steps

        Returns:
            Training results
        """
        if self.train_dataloader is None:
            raise ValueError("train_dataloader required for distillation")

        trainer_config = TrainingConfig(
            learning_rate=self.config.distillation.learning_rate,
            max_steps=num_steps or self.config.distillation.max_steps,
            warmup_steps=self.config.distillation.warmup_steps,
            temperature=self.config.distillation.temperature,
            alpha=self.config.distillation.alpha,
            output_dir=self.config.output_dir,
        )

        self._distillation_trainer = DistillationTrainer(
            teacher=self.teacher,
            student=self.student,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            config=trainer_config,
        )

        results = self._distillation_trainer.train()
        return results

    @property
    def best_individual(self) -> Optional[Individual]:
        """Get current best individual."""
        if self._population is not None:
            return self._population.best
        return None

    @property
    def population(self) -> Optional[Population]:
        """Get current population."""
        return self._population

    @property
    def current_generation(self) -> int:
        """Get current generation number."""
        return self._current_generation


def main():
    """Example usage of EvolutionaryOptimizer."""
    # Create configuration
    config = GenesisConfig(
        project_name="genesis_example",
        teacher_model="gpt2",  # Use small model for example
        use_lora=True,
        genetic=dict(
            population_size=10,
            generations=5,
            mutation_rate=0.1,
        ),
    )

    # Create optimizer
    optimizer = EvolutionaryOptimizer(config=config)

    # Note: In real usage, you would provide dataloaders
    # optimizer.initialize()
    # results = optimizer.run()

    print("Genesis Evolutionary Optimizer ready!")
    print(f"Config: {config.project_name}")


if __name__ == "__main__":
    main()
