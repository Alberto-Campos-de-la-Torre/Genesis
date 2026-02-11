#!/usr/bin/env python3
"""
Medical LLM Evolution Experiment

Evolves a medical question-answering model using knowledge distillation
from a larger teacher model and evolutionary optimization.
"""

import argparse
import logging
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer

from genesis import EvolutionaryOptimizer, GenesisConfig
from genesis.data.datasets import PubMedQADataset, create_dataloader
from genesis.models.teacher import TeacherModel
from genesis.models.student import StudentModel
from genesis.core.fitness import QAFitness
from genesis.utils.logging import setup_logging


def load_config(config_path: str) -> GenesisConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return GenesisConfig.from_dict(config_dict)


def main():
    parser = argparse.ArgumentParser(description="Medical LLM Evolution Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override number of generations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with minimal settings for testing",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))

    # Dry run settings
    if args.dry_run:
        config.genetic.population_size = 4
        config.genetic.generations = 2
        config.max_samples = 100
        config.eval_samples = 20
        config.distillation.max_steps = 10

    # Setup logging
    logger = setup_logging(
        log_level="INFO",
        log_dir=config.log_dir,
        name="medical_llm",
    )
    logger.info(f"Starting Medical LLM Evolution Experiment")
    logger.info(f"Config: {config.project_name}")

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    logger.info("Loading PubMedQA dataset...")
    train_dataset = PubMedQADataset(
        tokenizer=tokenizer,
        split="train",
        max_length=512,
        max_samples=config.max_samples,
    )

    eval_dataset = PubMedQADataset(
        tokenizer=tokenizer,
        split="train",  # PubMedQA doesn't have separate val split
        max_length=512,
        max_samples=config.eval_samples,
    )

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.distillation.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.distillation.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize models
    logger.info("Initializing teacher model...")
    teacher = TeacherModel(
        model_name_or_path=config.teacher_model,
        device=config.teacher_device,
        dtype=torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16,
    )
    teacher.load()

    logger.info("Initializing student model...")
    student = StudentModel(
        model_name_or_path=config.student_model or config.teacher_model,
        device=config.student_device,
        dtype=torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16,
        use_lora=config.use_lora,
        lora_config=config.lora,
    )
    student.load()

    # Create fitness evaluator
    fitness_evaluator = QAFitness(
        dataloader=eval_dataloader,
        tokenizer=tokenizer,
        device=config.student_device,
        max_samples=config.eval_samples,
    )

    # Create optimizer
    logger.info("Creating evolutionary optimizer...")
    optimizer = EvolutionaryOptimizer(
        config=config,
        teacher_model=teacher,
        student_model=student,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Set custom fitness evaluator
    optimizer.set_fitness_evaluator(fitness_evaluator)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer.load_checkpoint(args.resume)

    # Run evolution
    logger.info("Starting evolution...")
    results = optimizer.run(
        num_generations=args.generations,
        callback=lambda r: logger.info(
            f"Gen {r['generation']}: Best={r['best'].fitness:.4f}, Avg={r['avg_fitness']:.4f}"
        ),
    )

    # Log results
    logger.info("=" * 50)
    logger.info("Evolution Complete!")
    logger.info(f"Best Fitness: {results['best_fitness']:.4f}")
    logger.info(f"Generations: {results['generations']}")
    logger.info(f"Converged: {results['converged']}")
    logger.info("=" * 50)

    # Optional: Prune the best model
    if config.pruning.target_sparsity > 0:
        logger.info("Pruning best model...")
        prune_stats = optimizer.prune_model()
        logger.info(f"Pruning complete: {prune_stats['actual_sparsity']:.2%} sparsity")

    # Optional: Final distillation refinement
    logger.info("Running final distillation refinement...")
    distill_results = optimizer.distill(num_steps=config.distillation.max_steps)
    logger.info(f"Final distillation loss: {distill_results.get('best_eval_loss', 'N/A')}")

    logger.info(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
