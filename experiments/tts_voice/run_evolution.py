#!/usr/bin/env python3
"""
TTS Voice Evolution Experiment

Evolves TTS style tokens and speaker embeddings to optimize
for voice quality, naturalness, and target speaker similarity.
"""

import argparse
import logging
from pathlib import Path
import yaml
import torch
import numpy as np

from genesis.tts.tts_child import TTSChild, TTSConfig
from genesis.tts.style_evolution import StyleEvolution
from genesis.tts.mcd_fitness import MCDFitness, compute_mcd
from genesis.data.datasets import TTSDataset
from genesis.utils.logging import setup_logging, ProgressLogger
from genesis.utils.checkpointing import CheckpointManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TTS Voice Evolution Experiment")
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
        config["genetic"]["population_size"] = 5
        config["genetic"]["generations"] = 3
        config["data"]["max_samples"] = 10
        config["data"]["eval_samples"] = 5

    # Setup logging
    logger = setup_logging(
        log_level="INFO",
        log_dir=config["log_dir"],
        name="tts_evolution",
    )
    logger.info("Starting TTS Voice Evolution Experiment")

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Create directories
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Create TTS config
    tts_config = TTSConfig(
        model_type=config["tts"]["model_type"],
        style_dim=config["tts"]["style_dim"],
        speaker_dim=config["tts"]["speaker_dim"],
        num_speakers=config["tts"]["num_speakers"],
        sample_rate=config["tts"]["sample_rate"],
        n_mel_channels=config["tts"]["n_mel_channels"],
        hop_length=config["tts"]["hop_length"],
        win_length=config["tts"]["win_length"],
        n_fft=config["tts"]["n_fft"],
    )

    # Load reference audio for fitness evaluation
    logger.info("Loading reference audio...")
    try:
        tts_dataset = TTSDataset(
            data_dir=config["data"]["data_dir"],
            sample_rate=config["tts"]["sample_rate"],
            max_samples=config["data"]["eval_samples"],
        )
        reference_mels = [sample["mel_spectrogram"] for sample in tts_dataset]
        logger.info(f"Loaded {len(reference_mels)} reference mel spectrograms")
    except Exception as e:
        logger.warning(f"Could not load reference audio: {e}")
        logger.warning("Using synthetic reference mels for demonstration")
        # Create synthetic reference mels for demonstration
        reference_mels = [
            torch.randn(config["tts"]["n_mel_channels"], 100 + i * 10)
            for i in range(10)
        ]

    # Create fitness evaluator
    fitness_evaluator = MCDFitness(
        reference_mels=reference_mels,
        target_mcd=config["fitness"]["target_mcd"],
        weight_naturalness=config["fitness"]["weight_naturalness"],
        weight_similarity=config["fitness"]["weight_similarity"],
        device=config["device"],
    )

    # Initialize style evolution
    logger.info("Initializing style evolution...")
    style_evolution = StyleEvolution(
        style_dim=config["tts"]["style_dim"],
        num_tokens=config["style_evolution"]["num_style_tokens"],
        population_size=config["genetic"]["population_size"],
        elite_size=config["genetic"]["elite_size"],
        mutation_rate=config["genetic"]["mutation_rate"],
        mutation_scale=config["genetic"]["mutation_scale"],
        crossover_rate=config["genetic"]["crossover_rate"],
    )

    # Initialize population
    style_evolution.initialize_population(
        perturbation_scale=config["style_evolution"]["perturbation_scale"],
    )

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config["checkpoint_dir"],
        max_checkpoints=5,
        metric_name="fitness",
        mode="max",
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        style_evolution.load_state(args.resume)

    # Progress logging
    num_generations = args.generations or config["genetic"]["generations"]
    progress_logger = ProgressLogger(num_generations)

    # Create TTS child for synthesis
    tts_child = TTSChild(config=tts_config)

    # Evolution loop
    logger.info(f"Starting evolution for {num_generations} generations")
    best_overall_fitness = 0.0
    best_style_tokens = None

    for generation in range(num_generations):
        # Evaluate population
        population = style_evolution.get_population()
        fitnesses = []

        for idx, style_tokens in enumerate(population):
            # Set style tokens in TTS child
            tts_child.style_tokens = style_tokens

            # Synthesize mel spectrogram (using a test text)
            test_text = "This is a test sentence for voice evolution."

            try:
                output = tts_child.synthesize(
                    text=test_text,
                    device=config["device"],
                )
                synthesized_mel = output["mel_spectrogram"]
            except Exception:
                # If synthesis fails, create dummy output
                synthesized_mel = torch.randn(
                    1,
                    config["tts"]["n_mel_channels"],
                    100,
                )

            # Evaluate fitness
            fitness_result = fitness_evaluator.evaluate(synthesized_mel.squeeze(0))
            fitnesses.append(fitness_result["fitness"])

        # Update fitnesses in evolution
        style_evolution.set_all_fitnesses(fitnesses)

        # Get best individual
        best_tokens, best_fitness = style_evolution.get_best()

        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_style_tokens = best_tokens.clone()

        # Log progress
        avg_fitness = style_evolution.average_fitness
        progress_logger.log_generation(
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
        )

        # Checkpoint
        if generation % config["save_every_n_generations"] == 0:
            style_evolution.save_state(
                Path(config["checkpoint_dir"]) / f"evolution_gen{generation}.pt"
            )

        # Evolve to next generation
        style_evolution.evolve()

    # Save final results
    logger.info("=" * 50)
    logger.info("Evolution Complete!")
    logger.info(f"Best Fitness: {best_overall_fitness:.4f}")
    logger.info("=" * 50)

    # Save best style tokens
    output_path = Path(config["output_dir"]) / "best_style_tokens.pt"
    torch.save(
        {
            "style_tokens": best_style_tokens,
            "fitness": best_overall_fitness,
            "config": config,
        },
        output_path,
    )
    logger.info(f"Best style tokens saved to {output_path}")

    # Save final evolution state
    style_evolution.save_state(
        Path(config["checkpoint_dir"]) / "final_evolution_state.pt"
    )

    logger.info(f"Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
