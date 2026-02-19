"""Command-line interface for Genesis AI Evolution Laboratory."""

import argparse
import sys
import logging
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the full evolutionary optimization pipeline."""
    from genesis.config.settings import GenesisConfig
    from genesis.optimizer import EvolutionaryOptimizer

    if args.config:
        config = GenesisConfig.from_yaml(args.config)
    else:
        config = GenesisConfig()

    # CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.generations is not None:
        config.genetic.generations = args.generations
    if args.population_size is not None:
        config.genetic.population_size = args.population_size
    if args.teacher_model:
        config.teacher_model = args.teacher_model
    if args.student_model:
        config.student_model = args.student_model
    if args.no_lora:
        config.use_lora = False

    optimizer = EvolutionaryOptimizer(config=config)

    logging.info("Starting evolutionary optimization...")
    results = optimizer.run()

    print(f"\nEvolution complete.")
    print(f"  Best fitness : {results['best_fitness']:.6f}")
    print(f"  Generations  : {results['generations']}")
    print(f"  Converged    : {results['converged']}")
    print(f"  Output dir   : {config.output_dir}")
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    """Run knowledge distillation only."""
    from genesis.config.settings import GenesisConfig
    from genesis.optimizer import EvolutionaryOptimizer

    if args.config:
        config = GenesisConfig.from_yaml(args.config)
    else:
        config = GenesisConfig()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.steps is not None:
        config.distillation.max_steps = args.steps
    if args.teacher_model:
        config.teacher_model = args.teacher_model
    if args.student_model:
        config.student_model = args.student_model

    optimizer = EvolutionaryOptimizer(config=config)
    optimizer.initialize()

    logging.info("Starting knowledge distillation...")
    results = optimizer.distill(num_steps=args.steps)

    print(f"\nDistillation complete.")
    print(f"  Final loss : {results.get('final_loss', 'N/A')}")
    print(f"  Steps      : {results.get('total_steps', 'N/A')}")
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    """Run model pruning only."""
    from genesis.config.settings import GenesisConfig
    from genesis.optimizer import EvolutionaryOptimizer

    if args.config:
        config = GenesisConfig.from_yaml(args.config)
    else:
        config = GenesisConfig()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.sparsity is not None:
        config.pruning.target_sparsity = args.sparsity
    if args.method:
        config.pruning.pruning_method = args.method
    if args.student_model:
        config.student_model = args.student_model

    optimizer = EvolutionaryOptimizer(config=config)
    optimizer.initialize()

    logging.info("Starting model pruning...")
    stats = optimizer.prune_model(target_sparsity=args.sparsity)

    print(f"\nPruning complete.")
    print(f"  Target sparsity : {config.pruning.target_sparsity:.1%}")
    print(f"  Actual sparsity : {stats.get('actual_sparsity', 0):.1%}")
    print(f"  Params pruned   : {stats.get('pruned_params', 'N/A')}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Show hardware and environment information."""
    import torch
    from genesis import __version__
    from genesis.config.hardware import HardwareConfig

    print(f"Genesis AI Evolution Laboratory v{__version__}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {'available' if torch.cuda.is_available() else 'not available'}")

    if torch.cuda.is_available():
        print(f"  GPUs    : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024 ** 3
            print(f"    [{i}] {props.name} — {vram:.1f} GB VRAM")

    hw = HardwareConfig()
    print(f"\n{hw.memory_summary()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="genesis",
        description="Genesis AI Evolution Laboratory — evolve efficient AI models.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ------------------------------------------------------------------ run --
    run_p = subparsers.add_parser("run", help="Run the full evolutionary optimization pipeline")
    run_p.add_argument("--config", "-c", metavar="FILE", help="Path to YAML config file")
    run_p.add_argument("--output-dir", "-o", metavar="DIR", help="Override output directory")
    run_p.add_argument("--teacher-model", metavar="NAME", help="Teacher model name or path")
    run_p.add_argument("--student-model", metavar="NAME", help="Student model name or path")
    run_p.add_argument("--generations", "-g", type=int, metavar="N", help="Number of generations")
    run_p.add_argument("--population-size", "-p", type=int, metavar="N", help="Population size")
    run_p.add_argument("--no-lora", action="store_true", help="Disable LoRA adapters")
    run_p.set_defaults(func=_cmd_run)

    # --------------------------------------------------------------- distill --
    dist_p = subparsers.add_parser("distill", help="Run knowledge distillation only")
    dist_p.add_argument("--config", "-c", metavar="FILE", help="Path to YAML config file")
    dist_p.add_argument("--output-dir", "-o", metavar="DIR", help="Override output directory")
    dist_p.add_argument("--teacher-model", metavar="NAME", help="Teacher model name or path")
    dist_p.add_argument("--student-model", metavar="NAME", help="Student model name or path")
    dist_p.add_argument("--steps", "-s", type=int, metavar="N", help="Number of training steps")
    dist_p.set_defaults(func=_cmd_distill)

    # ----------------------------------------------------------------- prune --
    prune_p = subparsers.add_parser("prune", help="Prune a model to a target sparsity")
    prune_p.add_argument("--config", "-c", metavar="FILE", help="Path to YAML config file")
    prune_p.add_argument("--output-dir", "-o", metavar="DIR", help="Override output directory")
    prune_p.add_argument("--student-model", metavar="NAME", help="Model to prune")
    prune_p.add_argument("--sparsity", type=float, metavar="F",
                         help="Target sparsity, e.g. 0.3 for 30%%")
    prune_p.add_argument("--method", choices=["magnitude", "gradient", "taylor", "fisher"],
                         help="Pruning saliency method")
    prune_p.set_defaults(func=_cmd_prune)

    # ------------------------------------------------------------------ info --
    info_p = subparsers.add_parser("info", help="Show hardware and environment information")
    info_p.set_defaults(func=_cmd_info)

    return parser


def main() -> None:
    """Entry point for the `genesis` console script."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        exit_code = args.func(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logging.error(str(exc), exc_info=args.log_level == "DEBUG")
        sys.exit(1)


if __name__ == "__main__":
    main()
