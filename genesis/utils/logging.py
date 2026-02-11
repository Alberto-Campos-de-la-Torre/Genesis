"""Logging configuration for Genesis."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    name: str = "genesis",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_dir: Optional log directory (creates timestamped file)
        name: Logger name
        format_string: Optional custom format string

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "genesis") -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If no handlers, set up basic logging
    if not logger.handlers:
        setup_logging(name=name)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter with additional context.
    """

    def __init__(self, logger: logging.Logger, prefix: str = ""):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        if self.prefix:
            return f"[{self.prefix}] {msg}", kwargs
        return msg, kwargs


class TrainingLogger:
    """
    Specialized logger for training progress.
    """

    def __init__(
        self,
        name: str = "genesis.training",
        log_dir: Optional[str] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """
        Initialize training logger.

        Args:
            name: Logger name
            log_dir: Log directory
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
        """
        self.logger = get_logger(name)
        self.log_dir = log_dir

        self.tb_writer = None
        self.wandb_run = None

        if use_tensorboard and log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = Path(log_dir) / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"TensorBoard logging to {tb_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available")

        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(project=wandb_project or "genesis")
                self.logger.info("Weights & Biases logging enabled")
            except ImportError:
                self.logger.warning("wandb not available")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step
            prefix: Optional metric name prefix
        """
        # Log to console
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")

        # Log to TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.tb_writer.add_scalar(tag, value, step)

        # Log to W&B
        if self.wandb_run:
            import wandb

            prefixed_metrics = {
                f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()
            }
            wandb.log(prefixed_metrics, step=step)

    def log_histogram(
        self,
        name: str,
        values: "torch.Tensor",
        step: int,
    ) -> None:
        """Log histogram to TensorBoard."""
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text to TensorBoard."""
        if self.tb_writer:
            self.tb_writer.add_text(name, text, step)

    def close(self) -> None:
        """Close logging resources."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb

            wandb.finish()


class ProgressLogger:
    """
    Simple progress logger for evolutionary algorithms.
    """

    def __init__(self, total_generations: int, log_interval: int = 1):
        """
        Initialize progress logger.

        Args:
            total_generations: Total number of generations
            log_interval: Log every N generations
        """
        self.total_generations = total_generations
        self.log_interval = log_interval
        self.logger = get_logger("genesis.evolution")
        self.history: list[dict] = []

    def log_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Log generation statistics.

        Args:
            generation: Current generation
            best_fitness: Best fitness in population
            avg_fitness: Average fitness
            diversity: Population diversity
            **kwargs: Additional metrics
        """
        entry = {
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            **kwargs,
        }
        self.history.append(entry)

        if generation % self.log_interval == 0:
            progress = generation / self.total_generations * 100
            msg = (
                f"Gen {generation}/{self.total_generations} ({progress:.1f}%) | "
                f"Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}"
            )
            if diversity is not None:
                msg += f" | Div: {diversity:.4f}"
            self.logger.info(msg)

    def get_history(self) -> list[dict]:
        """Get logged history."""
        return self.history
