"""Live training dashboard with Rich terminal display and matplotlib plot generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


class TrainingDashboard:
    """
    Live terminal dashboard for training metrics + persistent plot/report generation.

    Uses Rich Live for a live-updating terminal panel and stores metric history
    in-memory for later plot generation.
    """

    def __init__(self, title: str = "Genesis Training Dashboard"):
        self.title = title
        self.console = Console()

        # History stores
        self._distill_history: list[dict] = []
        self._evo_history: list[dict] = []

        # Live display state (latest values for the live table)
        self._last_distill: dict = {}
        self._last_evo: dict = {}

        # Rich Live context
        self._live: Optional[Live] = None

    # ------------------------------------------------------------------
    # Live display lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Rich Live display."""
        self._live = Live(
            self._build_renderable(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()

    def stop(self) -> None:
        """Stop the Rich Live display."""
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

    def _build_renderable(self):
        """Build the Rich renderable panel from current state."""
        table = Table(title=self.title, expand=True, show_lines=True)

        # Distillation section
        if self._last_distill or self._distill_history:
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")

            d = self._last_distill
            table.add_row("Phase", "Distillation")
            table.add_row("Step", str(d.get("step", "—")))
            table.add_row("Loss", f"{d.get('loss', float('nan')):.4f}" if "loss" in d else "—")
            table.add_row("KD Loss", f"{d.get('kd_loss', float('nan')):.4f}" if "kd_loss" in d else "—")
            table.add_row("Hard Loss", f"{d.get('hard_loss', float('nan')):.4f}" if "hard_loss" in d else "—")
            table.add_row("LR", f"{d.get('lr', float('nan')):.2e}" if "lr" in d else "—")

        # Evolutionary section
        if self._last_evo or self._evo_history:
            e = self._last_evo
            if not (self._last_distill or self._distill_history):
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Value", style="green")
            table.add_row("—", "—")
            table.add_row("Phase", "Evolution")
            table.add_row("Generation", str(e.get("gen", "—")))
            table.add_row("Best Fitness", f"{e.get('best_fitness', float('nan')):.4f}" if "best_fitness" in e else "—")
            table.add_row("Avg Fitness", f"{e.get('avg_fitness', float('nan')):.4f}" if "avg_fitness" in e else "—")
            table.add_row("Diversity", f"{e.get('diversity', float('nan')):.4f}" if "diversity" in e else "—")
            table.add_row("Mutation Rate", f"{e.get('mutation_rate', float('nan')):.4f}" if "mutation_rate" in e else "—")

        if not table.columns:
            return Panel(Text("Waiting for metrics…", style="yellow"), title=self.title)

        return Panel(table, title=self.title)

    def _refresh(self) -> None:
        """Refresh the live display if active."""
        if self._live is not None:
            self._live.update(self._build_renderable())

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_distillation(
        self,
        payload: dict,
    ) -> None:
        """
        Update dashboard with a distillation step's metrics.

        Accepts either a raw dict (from trainer callback) or keyword args.
        The trainer callback dict contains: loss, kd_loss, hard_loss, step, lr.
        """
        step = payload.get("step", len(self._distill_history))
        loss = payload.get("loss", float("nan"))
        kd_loss = payload.get("kd_loss", float("nan"))
        hard_loss = payload.get("hard_loss", float("nan"))
        lr = payload.get("lr", float("nan"))

        entry = {
            "step": step,
            "loss": loss,
            "kd_loss": kd_loss,
            "hard_loss": hard_loss,
            "lr": lr,
        }
        self._distill_history.append(entry)
        self._last_distill = entry
        self._refresh()

    def update_evolution(
        self,
        gen: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: float,
        mutation_rate: float,
    ) -> None:
        """Update dashboard with an evolutionary generation's metrics."""
        entry = {
            "gen": gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            "mutation_rate": mutation_rate,
        }
        self._evo_history.append(entry)
        self._last_evo = entry
        self._refresh()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_plots(self, output_dir: str) -> None:
        """
        Generate and save 4 matplotlib plots as PNGs.

        Plots saved:
        - loss_curve.png  : total loss + kd_loss + hard_loss vs step
        - lr_curve.png    : learning rate schedule vs step
        - fitness_curve.png: best/avg fitness vs generation
        - diversity_curve.png: population diversity vs generation
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- Loss curve ---
        if self._distill_history:
            steps = [e["step"] for e in self._distill_history]
            total = [e["loss"] for e in self._distill_history]
            kd = [e["kd_loss"] for e in self._distill_history]
            hard = [e["hard_loss"] for e in self._distill_history]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, total, label="Total Loss", linewidth=2)
            ax.plot(steps, kd, label="KD Loss", linewidth=1.5, linestyle="--")
            ax.plot(steps, hard, label="Hard Loss", linewidth=1.5, linestyle=":")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Distillation Loss Curves")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "loss_curve.png", dpi=120)
            plt.close(fig)

            # --- LR curve ---
            lrs = [e["lr"] for e in self._distill_history]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, lrs, color="orange", linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "lr_curve.png", dpi=120)
            plt.close(fig)

        # --- Fitness curve ---
        if self._evo_history:
            gens = [e["gen"] for e in self._evo_history]
            best = [e["best_fitness"] for e in self._evo_history]
            avg = [e["avg_fitness"] for e in self._evo_history]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(gens, best, label="Best Fitness", linewidth=2, color="green")
            ax.plot(gens, avg, label="Avg Fitness", linewidth=1.5, linestyle="--", color="blue")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.set_title("Evolutionary Fitness Curves")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "fitness_curve.png", dpi=120)
            plt.close(fig)

            # --- Diversity curve ---
            diversity = [e["diversity"] for e in self._evo_history]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(gens, diversity, color="purple", linewidth=2)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Diversity")
            ax.set_title("Population Diversity")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "diversity_curve.png", dpi=120)
            plt.close(fig)

        self.console.print(f"[green]Plots saved to {plots_dir}[/green]")

    def save_report(self, output_dir: str) -> None:
        """Save a JSON report with all metric history."""
        output_path = Path(output_dir) / "training_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "distillation": self._distill_history,
            "evolution": self._evo_history,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        self.console.print(f"[green]Report saved to {output_path}[/green]")
