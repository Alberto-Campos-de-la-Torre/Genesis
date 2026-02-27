#!/usr/bin/env python
"""End-to-end Qwen3 KD distillation + evolutionary optimization training entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.table import Table

console = Console()

STUDENT_PATH = "/media/ttech-main/42A4266DA426639F/Models/Qwen3-1.7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-1.7B student via KD from Ollama teacher, then evolve."
    )
    parser.add_argument("--steps", type=int, default=20, help="Distillation optimizer steps")
    parser.add_argument("--generations", type=int, default=5, help="Evolutionary generations (0 = skip)")
    parser.add_argument("--pop-size", type=int, default=4, help="Population size for evolution phase")
    parser.add_argument("--seq-len", type=int, default=128, help="Token sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"./outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for checkpoints, plots, and report",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server base URL",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=os.environ.get("OLLAMA_MODEL", "qwen3:30b-a3b"),
        help="Ollama model tag to use as primary teacher (must be a local model for logprobs support)",
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        default=os.environ.get("OLLAMA_FALLBACK_MODEL", None),
        help="Ollama model tag to fall back to when primary returns 429 (e.g. qwen3:30b)",
    )
    parser.add_argument("--elite-size", type=int, default=2, help="Number of elite individuals preserved each generation")
    parser.add_argument("--mutation-scale", type=float, default=0.05, help="Scale of Gaussian mutation noise (higher = more exploration)")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Per-generation probability of mutation occurring")
    parser.add_argument("--student-path", type=str, default=STUDENT_PATH, help="Path to student model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for student model (distillation)")
    parser.add_argument("--eval-device", type=str, default="cuda:1", help="Device for population fitness evaluation (evolution phase)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_dataloaders(tokenizer, seq_len: int, batch_size: int):
    """Load wikitext-2-raw-v1 and return (train_dataloader, eval_dataloader)."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset

    console.print("[cyan]Loading wikitext-2-raw-v1 dataset…[/cyan]")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    train_texts = [raw[i]["text"] for i in range(min(200, len(raw))) if raw[i]["text"].strip()]
    eval_texts = [raw[i]["text"] for i in range(200, min(250, len(raw))) if raw[i]["text"].strip()]

    def tokenize(texts: list[str]) -> dict:
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        enc["labels"] = enc["input_ids"].clone()
        # Mask padding positions so they don't contribute to the CE loss
        enc["labels"][enc["attention_mask"] == 0] = -100
        return enc

    console.print(f"[cyan]Tokenizing {len(train_texts)} train + {len(eval_texts)} eval samples…[/cyan]")
    train_enc = tokenize(train_texts)
    eval_enc = tokenize(eval_texts)

    train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_enc["labels"])
    eval_ds = TensorDataset(eval_enc["input_ids"], eval_enc["attention_mask"], eval_enc["labels"])

    def collate(batch):
        ids, masks, labels = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, eval_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Logging setup
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy libraries
    for noisy in ("transformers", "datasets", "peft", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold blue]Genesis Qwen3 Training[/bold blue]")
    console.print(f"Output dir : [green]{output_dir}[/green]")
    fallback_str = f" → [yellow]{args.fallback_model}[/yellow]" if args.fallback_model else ""
    console.print(f"Teacher    : [cyan]{args.ollama_model}[/cyan]{fallback_str} @ {args.ollama_host}")
    console.print(f"Student    : [cyan]{args.student_path}[/cyan] on {args.device}")
    console.print(f"Steps      : {args.steps}  |  Generations: {args.generations}  |  Pop: {args.pop_size}")
    console.print(f"Elite size : {args.elite_size}  |  Mutation scale: {args.mutation_scale}  |  Mutation rate: {args.mutation_rate}")
    console.print(f"Devices    : distillation=[cyan]{args.device}[/cyan]  evolution eval=[cyan]{args.eval_device}[/cyan]")

    # ------------------------------------------------------------------
    # 1. Dashboard
    # ------------------------------------------------------------------
    from genesis.utils.dashboard import TrainingDashboard

    dashboard = TrainingDashboard()
    dashboard.start()

    try:
        _run_training(args, output_dir, dashboard)
    finally:
        dashboard.stop()


def _cpu_diversity(population) -> float:
    """Compute population diversity with all tensors moved to CPU."""
    import numpy as np

    individuals = population.individuals
    if len(individuals) < 2:
        return 0.0

    keys = list(individuals[0].state_dict.keys())
    total_std = 0.0
    count = 0

    for key in keys:
        try:
            params = torch.stack([ind.state_dict[key].cpu().float() for ind in individuals])
            total_std += params.std(dim=0).mean().item()
            count += 1
        except Exception:
            pass

    return total_std / count if count > 0 else 0.0


def _run_training(args, output_dir: Path, dashboard) -> None:
    from genesis.models.ollama_teacher import OllamaTeacher
    from genesis.models.student import StudentModel
    from genesis.models.lora_manager import LoRAConfig
    from genesis.distillation.trainer import DistillationTrainer, TrainingConfig

    # ------------------------------------------------------------------
    # 2. Build teacher
    # ------------------------------------------------------------------
    console.print("\n[bold]Setting up teacher (OllamaTeacher)…[/bold]")
    teacher = OllamaTeacher(
        model_name=args.ollama_model,
        base_url=args.ollama_host,
        tokenizer_path=args.student_path,  # shared Qwen3 tokenizer
        top_logprobs=20,
        fallback_model=args.fallback_model,
    )
    teacher.load()
    fallback_note = f" (fallback: [yellow]{args.fallback_model}[/yellow])" if args.fallback_model else ""
    console.print(f"[green]Teacher ready.[/green]{fallback_note}")

    # ------------------------------------------------------------------
    # 3. Build student
    # ------------------------------------------------------------------
    console.print("[bold]Loading student model (Qwen3-1.7B + LoRA)…[/bold]")
    lora_config = LoRAConfig(r=16, lora_alpha=32)
    student = StudentModel(
        model_name_or_path=args.student_path,
        device=args.device,
        dtype=torch.bfloat16,
        use_lora=True,
        lora_config=lora_config,
    )
    student.load()
    console.print("[green]Student ready.[/green]")

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    train_loader, eval_loader = build_dataloaders(
        student.tokenizer, args.seq_len, args.batch_size
    )

    # ------------------------------------------------------------------
    # 5. Phase 1 — KD Distillation
    # ------------------------------------------------------------------
    console.print("\n[bold yellow]Phase 1: Knowledge Distillation[/bold yellow]")

    training_config = TrainingConfig(
        max_steps=args.steps,
        logging_steps=1,
        eval_steps=max(1, args.steps // 5),   # evaluate every ~20% of training
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        temperature=4.0,
        alpha=0.5,
        output_dir=str(output_dir),
        warmup_steps=max(1, args.steps // 10),
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config,
    )

    distill_results = trainer.train(callback=dashboard.update_distillation)

    console.print(
        f"[green]Distillation done. Steps={distill_results['global_step']}, "
        f"Best eval loss={distill_results['best_eval_loss']:.4f}[/green]"
    )

    # ------------------------------------------------------------------
    # 6. Phase 2 — Evolutionary Optimization
    # ------------------------------------------------------------------
    best_fitness = 0.0
    generations_done = 0

    if args.generations > 0:
        console.print("\n[bold yellow]Phase 2: Evolutionary Optimization[/bold yellow]")

        from genesis.core.population import Population
        from genesis.core.genetics import Genetics, mutate as _mutate
        from genesis.core.fitness import PerplexityFitness

        # ── Dual-GPU evolution setup ──────────────────────────────────────
        # cuda:0: keeps the distilled student for saving.
        # cuda:1: eval model — base weights fixed, only LoRA swapped per individual.
        #
        # Population holds ONLY LoRA adapter weights on CPU (~25 MB each).
        # This replaces full model state dicts (~3.4 GB each × 8 = 27 GB RAM OOM).
        # For each individual: inject LoRA weights into cuda:1 eval model → evaluate.
        # ─────────────────────────────────────────────────────────────────
        console.print(f"\n[bold]Loading eval model on {args.eval_device} for population fitness…[/bold]")
        eval_student = StudentModel(
            model_name_or_path=args.student_path,
            device=args.eval_device,
            dtype=torch.bfloat16,
            use_lora=True,
            lora_config=lora_config,
        )
        eval_student.load()

        # Seed eval model base with distilled LoRA weights from cuda:0 student
        lora_mgr_train = student.lora_manager
        lora_mgr_eval  = eval_student.lora_manager
        base_lora_cpu  = {k: v.cpu() for k, v in lora_mgr_train.get_lora_state_dict().items()}
        lora_mgr_eval.set_lora_state_dict({k: v.to(args.eval_device) for k, v in base_lora_cpu.items()}, strict=False)
        console.print(f"[green]Eval model ready on {args.eval_device} "
                      f"({len(base_lora_cpu)} LoRA tensors, ~{sum(v.nbytes for v in base_lora_cpu.values())//1024**2} MB).[/green]")

        eval_fitness = PerplexityFitness(dataloader=eval_loader, device=args.eval_device)

        # Seed population with LoRA-only state dicts (CPU)
        seed_states = [base_lora_cpu] + [
            _mutate(base_lora_cpu, mutation_rate=1.0, mutation_scale=args.mutation_scale)
            for _ in range(args.pop_size - 1)
        ]

        genetics = Genetics(
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            adaptive_mutation=True,
            mutation_decay=0.98,
            min_mutation_rate=0.05,
        )
        population = Population(size=args.pop_size, genetics=genetics, elite_size=args.elite_size)
        population.initialize_from_state_dicts(seed_states)
        lora_mb = sum(v.nbytes for v in base_lora_cpu.values()) * args.pop_size // 1024**2
        console.print(f"[cyan]Population of {args.pop_size} LoRA-only individuals initialised "
                      f"(total CPU footprint ~{lora_mb} MB).[/cyan]")

        def fitness_fn(lora_state):
            # Inject LoRA weights onto cuda:1 eval model; base weights stay fixed
            lora_mgr_eval.set_lora_state_dict(
                {k: v.to(args.eval_device) for k, v in lora_state.items()}, strict=False
            )
            return eval_fitness.evaluate(eval_student.model, state_dict=None).score

        for gen in range(1, args.generations + 1):
            console.print(f"  [cyan]Generation {gen}/{args.generations}…[/cyan]")
            population.evaluate(fitness_fn)

            best_ind = population.best
            avg_fit  = population.average_fitness
            div      = _cpu_diversity(population)
            mut_rate = genetics._get_current_mutation_rate()

            dashboard.update_evolution(
                gen=gen,
                best_fitness=best_ind.fitness,
                avg_fitness=avg_fit,
                diversity=div,
                mutation_rate=mut_rate,
            )

            best_fitness     = best_ind.fitness
            generations_done = gen

            if gen < args.generations:
                population.evolve()

        # Load best LoRA weights back onto cuda:0 student for saving
        lora_mgr_train.set_lora_state_dict(
            {k: v.to(args.device) for k, v in population.best.state_dict.items()}, strict=False
        )
        eval_student.unload()
        console.print(f"[green]Evolution done. Best fitness={best_fitness:.4f}[/green]")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    console.print("\n[bold]Saving plots and report…[/bold]")
    dashboard.save_plots(str(output_dir))
    dashboard.save_report(str(output_dir))

    # Save final student checkpoint
    checkpoint_dir = output_dir / "checkpoint-final"
    student.save(str(checkpoint_dir))
    console.print(f"[green]Final checkpoint saved to {checkpoint_dir}[/green]")

    # ------------------------------------------------------------------
    # 8. Summary table
    # ------------------------------------------------------------------
    _print_summary(
        args=args,
        output_dir=output_dir,
        distill_results=distill_results,
        generations_done=generations_done,
        best_fitness=best_fitness,
    )


def _print_summary(
    args,
    output_dir: Path,
    distill_results: dict,
    generations_done: int,
    best_fitness: float,
) -> None:
    console.rule("[bold green]Training Summary[/bold green]")

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total distillation steps", str(distill_results["global_step"]))
    table.add_row(
        "Final distillation loss",
        f"{distill_results['training_logs'][-1]['loss']:.4f}"
        if distill_results["training_logs"]
        else "—",
    )
    table.add_row("Best eval loss", f"{distill_results['best_eval_loss']:.4f}")
    table.add_row("Generations completed", str(generations_done))
    table.add_row("Best fitness", f"{best_fitness:.4f}" if generations_done > 0 else "—")
    table.add_row("Output directory", str(output_dir))
    table.add_row(
        "TensorBoard",
        f"tensorboard --logdir {output_dir / 'tensorboard'}",
    )

    console.print(table)


if __name__ == "__main__":
    main()
