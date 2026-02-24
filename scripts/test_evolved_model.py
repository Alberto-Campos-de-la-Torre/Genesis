#!/usr/bin/env python3
"""Evaluate and compare the best evolved model from a Genesis training run.

Usage:
    python scripts/test_evolved_model.py                          # latest run
    python scripts/test_evolved_model.py --run-dir outputs/run_X  # specific run
    python scripts/test_evolved_model.py --run-dir outputs/run_X --prompts 5

Exit code:
    0  evolved model perplexity ≤ base model perplexity  (improvement / no regression)
    1  evolved model is worse than base model            (regression detected)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.table import Table

console = Console()

STUDENT_PATH = "/media/ttech-main/42A4266DA426639F/Models/Qwen3-1.7B"

GENERATION_PROMPTS = [
    "The history of artificial intelligence begins",
    "In quantum mechanics, the uncertainty principle states",
    "The primary causes of the French Revolution were",
    "To train a neural network effectively, one must",
    "The Amazon rainforest is important because",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_run_dir(run_dir: str | None) -> Path:
    """Return the requested run dir, or auto-detect the latest one."""
    outputs = Path("outputs")
    if run_dir:
        path = Path(run_dir)
        if not path.exists():
            sys.exit(f"[error] run dir not found: {path}")
        return path
    runs = sorted(
        [d for d in outputs.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not runs:
        sys.exit("[error] No run dirs found under outputs/")
    return runs[-1]


def _best_checkpoint(run_dir: Path) -> Path:
    """Return checkpoint-best if it exists, else checkpoint-final."""
    best = run_dir / "checkpoint-best"
    final = run_dir / "checkpoint-final"
    if best.exists() and (best / "adapter_model.safetensors").exists():
        return best
    if final.exists() and (final / "adapter_model.safetensors").exists():
        return final
    sys.exit(f"[error] No valid checkpoint found in {run_dir}")


def _load_report(run_dir: Path) -> dict:
    report_path = run_dir / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {}


def _build_eval_loader(tokenizer, seq_len: int, batch_size: int):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    eval_texts = [raw[i]["text"] for i in range(200, min(250, len(raw)))
                  if raw[i]["text"].strip()]
    enc = tokenizer(
        eval_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)

    def collate(batch):
        ids, masks, lbls = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(lbls),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


@torch.no_grad()
def _evaluate_perplexity(model, dataloader, device: str) -> tuple[float, float]:
    """Return (mean_ce_loss, perplexity) on the given dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # mean CE over non-masked positions
        num_tokens = (labels != -100).sum().item()
        if num_tokens > 0:
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 20))   # cap at exp(20) to avoid overflow display
    return avg_loss, ppl


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, device: str, max_new: int = 60) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(run_dir_arg: str | None, num_prompts: int, seq_len: int, batch_size: int) -> int:
    run_dir    = _find_run_dir(run_dir_arg)
    ckpt_path  = _best_checkpoint(run_dir)
    report     = _load_report(run_dir)

    console.rule(f"[bold blue]Genesis Model Evaluation[/bold blue]")
    console.print(f"Run dir    : [green]{run_dir}[/green]")
    console.print(f"Checkpoint : [cyan]{ckpt_path.name}[/cyan]")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── 1. Load base model (no LoRA) ────────────────────────────────────
    console.print("\n[bold]Loading base model (no LoRA)…[/bold]")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_PATH, torch_dtype=torch.bfloat16
    ).to(device)

    eval_loader = _build_eval_loader(tokenizer, seq_len, batch_size)
    console.print(f"[dim]Eval set: {len(eval_loader.dataset)} samples, seq_len={seq_len}[/dim]")

    console.print("[bold]Evaluating base model…[/bold]")
    base_loss, base_ppl = _evaluate_perplexity(base_model, eval_loader, device)
    base_fitness = 1.0 / (1.0 + base_ppl)

    # ── 2. Load evolved model (base + LoRA adapter) ──────────────────────
    console.print("[bold]Loading evolved model (base + LoRA adapter)…[/bold]")
    from peft import PeftModel

    evolved_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
    evolved_model = evolved_model.to(device)

    console.print("[bold]Evaluating evolved model…[/bold]")
    evo_loss, evo_ppl = _evaluate_perplexity(evolved_model, eval_loader, device)
    evo_fitness = 1.0 / (1.0 + evo_ppl)

    # ── 3. Metrics table ─────────────────────────────────────────────────
    delta_ppl  = evo_ppl  - base_ppl
    delta_loss = evo_loss - base_loss
    improved   = evo_ppl <= base_ppl

    console.rule("[bold green]Results[/bold green]")
    t = Table(show_header=True, header_style="bold magenta", expand=False)
    t.add_column("Metric",        style="cyan",  no_wrap=True)
    t.add_column("Base model",    style="white", justify="right")
    t.add_column("Evolved model", style="white", justify="right")
    t.add_column("Δ",             justify="right")

    def _delta(v, invert=False):
        good = v < 0 if not invert else v > 0
        color = "green" if good else "red"
        sign  = "+" if v >= 0 else ""
        return f"[{color}]{sign}{v:.4f}[/{color}]"

    t.add_row("CE loss",  f"{base_loss:.4f}", f"{evo_loss:.4f}",  _delta(delta_loss))
    t.add_row("PPL",      f"{base_ppl:.2f}",  f"{evo_ppl:.2f}",   _delta(delta_ppl))
    t.add_row("Fitness",  f"{base_fitness:.6f}", f"{evo_fitness:.6f}", _delta(evo_fitness - base_fitness, invert=True))

    # Training report summary — evolution is a list of per-gen dicts
    if report:
        evo_gens = report.get("evolution", [])
        if evo_gens:
            reported_best = max(g.get("best_fitness", 0) for g in evo_gens)
            t.add_row("Reported best fitness (train)", "—", f"{reported_best:.6f}", "")

    console.print(t)

    verdict = "[bold green]PASS — evolved ≤ base perplexity[/bold green]" \
              if improved else \
              "[bold red]FAIL — evolved > base perplexity (regression)[/bold red]"
    console.print(f"\nVerdict: {verdict}")

    # ── 4. Generation comparison ─────────────────────────────────────────
    prompts = GENERATION_PROMPTS[:num_prompts]
    if prompts:
        console.rule("[bold yellow]Generation Comparison[/bold yellow]")
        gen_table = Table(show_header=True, header_style="bold", expand=True)
        gen_table.add_column("Prompt",        style="cyan",  width=30, no_wrap=True)
        gen_table.add_column("Base model",    style="white", width=55)
        gen_table.add_column("Evolved model", style="green", width=55)

        for prompt in prompts:
            base_gen = _generate(base_model,    tokenizer, prompt, device)
            evo_gen  = _generate(evolved_model, tokenizer, prompt, device)
            gen_table.add_row(
                prompt[:28] + "…" if len(prompt) > 28 else prompt,
                base_gen[:200],
                evo_gen[:200],
            )

        console.print(gen_table)

    return 0 if improved else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate best evolved Genesis model")
    parser.add_argument(
        "--run-dir", default=None,
        help="Training run directory (default: latest outputs/run_*)"
    )
    parser.add_argument("--prompts",    type=int, default=3,   help="Generation prompts to show (0=skip)")
    parser.add_argument("--seq-len",    type=int, default=128, help="Token sequence length for eval")
    parser.add_argument("--batch-size", type=int, default=4,   help="Eval batch size")
    args = parser.parse_args()

    exit_code = evaluate(args.run_dir, args.prompts, args.seq_len, args.batch_size)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
