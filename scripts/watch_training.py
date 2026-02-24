#!/usr/bin/env python3
"""Live Genesis training monitor.

Usage:
    python scripts/watch_training.py                          # default /tmp/genesis_training.log
    python scripts/watch_training.py /path/to/output.log     # explicit file
"""

import re
import sys
import time
import argparse
from pathlib import Path

DEFAULT_LOG = "/tmp/genesis_training.log"

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.console import Console
    from rich.rule import Rule
except ImportError:
    sys.exit("rich not found — run: .venv/bin/pip install rich")

console = Console()


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_header(lines: list[str]) -> dict:
    """Extract total_steps and total_gens from the run header line."""
    total_steps, total_gens, pop_size = 0, 0, 0
    for l in lines:
        m = re.search(r"Steps\s*:\s*(\d+)", l)
        if m:
            total_steps = int(m.group(1))
        m = re.search(r"Generations:\s*(\d+)", l)
        if m:
            total_gens = int(m.group(1))
        m = re.search(r"Pop:\s*(\d+)", l)
        if m:
            pop_size = int(m.group(1))
        if total_steps and total_gens:
            break
    return {"total_steps": total_steps or 1, "total_gens": total_gens or 1, "pop_size": pop_size}


def _parse_distillation(lines: list[str]) -> dict:
    """Parse step losses and eval losses."""
    step_lines = [l for l in lines if re.search(r"Step \d+: loss=", l)]
    eval_lines = [l for l in lines if "Evaluation:" in l and "loss" in l]

    step_num = loss = None
    if step_lines:
        last = step_lines[-1]
        m = re.search(r"Step (\d+):", last)
        if m:
            step_num = int(m.group(1))
        m = re.search(r"loss=([\d.]+)", last)
        if m:
            loss = float(m.group(1))

    # Collect all per-step losses for sparkline
    all_losses = []
    for l in step_lines[-15:]:
        m = re.search(r"loss=([\d.]+)", l)
        if m:
            all_losses.append(float(m.group(1)))

    # Best eval loss
    best_eval = None
    eval_losses = []
    for l in eval_lines:
        m = re.search(r"'loss':\s*([\d.]+)", l)
        if m:
            v = float(m.group(1))
            eval_losses.append(v)
            if best_eval is None or v < best_eval:
                best_eval = v

    # Progress bar line (last tqdm line)
    prog = ""
    for l in reversed(lines):
        if ("it/s" in l or "s/it" in l) and "Training" in l:
            prog = l.strip()
            break

    return dict(
        step_num=step_num,
        loss=loss,
        all_losses=all_losses,
        best_eval=best_eval,
        eval_losses=eval_losses,
        prog=prog,
    )


def _parse_evolution(lines: list[str]) -> dict:
    """Parse evolutionary generation stats."""
    gen_lines = [l for l in lines if re.search(r"Generation \d+: Best=", l)]

    gen_num = best_fit = avg_fit = None
    if gen_lines:
        last = gen_lines[-1]
        m = re.search(r"Generation (\d+):", last)
        if m:
            gen_num = int(m.group(1))
        m = re.search(r"Best=([\d.]+)", last)
        if m:
            best_fit = float(m.group(1))
        m = re.search(r"Avg=([\d.]+)", last)
        if m:
            avg_fit = float(m.group(1))

    all_fits = []
    for l in gen_lines:
        m = re.search(r"Best=([\d.]+)", l)
        if m:
            all_fits.append(float(m.group(1)))

    return dict(gen_num=gen_num, best_fit=best_fit, avg_fit=avg_fit, all_fits=all_fits)


def _parse_teacher(lines: list[str], data: str) -> dict:
    """Parse teacher / logprobs status."""
    logprobs_ok = any("logprobs: SUPPORTED" in l for l in lines)
    logprobs_no = any("logprobs: NOT SUPPORTED" in l for l in lines)

    mode = "unknown"
    if logprobs_ok:
        mode = "KD soft-target"
    elif logprobs_no:
        mode = "hard-label CE only"

    return dict(
        mode=mode,
        fallbacks=data.count("Switching to fallback"),
        tmr=data.count("Too Many Requests"),
        exhausted=data.count("All teacher models exhausted"),
    )


def parse(data: str) -> dict:
    lines = data.splitlines()
    header  = _parse_header(lines)
    distill = _parse_distillation(lines)
    evo     = _parse_evolution(lines)
    teacher = _parse_teacher(lines, data)
    done    = "Evolution done" in data or "checkpoint-final" in data
    return {**header, **distill, **evo, **teacher, "done": done, "lines": lines}


# ── Sparkline helper ──────────────────────────────────────────────────────────

def _spark(values: list[float], width: int = 12) -> str:
    if len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    blocks = "▁▂▃▄▅▆▇█"
    return " ".join(blocks[min(7, int(7 * (v - mn) / (mx - mn + 1e-9)))] for v in values[-width:])


# ── Panel builder ─────────────────────────────────────────────────────────────

def build_panel(d: dict, log_path: str) -> Panel:
    total_steps = d["total_steps"]
    total_gens  = d["total_gens"]

    # ── Distillation ─────────────────────────────────────────────────────
    dt = Table(show_header=False, box=None, padding=(0, 1))
    dt.add_column("k", style="cyan", no_wrap=True)
    dt.add_column("v", style="green")

    if d["step_num"] is not None:
        pct = 100 * d["step_num"] / total_steps
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        dt.add_row("Step",      f"{d['step_num']} / {total_steps}  [{bar}] {pct:.1f}%")
        dt.add_row("Loss",      f"{d['loss']:.4f}" if d["loss"] is not None else "—")
        spark = _spark(d["all_losses"])
        if spark:
            dt.add_row("Trend", spark)
        if d["best_eval"] is not None:
            dt.add_row("Best eval loss", f"[bold]{d['best_eval']:.4f}[/bold]")
            if len(d["eval_losses"]) > 1:
                dt.add_row("Eval trend", _spark(d["eval_losses"]))
    else:
        phase = "distillation starting…"
        if any("Phase 1" in l for l in d["lines"]):
            phase = "loading model…"
        dt.add_row("Status", phase)

    if d["prog"]:
        dt.add_row("Progress", d["prog"][-80:])

    # ── Evolution ────────────────────────────────────────────────────────
    et = Table(show_header=False, box=None, padding=(0, 1))
    et.add_column("k", style="cyan", no_wrap=True)
    et.add_column("v", style="magenta")

    if d["gen_num"] is not None:
        pct = 100 * (d["gen_num"] + 1) / total_gens
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        et.add_row("Gen",          f"{d['gen_num'] + 1} / {total_gens}  [{bar}] {pct:.1f}%")
        et.add_row("Best fitness", f"{d['best_fit']:.6f}" if d["best_fit"] is not None else "—")
        et.add_row("Avg fitness",  f"{d['avg_fit']:.6f}"  if d["avg_fit"]  is not None else "—")
        if d["best_fit"] is not None:
            ppl = (1.0 / d["best_fit"] - 1.0) if d["best_fit"] > 0 else float("inf")
            et.add_row("Best PPL", f"~{ppl:.1f}")
        spark = _spark(d["all_fits"])
        if spark:
            et.add_row("Trend", spark)
    elif d["done"]:
        et.add_row("Status", "[bold green]complete[/bold green]")
    else:
        et.add_row("Status", "waiting for distillation…")

    # ── Teacher / mode ───────────────────────────────────────────────────
    tt = Table(show_header=False, box=None, padding=(0, 1))
    tt.add_column("k", style="cyan", no_wrap=True)
    tt.add_column("v")

    mode_color = "green" if d["mode"] == "KD soft-target" else "yellow"
    tt.add_row("Training mode", Text(d["mode"], style=f"bold {mode_color}"))

    color_fb  = "green" if d["fallbacks"] == 0 else "yellow"
    color_tmr = "green" if d["tmr"] == 0 else "red"
    color_ex  = "green" if d["exhausted"] == 0 else "red"
    tt.add_row("Fallbacks",        Text(str(d["fallbacks"]), style=color_fb))
    tt.add_row("429s",             Text(str(d["tmr"]),       style=color_tmr))
    tt.add_row("Full exhaustions", Text(str(d["exhausted"]), style=color_ex))

    status = "[bold green]DONE[/bold green]" if d["done"] else "[yellow]running…[/yellow]"
    tt.add_row("Run status", status)

    return Panel(
        Columns([
            Panel(dt, title="[bold yellow]Distillation[/bold yellow]",  width=58),
            Panel(et, title="[bold magenta]Evolution[/bold magenta]",   width=42),
            Panel(tt, title="[bold cyan]Teacher / Mode[/bold cyan]",    width=32),
        ]),
        title="[bold blue]Genesis Training Monitor[/bold blue]",
        subtitle=f"[dim]{log_path}  |  refresh: 2s  |  Ctrl-C to exit[/dim]",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live Genesis training monitor")
    parser.add_argument(
        "log",
        nargs="?",
        default=DEFAULT_LOG,
        help=f"Path to training log file (default: {DEFAULT_LOG})",
    )
    args = parser.parse_args()

    log_path = args.log
    path = Path(log_path)

    with Live(console=console, refresh_per_second=0.5, screen=False) as live:
        while True:
            try:
                data = path.read_text(errors="replace")
                d = parse(data)
                live.update(build_panel(d, log_path))
                if d["done"]:
                    time.sleep(2)
                    break
            except FileNotFoundError:
                live.update(Panel(
                    f"[yellow]Waiting for log file:[/yellow]\n[dim]{log_path}[/dim]\n\n"
                    "[dim]Start training with:[/dim]\n"
                    "[cyan]OLLAMA_HOST=http://192.168.1.97:11434 OLLAMA_MODEL=qwen3.5:cloud \\\n"
                    "  .venv/bin/python scripts/train_qwen3.py --steps 100 --generations 10 "
                    "--pop-size 4 > /tmp/genesis_training.log 2>&1 &[/cyan]",
                    title="[bold blue]Genesis Training Monitor[/bold blue]",
                ))
            except KeyboardInterrupt:
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
