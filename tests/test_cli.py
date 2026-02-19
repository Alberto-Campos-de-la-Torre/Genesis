"""Tests for the genesis CLI parser (genesis/cli.py)."""

import sys
import pytest
import importlib.util
from pathlib import Path


# Load cli module in isolation — avoids pulling in torch/transformers at import time
def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "genesis.cli",
        Path(__file__).parent.parent / "genesis" / "cli.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["genesis.cli"] = mod
    spec.loader.exec_module(mod)
    return mod


CLI = _load_cli()


# ── Parser construction ───────────────────────────────────────────────────────────

class TestParserConstruction:

    def test_build_parser_returns_parser(self):
        p = CLI.build_parser()
        assert p is not None

    def test_subcommand_required(self):
        p = CLI.build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])  # no subcommand → error


# ── `run` subcommand ─────────────────────────────────────────────────────────────

class TestRunSubcommand:

    def _parse(self, *args):
        return CLI.build_parser().parse_args(["run", *args])

    def test_minimal(self):
        ns = self._parse()
        assert ns.command == "run"

    def test_config_flag(self):
        ns = self._parse("--config", "experiment.yaml")
        assert ns.config == "experiment.yaml"

    def test_config_short_flag(self):
        ns = self._parse("-c", "cfg.yaml")
        assert ns.config == "cfg.yaml"

    def test_output_dir_flag(self):
        ns = self._parse("--output-dir", "/tmp/out")
        assert ns.output_dir == "/tmp/out"

    def test_generations_flag(self):
        ns = self._parse("--generations", "25")
        assert ns.generations == 25

    def test_generations_short_flag(self):
        ns = self._parse("-g", "10")
        assert ns.generations == 10

    def test_population_size_flag(self):
        ns = self._parse("--population-size", "15")
        assert ns.population_size == 15

    def test_teacher_model_flag(self):
        ns = self._parse("--teacher-model", "gpt2")
        assert ns.teacher_model == "gpt2"

    def test_student_model_flag(self):
        ns = self._parse("--student-model", "distilgpt2")
        assert ns.student_model == "distilgpt2"

    def test_no_lora_flag(self):
        ns = self._parse("--no-lora")
        assert ns.no_lora is True

    def test_no_lora_absent_is_false(self):
        ns = self._parse()
        assert ns.no_lora is False

    def test_all_flags_together(self):
        ns = self._parse(
            "--config", "cfg.yaml",
            "--output-dir", "/out",
            "--generations", "5",
            "--population-size", "8",
            "--teacher-model", "gpt2",
            "--student-model", "distilgpt2",
            "--no-lora",
        )
        assert ns.config == "cfg.yaml"
        assert ns.output_dir == "/out"
        assert ns.generations == 5
        assert ns.population_size == 8
        assert ns.teacher_model == "gpt2"
        assert ns.student_model == "distilgpt2"
        assert ns.no_lora is True

    def test_func_is_callable(self):
        ns = self._parse()
        assert callable(ns.func)


# ── `distill` subcommand ──────────────────────────────────────────────────────────

class TestDistillSubcommand:

    def _parse(self, *args):
        return CLI.build_parser().parse_args(["distill", *args])

    def test_minimal(self):
        ns = self._parse()
        assert ns.command == "distill"

    def test_config_flag(self):
        ns = self._parse("-c", "d.yaml")
        assert ns.config == "d.yaml"

    def test_steps_flag(self):
        ns = self._parse("--steps", "500")
        assert ns.steps == 500

    def test_steps_short_flag(self):
        ns = self._parse("-s", "100")
        assert ns.steps == 100

    def test_teacher_model_flag(self):
        ns = self._parse("--teacher-model", "llama")
        assert ns.teacher_model == "llama"

    def test_student_model_flag(self):
        ns = self._parse("--student-model", "small")
        assert ns.student_model == "small"

    def test_func_is_callable(self):
        ns = self._parse()
        assert callable(ns.func)


# ── `prune` subcommand ────────────────────────────────────────────────────────────

class TestPruneSubcommand:

    def _parse(self, *args):
        return CLI.build_parser().parse_args(["prune", *args])

    def test_minimal(self):
        ns = self._parse()
        assert ns.command == "prune"

    def test_sparsity_flag(self):
        ns = self._parse("--sparsity", "0.3")
        assert ns.sparsity == pytest.approx(0.3)

    def test_method_flag(self):
        for method in ["magnitude", "gradient", "taylor", "fisher"]:
            ns = self._parse("--method", method)
            assert ns.method == method

    def test_invalid_method_raises(self):
        with pytest.raises(SystemExit):
            CLI.build_parser().parse_args(["prune", "--method", "unknown"])

    def test_config_flag(self):
        ns = self._parse("--config", "p.yaml")
        assert ns.config == "p.yaml"

    def test_student_model_flag(self):
        ns = self._parse("--student-model", "mymodel")
        assert ns.student_model == "mymodel"

    def test_func_is_callable(self):
        ns = self._parse()
        assert callable(ns.func)


# ── `info` subcommand ─────────────────────────────────────────────────────────────

class TestInfoSubcommand:

    def _parse(self, *args):
        return CLI.build_parser().parse_args(["info", *args])

    def test_minimal(self):
        ns = self._parse()
        assert ns.command == "info"

    def test_func_is_callable(self):
        ns = self._parse()
        assert callable(ns.func)


# ── Global flags ──────────────────────────────────────────────────────────────────

class TestGlobalFlags:

    def test_log_level_default(self):
        ns = CLI.build_parser().parse_args(["info"])
        assert ns.log_level == "INFO"

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_log_level_choices(self, level):
        ns = CLI.build_parser().parse_args(["--log-level", level, "info"])
        assert ns.log_level == level

    def test_invalid_log_level_raises(self):
        with pytest.raises(SystemExit):
            CLI.build_parser().parse_args(["--log-level", "TRACE", "info"])


# ── Each subcommand maps to the right function ────────────────────────────────────

class TestSubcommandFunctions:

    @pytest.mark.parametrize("cmd,fn_name", [
        (["run"], "_cmd_run"),
        (["distill"], "_cmd_distill"),
        (["prune"], "_cmd_prune"),
        (["info"], "_cmd_info"),
    ])
    def test_func_name(self, cmd, fn_name):
        ns = CLI.build_parser().parse_args(cmd)
        assert ns.func.__name__ == fn_name
