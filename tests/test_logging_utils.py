"""Tests for genesis/utils/logging.py and genesis/data/preprocessing.py."""

import logging
import pytest

from genesis.utils.logging import (
    setup_logging,
    get_logger,
    LoggerAdapter,
    TrainingLogger,
    ProgressLogger,
)
from genesis.data.preprocessing import TextPreprocessor


# ── setup_logging ─────────────────────────────────────────────────────────────

class TestSetupLogging:

    def test_returns_logger(self):
        logger = setup_logging(name="test_setup_unique")
        assert isinstance(logger, logging.Logger)

    def test_log_level_debug(self):
        logger = setup_logging(log_level="DEBUG", name="test_debug_unique")
        assert logger.level == logging.DEBUG

    def test_log_level_warning(self):
        logger = setup_logging(log_level="WARNING", name="test_warn_unique")
        assert logger.level == logging.WARNING

    def test_file_handler_created(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = setup_logging(log_file=log_file, name="test_file_unique")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "FileHandler" in handler_types
        assert (tmp_path / "test.log").exists()

    def test_log_dir_creates_timestamped_file(self, tmp_path):
        setup_logging(log_dir=str(tmp_path), name="test_logdir_unique")
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1

    def test_console_handler_always_added(self):
        logger = setup_logging(name="test_console_unique2")
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types


# ── get_logger ────────────────────────────────────────────────────────────────

class TestGetLogger:

    def test_returns_logger_instance(self):
        logger = get_logger("test_get_unique")
        assert isinstance(logger, logging.Logger)

    def test_same_name_returns_same_logger(self):
        l1 = get_logger("test_same_unique")
        l2 = get_logger("test_same_unique")
        assert l1 is l2


# ── LoggerAdapter ─────────────────────────────────────────────────────────────

class TestLoggerAdapter:

    def test_prefix_prepended(self):
        base = logging.getLogger("test_adapter_base")
        adapter = LoggerAdapter(base, prefix="GEN5")
        msg, _ = adapter.process("hello", {})
        assert msg == "[GEN5] hello"

    def test_empty_prefix_no_change(self):
        base = logging.getLogger("test_adapter_noprefix")
        adapter = LoggerAdapter(base, prefix="")
        msg, _ = adapter.process("hello", {})
        assert msg == "hello"


# ── ProgressLogger ────────────────────────────────────────────────────────────

class TestProgressLogger:

    def test_log_generation_stores_entry(self):
        pl = ProgressLogger(total_generations=10)
        pl.log_generation(1, best_fitness=0.7, avg_fitness=0.5)
        assert len(pl.get_history()) == 1

    def test_history_entry_has_required_keys(self):
        pl = ProgressLogger(total_generations=10)
        pl.log_generation(1, best_fitness=0.7, avg_fitness=0.5, diversity=0.3)
        entry = pl.get_history()[0]
        assert entry["generation"] == 1
        assert entry["best_fitness"] == pytest.approx(0.7)
        assert entry["avg_fitness"] == pytest.approx(0.5)
        assert entry["diversity"] == pytest.approx(0.3)

    def test_extra_kwargs_stored_in_history(self):
        pl = ProgressLogger(total_generations=5)
        pl.log_generation(0, best_fitness=0.5, avg_fitness=0.4, custom=42)
        assert pl.get_history()[0]["custom"] == 42

    def test_multiple_generations_logged(self):
        pl = ProgressLogger(total_generations=5)
        for i in range(5):
            pl.log_generation(i, best_fitness=float(i), avg_fitness=0.0)
        assert len(pl.get_history()) == 5


# ── TrainingLogger ────────────────────────────────────────────────────────────

class TestTrainingLogger:

    def test_creates_without_log_dir(self):
        tl = TrainingLogger(name="test_tl_nodisk", use_tensorboard=False)
        assert tl.tb_writer is None

    def test_log_metrics_does_not_raise(self):
        tl = TrainingLogger(name="test_tl_metrics", use_tensorboard=False)
        tl.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)

    def test_close_does_not_raise(self):
        tl = TrainingLogger(name="test_tl_close", use_tensorboard=False)
        tl.close()


# ── TextPreprocessor ──────────────────────────────────────────────────────────

class TestTextPreprocessor:

    def test_lowercase(self):
        tp = TextPreprocessor(lowercase=True)
        assert tp.preprocess("Hello WORLD") == "hello world"

    def test_normalize_whitespace(self):
        tp = TextPreprocessor(normalize_whitespace=True)
        assert tp.preprocess("  hello   world  ") == "hello world"

    def test_remove_special_chars(self):
        tp = TextPreprocessor(remove_special_chars=True)
        result = tp.preprocess("Hello, @world! #test")
        assert "@" not in result
        assert "#" not in result

    def test_no_ops_by_default(self):
        tp = TextPreprocessor(lowercase=False, remove_special_chars=False,
                              normalize_whitespace=True)
        assert tp.preprocess("Hello World") == "Hello World"

    def test_all_options_combined(self):
        tp = TextPreprocessor(lowercase=True, remove_special_chars=True,
                              normalize_whitespace=True)
        result = tp.preprocess("  HELLO @World!  ")
        assert result == result.lower()
        assert "@" not in result

    def test_tokenize_without_tokenizer_raises(self):
        tp = TextPreprocessor()
        with pytest.raises(ValueError):
            tp.tokenize("hello world")

    def test_batch_tokenize_without_tokenizer_raises(self):
        tp = TextPreprocessor()
        with pytest.raises(ValueError):
            tp.batch_tokenize(["hello", "world"])
