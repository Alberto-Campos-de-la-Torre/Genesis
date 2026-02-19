"""Tests for genesis/utils/metrics.py."""

import pytest
import torch
import numpy as np

from genesis.utils.metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_bleu,
    compute_rouge,
    MetricsTracker,
    FitnessTracker,
)


# ── compute_perplexity ────────────────────────────────────────────────────────

class TestComputePerplexity:

    def test_loss_zero_gives_perplexity_one(self):
        assert compute_perplexity(0.0) == pytest.approx(1.0)

    def test_tensor_input(self):
        loss = torch.tensor(1.0)
        result = compute_perplexity(loss)
        assert isinstance(result, float)
        assert result == pytest.approx(np.e)

    def test_higher_loss_higher_perplexity(self):
        assert compute_perplexity(2.0) > compute_perplexity(1.0)

    def test_base_2(self):
        result = compute_perplexity(1.0, base=2.0)
        assert result == pytest.approx(2.0)


# ── compute_accuracy ──────────────────────────────────────────────────────────

class TestComputeAccuracy:

    def test_perfect_accuracy(self):
        preds = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3])
        assert compute_accuracy(preds, labels) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        preds = torch.tensor([1, 2, 3, 0])
        labels = torch.tensor([0, 1, 2, 3])
        assert compute_accuracy(preds, labels) == pytest.approx(0.0)

    def test_half_accuracy(self):
        preds = torch.tensor([0, 0, 2, 3])
        labels = torch.tensor([0, 1, 2, 4])
        # 0==0 correct, 0!=1 wrong, 2==2 correct, 3!=4 wrong → 2/4 = 0.5
        assert compute_accuracy(preds, labels) == pytest.approx(0.5)

    def test_ignore_index_excluded(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, -100, 2])
        # Only positions 0 and 2 count (both correct)
        assert compute_accuracy(preds, labels) == pytest.approx(1.0)

    def test_all_ignored_returns_zero(self):
        preds = torch.tensor([0, 1])
        labels = torch.tensor([-100, -100])
        assert compute_accuracy(preds, labels) == pytest.approx(0.0)


# ── compute_bleu ──────────────────────────────────────────────────────────────

class TestComputeBleu:

    def test_identical_strings_bleu4_one(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = compute_bleu(preds, refs)
        assert result["bleu_4"] == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap_bleu_zero(self):
        preds = ["abc def ghi jkl"]
        refs = ["xyz uvw rst opq"]
        result = compute_bleu(preds, refs, smooth=False)
        assert result["bleu_1"] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        preds = ["hello world"]
        refs = ["hello world"]
        result = compute_bleu(preds, refs)
        for key in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "brevity_penalty"]:
            assert key in result

    def test_brevity_penalty_for_short_prediction(self):
        preds = ["cat"]
        refs = ["the cat sat on the mat"]
        result = compute_bleu(preds, refs)
        assert result["brevity_penalty"] < 1.0

    def test_bleu_scores_in_range(self):
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        result = compute_bleu(preds, refs)
        for n in range(1, 5):
            assert 0.0 <= result[f"bleu_{n}"] <= 1.0


# ── compute_rouge ─────────────────────────────────────────────────────────────

class TestComputeRouge:

    def test_identical_strings_rouge_one(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = compute_rouge(preds, refs)
        assert result["rouge_1"] == pytest.approx(1.0, abs=1e-4)
        assert result["rouge_l"] == pytest.approx(1.0, abs=1e-4)

    def test_returns_all_keys(self):
        preds = ["hello world"]
        refs = ["hello world"]
        result = compute_rouge(preds, refs)
        assert "rouge_1" in result
        assert "rouge_2" in result
        assert "rouge_l" in result

    def test_scores_in_range(self):
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        result = compute_rouge(preds, refs)
        for key in ["rouge_1", "rouge_2", "rouge_l"]:
            assert 0.0 <= result[key] <= 1.0

    def test_no_overlap_is_zero(self):
        preds = ["xyz abc"]
        refs = ["hello world"]
        result = compute_rouge(preds, refs)
        assert result["rouge_1"] == pytest.approx(0.0, abs=1e-6)


# ── MetricsTracker ────────────────────────────────────────────────────────────

class TestMetricsTracker:

    def test_update_and_get_latest(self):
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5})
        assert tracker.get_latest("loss") == pytest.approx(0.5)

    def test_get_average_all(self):
        tracker = MetricsTracker()
        tracker.update({"loss": 1.0})
        tracker.update({"loss": 3.0})
        assert tracker.get_average("loss") == pytest.approx(2.0)

    def test_get_average_with_window(self):
        tracker = MetricsTracker()
        for v in [10.0, 1.0, 1.0]:
            tracker.update({"loss": v})
        # Window of 2: average of last two values = 1.0
        assert tracker.get_average("loss", window=2) == pytest.approx(1.0)

    def test_unknown_metric_returns_zero(self):
        tracker = MetricsTracker()
        assert tracker.get_latest("nonexistent") == 0.0
        assert tracker.get_average("nonexistent") == 0.0

    def test_step_increments(self):
        tracker = MetricsTracker()
        assert tracker.step == 0
        tracker.update({"loss": 1.0})
        tracker.update({"loss": 2.0})
        assert tracker.step == 2

    def test_tensor_value_converted(self):
        tracker = MetricsTracker()
        tracker.update({"loss": torch.tensor(0.42)})
        assert isinstance(tracker.get_latest("loss"), float)

    def test_reset_clears_all(self):
        tracker = MetricsTracker()
        tracker.update({"loss": 1.0})
        tracker.reset()
        assert tracker.get_latest("loss") == 0.0
        assert tracker.step == 0

    def test_get_summary_structure(self):
        tracker = MetricsTracker()
        tracker.update({"loss": 1.0, "acc": 0.8})
        tracker.update({"loss": 0.5, "acc": 0.9})
        summary = tracker.get_summary()
        assert "loss" in summary
        assert "acc" in summary
        for key in ["mean", "std", "min", "max", "latest", "count"]:
            assert key in summary["loss"]

    def test_get_all_returns_copy(self):
        tracker = MetricsTracker()
        tracker.update({"x": 1.0})
        vals = tracker.get_all("x")
        vals.append(99.0)
        assert tracker.get_all("x") == [1.0]


# ── FitnessTracker ────────────────────────────────────────────────────────────

class TestFitnessTracker:

    def test_update_and_history_length(self):
        ft = FitnessTracker()
        for i in range(5):
            ft.update(generation=i, best_fitness=float(i), avg_fitness=float(i) * 0.5)
        assert len(ft.best_fitness_history) == 5
        assert len(ft.generation_history) == 5

    def test_diversity_optional(self):
        ft = FitnessTracker()
        ft.update(generation=0, best_fitness=0.5, avg_fitness=0.3)
        assert ft.diversity_history == []
        ft.update(generation=1, best_fitness=0.6, avg_fitness=0.4, diversity=0.7)
        assert len(ft.diversity_history) == 1

    def test_get_improvement_positive(self):
        ft = FitnessTracker()
        for i in range(10):
            ft.update(generation=i, best_fitness=float(i) * 0.1, avg_fitness=0.0)
        improvement = ft.get_improvement(window=10)
        assert improvement > 0

    def test_get_improvement_insufficient_data(self):
        ft = FitnessTracker()
        ft.update(generation=0, best_fitness=0.5, avg_fitness=0.3)
        assert ft.get_improvement(window=10) == 0.0

    def test_is_converged_when_flat(self):
        ft = FitnessTracker()
        for i in range(15):
            ft.update(generation=i, best_fitness=0.9, avg_fitness=0.8)
        assert ft.is_converged(threshold=1e-4, window=10)

    def test_is_not_converged_when_improving(self):
        ft = FitnessTracker()
        for i in range(15):
            ft.update(generation=i, best_fitness=float(i) * 0.1, avg_fitness=0.0)
        assert not ft.is_converged(threshold=1e-4, window=10)

    def test_get_summary_keys(self):
        ft = FitnessTracker()
        ft.update(generation=0, best_fitness=0.7, avg_fitness=0.5)
        summary = ft.get_summary()
        assert "generations" in summary
        assert "best_fitness" in summary
        assert "final_best" in summary
        assert "converged" in summary

    def test_empty_tracker_summary(self):
        ft = FitnessTracker()
        summary = ft.get_summary()
        assert summary["generations"] == 0
        assert summary["best_fitness"] == 0
