"""Tests for pruning: saliency calculation and Pruner."""

import pytest
import torch
import torch.nn as nn

from genesis.pruning.saliency import compute_weight_importance, SaliencyCalculator
from genesis.pruning.pruner import Pruner, PruningConfig


# ── Shared helpers ──────────────────────────────────────────────────────────────

class SimpleModel(nn.Module):
    """Minimal model whose forward() produces a loss (needed for gradient methods)."""

    def __init__(self, vocab_size: int = 50, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        hidden = self.embed(input_ids)
        logits = self.fc(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1)
            )

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        out.loss = loss
        return out


def _make_batches(batch_size: int = 2, seq_len: int = 8,
                  vocab_size: int = 50, num_batches: int = 2):
    """Return a list of dict batches (acts as an iterable dataloader)."""
    n = batch_size * num_batches
    ids = torch.randint(0, vocab_size, (n, seq_len))
    batches = []
    for i in range(0, n, batch_size):
        chunk = ids[i : i + batch_size]
        batches.append({"input_ids": chunk, "attention_mask": torch.ones_like(chunk),
                        "labels": chunk})
    return batches


# ── compute_weight_importance ───────────────────────────────────────────────────

class TestComputeWeightImportance:

    def test_magnitude_needs_no_dataloader(self):
        model = SimpleModel()
        imp = compute_weight_importance(model, method="magnitude", device="cpu")
        assert len(imp) > 0

    def test_magnitude_scores_are_non_negative(self):
        model = nn.Linear(5, 5, bias=False)
        model.weight.data.fill_(-3.0)
        imp = compute_weight_importance(model, method="magnitude", device="cpu")
        assert torch.allclose(imp["weight"], torch.full((5, 5), 3.0))

    def test_magnitude_shape_matches_parameters(self):
        model = SimpleModel()
        imp = compute_weight_importance(model, method="magnitude", device="cpu")
        params = dict(model.named_parameters())
        for name, scores in imp.items():
            assert scores.shape == params[name].shape

    def test_gradient_requires_dataloader(self):
        with pytest.raises(ValueError, match="DataLoader"):
            compute_weight_importance(SimpleModel(), method="gradient", device="cpu")

    def test_taylor_requires_dataloader(self):
        with pytest.raises(ValueError, match="DataLoader"):
            compute_weight_importance(SimpleModel(), method="taylor", device="cpu")

    def test_fisher_requires_dataloader(self):
        with pytest.raises(ValueError, match="DataLoader"):
            compute_weight_importance(SimpleModel(), method="fisher", device="cpu")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            compute_weight_importance(SimpleModel(), method="unknown", device="cpu")

    def test_gradient_with_dataloader(self):
        model = SimpleModel()
        imp = compute_weight_importance(
            model, method="gradient",
            dataloader=_make_batches(), device="cpu", num_samples=4,
        )
        assert len(imp) > 0
        for scores in imp.values():
            assert (scores >= 0).all()

    def test_taylor_with_dataloader(self):
        model = SimpleModel()
        imp = compute_weight_importance(
            model, method="taylor",
            dataloader=_make_batches(), device="cpu", num_samples=4,
        )
        assert len(imp) > 0
        for scores in imp.values():
            assert (scores >= 0).all()

    def test_fisher_with_dataloader(self):
        model = SimpleModel()
        imp = compute_weight_importance(
            model, method="fisher",
            dataloader=_make_batches(), device="cpu", num_samples=4,
        )
        assert len(imp) > 0
        for scores in imp.values():
            assert (scores >= 0).all()


# ── SaliencyCalculator ──────────────────────────────────────────────────────────

class TestSaliencyCalculator:

    def test_compute_returns_importance_dict(self):
        calc = SaliencyCalculator(SimpleModel(), method="magnitude", device="cpu")
        imp = calc.compute()
        assert isinstance(imp, dict)
        assert len(imp) > 0

    def test_compute_caches_result(self):
        calc = SaliencyCalculator(SimpleModel(), method="magnitude", device="cpu")
        imp1 = calc.compute()
        imp2 = calc.compute()
        assert imp1 is imp2  # same object — cache hit

    def test_force_recompute_returns_new_object(self):
        calc = SaliencyCalculator(SimpleModel(), method="magnitude", device="cpu")
        imp1 = calc.compute()
        imp2 = calc.compute(force_recompute=True)
        assert imp1 is not imp2

    def test_clear_cache(self):
        calc = SaliencyCalculator(SimpleModel(), method="magnitude", device="cpu")
        calc.compute()
        assert calc._importance_cache is not None
        calc.clear_cache()
        assert calc._importance_cache is None

    def test_get_top_k_mask_per_layer_keep_ratio(self):
        model = nn.Linear(20, 20, bias=False)
        calc = SaliencyCalculator(model, method="magnitude", device="cpu")
        masks = calc.get_top_k_mask(k=0.5, per_layer=True)
        assert "weight" in masks
        keep_ratio = masks["weight"].sum().item() / masks["weight"].numel()
        assert 0.4 <= keep_ratio <= 0.6

    def test_get_top_k_mask_global(self):
        model = nn.Linear(10, 10, bias=False)
        calc = SaliencyCalculator(model, method="magnitude", device="cpu")
        masks = calc.get_top_k_mask(k=0.3, per_layer=False)
        assert "weight" in masks
        # At most 30% kept globally
        keep_ratio = masks["weight"].sum().item() / masks["weight"].numel()
        assert keep_ratio <= 0.35  # allow small rounding

    def test_get_pruning_mask_sparsity_complement(self):
        """pruning_mask(sparsity=s) == top_k_mask(k=1-s)."""
        model = nn.Linear(20, 20, bias=False)
        calc = SaliencyCalculator(model, method="magnitude", device="cpu")
        masks = calc.get_pruning_mask(sparsity=0.3, per_layer=True)
        keep_ratio = masks["weight"].sum().item() / masks["weight"].numel()
        # ~70% of weights should survive
        assert 0.65 <= keep_ratio <= 0.75

    def test_masks_are_binary(self):
        model = nn.Linear(10, 10, bias=False)
        calc = SaliencyCalculator(model, method="magnitude", device="cpu")
        masks = calc.get_pruning_mask(sparsity=0.5)
        for mask in masks.values():
            unique_vals = torch.unique(mask)
            assert all(v.item() in {0.0, 1.0} for v in unique_vals)

    def test_importance_ranking_sorted_ascending(self):
        model = nn.Linear(5, 5, bias=False)
        calc = SaliencyCalculator(model, method="magnitude", device="cpu")
        rankings = calc.get_importance_ranking()
        assert len(rankings) == 25  # 5 × 5 weights
        scores = [r[2] for r in rankings]
        assert scores == sorted(scores)  # ascending

    def test_importance_ranking_entry_format(self):
        model = nn.Linear(4, 4, bias=False)
        calc = SaliencyCalculator(model, method="magnitude", device="cpu")
        rankings = calc.get_importance_ranking()
        name, idx, score = rankings[0]
        assert isinstance(name, str)
        assert isinstance(idx, int)
        assert isinstance(score, float)


# ── Pruner ──────────────────────────────────────────────────────────────────────

class TestPruner:

    def test_initial_sparsity_is_zero(self):
        pruner = Pruner(nn.Linear(10, 10), PruningConfig(), device="cpu")
        assert pruner.current_sparsity == 0.0

    def test_unstructured_prune_returns_stats(self):
        model = nn.Linear(20, 20, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.5), device="cpu")
        stats = pruner.prune()
        assert "actual_sparsity" in stats
        assert "total_params" in stats
        assert "pruned_params" in stats

    def test_prune_zeros_weights(self):
        model = nn.Linear(20, 20, bias=False)
        model.weight.data = torch.rand(20, 20) + 1.0  # all positive
        pruner = Pruner(model, PruningConfig(target_sparsity=0.5), device="cpu")
        pruner.prune()
        zeros = (model.weight.data == 0).sum().item()
        assert zeros > 0

    def test_current_sparsity_updated_after_prune(self):
        model = nn.Linear(20, 20, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.3), device="cpu")
        pruner.prune()
        assert pruner.current_sparsity > 0.0

    def test_restore_weights_undoes_pruning(self):
        model = nn.Linear(10, 10, bias=False)
        original = model.weight.data.clone()
        pruner = Pruner(model, PruningConfig(target_sparsity=0.5), device="cpu")
        pruner.prune()
        assert not torch.allclose(model.weight.data, original)
        pruner.restore_weights()
        assert torch.allclose(model.weight.data, original)

    def test_restore_clears_sparsity(self):
        model = nn.Linear(10, 10, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.4), device="cpu")
        pruner.prune()
        pruner.restore_weights()
        assert pruner.current_sparsity == 0.0

    def test_get_sparsity_stats(self):
        model = nn.Linear(10, 10, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.3), device="cpu")
        pruner.prune()
        stats = pruner.get_sparsity_stats()
        assert "global_sparsity" in stats
        assert "total_params" in stats
        assert "zero_params" in stats
        assert "layer_stats" in stats
        assert stats["global_sparsity"] > 0.0

    def test_make_pruning_permanent_clears_originals(self):
        model = nn.Linear(10, 10, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.3), device="cpu")
        pruner.prune()
        assert len(pruner._original_weights) > 0
        pruner.make_pruning_permanent()
        assert len(pruner._original_weights) == 0

    def test_apply_masks_re_zeroes_weights(self):
        model = nn.Linear(10, 10, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.5), device="cpu")
        pruner.prune()
        # Manually fill weights back to 1
        model.weight.data.fill_(1.0)
        pruner.apply_masks()
        # Masked weights should be zeroed again
        zeros = (model.weight.data == 0).sum().item()
        assert zeros > 0

    def test_skip_layers_prevents_pruning(self):
        model = nn.Linear(10, 10, bias=False)
        original = model.weight.data.clone()
        config = PruningConfig(target_sparsity=0.5, skip_layers=["weight"])
        pruner = Pruner(model, config, device="cpu")
        pruner.prune()
        assert torch.allclose(model.weight.data, original)

    def test_structured_prune_row(self):
        model = nn.Linear(20, 20, bias=False)
        config = PruningConfig(
            target_sparsity=0.3,
            structured=True,
            granularity="row",
        )
        pruner = Pruner(model, config, device="cpu")
        stats = pruner.prune()
        assert stats["granularity"] == "row"
        assert "actual_sparsity" in stats

    def test_structured_prune_column(self):
        model = nn.Linear(20, 20, bias=False)
        config = PruningConfig(
            target_sparsity=0.3,
            structured=True,
            granularity="column",
        )
        pruner = Pruner(model, config, device="cpu")
        stats = pruner.prune()
        assert stats["granularity"] == "column"

    def test_pruner_masks_property(self):
        model = nn.Linear(10, 10, bias=False)
        pruner = Pruner(model, PruningConfig(target_sparsity=0.4), device="cpu")
        assert isinstance(pruner.masks, dict)
        pruner.prune()
        assert len(pruner.masks) > 0
