"""Tests for genesis/tts/mcd_fitness.py, style_evolution.py, and tts_child.py."""

import pytest
import torch
import numpy as np

from genesis.tts.mcd_fitness import (
    compute_mcd,
    compute_f0_rmse,
    compute_vuv_error,
    MCDFitness,
)


# ── compute_mcd ───────────────────────────────────────────────────────────────

class TestComputeMCD:

    def _mel(self, batch=1, mel_dim=80, time=50):
        """Random positive mel spectrogram (values > 0 for log)."""
        return torch.rand(batch, mel_dim, time) + 0.1

    def test_identical_mels_zero_mcd(self):
        mel = self._mel()
        mcd = compute_mcd(mel, mel, reduction="mean")
        assert mcd.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_mels_positive_mcd(self):
        ref = self._mel()
        syn = torch.rand_like(ref) * 10 + 0.1
        mcd = compute_mcd(ref, syn, reduction="mean")
        assert mcd.item() > 0.0

    def test_reduction_mean_is_scalar(self):
        mel = self._mel()
        mcd = compute_mcd(mel, mel, reduction="mean")
        assert mcd.ndim == 0

    def test_reduction_sum_is_scalar(self):
        mel = self._mel()
        mcd = compute_mcd(mel, mel, reduction="sum")
        assert mcd.ndim == 0

    def test_reduction_none_shape(self):
        mel = self._mel(batch=2, mel_dim=80, time=10)
        mcd = compute_mcd(mel, mel, reduction="none")
        # Should be (batch, time)
        assert mcd.ndim == 2

    def test_handles_different_lengths(self):
        ref = torch.rand(1, 80, 50) + 0.1
        syn = torch.rand(1, 80, 30) + 0.1
        # Should truncate to min length and not raise
        mcd = compute_mcd(ref, syn, reduction="mean")
        assert torch.isfinite(mcd)

    def test_batch_dimension(self):
        mel = self._mel(batch=4)
        mcd = compute_mcd(mel, mel, reduction="mean")
        assert torch.isfinite(mcd)

    def test_higher_distortion_higher_mcd(self):
        ref = self._mel()
        close = (ref + torch.rand_like(ref) * 0.001).clamp(min=0.01)
        far = (ref * 50 + 5.0).clamp(min=0.01)  # clearly different, always positive
        mcd_close = compute_mcd(ref, close, reduction="mean").item()
        mcd_far = compute_mcd(ref, far, reduction="mean").item()
        assert mcd_far > mcd_close


# ── compute_f0_rmse ───────────────────────────────────────────────────────────

class TestComputeF0RMSE:

    def test_identical_f0_zero_rmse(self):
        f0 = torch.tensor([110.0, 120.0, 130.0, 0.0])  # last frame unvoiced
        rmse = compute_f0_rmse(f0, f0)
        assert rmse.item() == pytest.approx(0.0, abs=1e-4)

    def test_all_unvoiced_returns_inf(self):
        f0 = torch.zeros(10)
        rmse = compute_f0_rmse(f0, f0)
        assert rmse.item() == float("inf")

    def test_different_f0_positive_rmse(self):
        ref = torch.tensor([100.0, 110.0, 120.0, 130.0])
        syn = torch.tensor([200.0, 220.0, 240.0, 260.0])
        rmse = compute_f0_rmse(ref, syn)
        assert rmse.item() > 0

    def test_handles_length_mismatch(self):
        ref = torch.ones(10) * 100.0
        syn = torch.ones(7) * 100.0
        rmse = compute_f0_rmse(ref, syn)
        # Should not raise; identical F0 in voiced region → 0
        assert rmse.item() == pytest.approx(0.0, abs=1e-4)

    def test_rmse_in_cents(self):
        # One octave difference: ratio 2 → 1200 cents
        ref = torch.tensor([100.0])
        syn = torch.tensor([200.0])
        rmse = compute_f0_rmse(ref, syn)
        assert rmse.item() == pytest.approx(1200.0, abs=1.0)


# ── compute_vuv_error ─────────────────────────────────────────────────────────

class TestComputeVUVError:

    def test_identical_vuv_zero_error(self):
        f0 = torch.tensor([100.0, 0.0, 120.0, 0.0, 130.0])
        err = compute_vuv_error(f0, f0)
        assert err == pytest.approx(0.0)

    def test_all_voiced_vs_all_unvoiced(self):
        ref = torch.ones(10) * 100.0     # all voiced
        syn = torch.zeros(10)            # all unvoiced
        err = compute_vuv_error(ref, syn)
        assert err == pytest.approx(1.0)

    def test_half_mismatch(self):
        ref = torch.tensor([100.0, 100.0, 0.0, 0.0])
        syn = torch.tensor([0.0, 0.0, 100.0, 100.0])
        err = compute_vuv_error(ref, syn)
        assert err == pytest.approx(1.0)

    def test_handles_length_mismatch(self):
        ref = torch.ones(10) * 100.0
        syn = torch.ones(8) * 100.0
        err = compute_vuv_error(ref, syn)
        assert 0.0 <= err <= 1.0


# ── MCDFitness ────────────────────────────────────────────────────────────────

class TestMCDFitness:

    def _mel(self, batch=1, mel_dim=80, time=30):
        return torch.rand(batch, mel_dim, time) + 0.1

    def test_raises_without_reference(self):
        evaluator = MCDFitness()
        with pytest.raises(ValueError, match="No reference"):
            evaluator.evaluate(self._mel())

    def test_evaluate_identical_high_fitness(self):
        mel = self._mel()
        evaluator = MCDFitness(reference_mels=[mel])
        result = evaluator.evaluate(mel)
        # MCD=0 → similarity_fitness = 1/(1+0) = 1.0
        assert result["similarity_fitness"] == pytest.approx(1.0, abs=1e-5)

    def test_evaluate_returns_expected_keys(self):
        mel = self._mel()
        evaluator = MCDFitness(reference_mels=[mel])
        result = evaluator.evaluate(mel)
        for key in ["fitness", "mcd", "min_mcd", "similarity_fitness", "naturalness_fitness"]:
            assert key in result

    def test_fitness_in_range(self):
        ref = self._mel()
        syn = self._mel()
        evaluator = MCDFitness(reference_mels=[ref])
        result = evaluator.evaluate(syn)
        assert 0.0 <= result["fitness"] <= 1.0

    def test_add_reference(self):
        evaluator = MCDFitness()
        mel = self._mel()
        evaluator.add_reference(mel)
        assert len(evaluator.reference_mels) == 1

    def test_evaluate_with_specific_reference_idx(self):
        mel1 = self._mel()
        mel2 = self._mel() * 5  # clearly different
        evaluator = MCDFitness(reference_mels=[mel1, mel2])
        r0 = evaluator.evaluate(mel1, reference_idx=0)
        assert r0["mcd"] == pytest.approx(0.0, abs=1e-5)

    def test_batch_evaluate_length(self):
        ref = self._mel()
        evaluator = MCDFitness(reference_mels=[ref])
        mels = [self._mel() for _ in range(3)]
        results = evaluator.batch_evaluate(mels)
        assert len(results) == 3

    def test_batch_evaluate_all_have_keys(self):
        ref = self._mel()
        evaluator = MCDFitness(reference_mels=[ref])
        mels = [self._mel() for _ in range(2)]
        results = evaluator.batch_evaluate(mels)
        for r in results:
            assert "fitness" in r

    def test_rank_population_sorted_descending(self):
        ref = self._mel()
        evaluator = MCDFitness(reference_mels=[ref])
        # Include the reference itself to ensure at least one high-fitness entry
        mels = [ref, self._mel() * 10]
        ranked = evaluator.rank_population(mels)
        assert len(ranked) == 2
        # First rank should have higher fitness than second
        assert ranked[0][1] >= ranked[1][1]

    def test_naturalness_score_in_range(self):
        mel = self._mel()
        evaluator = MCDFitness(reference_mels=[mel])
        result = evaluator.evaluate(mel)
        assert 0.0 <= result["naturalness_fitness"] <= 1.0

    def test_lower_mcd_higher_similarity_fitness(self):
        ref = self._mel()
        close = ref + torch.randn_like(ref) * 0.01
        far = self._mel() * 10
        evaluator = MCDFitness(reference_mels=[ref])
        r_close = evaluator.evaluate(close)
        r_far = evaluator.evaluate(far)
        assert r_close["similarity_fitness"] > r_far["similarity_fitness"]

    def test_multiple_references_averages(self):
        ref1 = self._mel()
        ref2 = self._mel()
        syn = self._mel()
        evaluator = MCDFitness(reference_mels=[ref1, ref2])
        result = evaluator.evaluate(syn)
        # Should be the average MCD over both references
        assert result["mcd"] >= 0


# ── StyleToken ────────────────────────────────────────────────────────────────

class TestStyleToken:

    def test_clone_is_independent(self):
        from genesis.tts.style_evolution import StyleToken
        emb = torch.randn(10, 64)
        tok = StyleToken(embedding=emb, name="test", fitness=0.5)
        clone = tok.clone()
        clone.embedding[0, 0] = 999.0
        # Original should be untouched
        assert tok.embedding[0, 0] != 999.0

    def test_clone_copies_metadata(self):
        from genesis.tts.style_evolution import StyleToken
        tok = StyleToken(embedding=torch.randn(4, 8), metadata={"key": "value"})
        clone = tok.clone()
        assert clone.metadata["key"] == "value"

    def test_clone_has_same_fitness(self):
        from genesis.tts.style_evolution import StyleToken
        tok = StyleToken(embedding=torch.randn(4, 8), fitness=0.75)
        clone = tok.clone()
        assert clone.fitness == 0.75


# ── StyleEvolution ────────────────────────────────────────────────────────────

class TestStyleEvolution:

    def _make_evo(self, pop_size=5, style_dim=32, num_tokens=4):
        from genesis.tts.style_evolution import StyleEvolution
        return StyleEvolution(
            style_dim=style_dim,
            num_tokens=num_tokens,
            population_size=pop_size,
            elite_size=1,
            mutation_rate=0.5,
            mutation_scale=0.05,
            crossover_rate=0.5,
        )

    def test_initialize_random_population(self):
        evo = self._make_evo()
        evo.initialize_population()
        assert len(evo.get_population()) == 5

    def test_initialize_from_base_tokens(self):
        evo = self._make_evo()
        base = torch.randn(4, 32)
        evo.initialize_population(base_tokens=base)
        assert len(evo.get_population()) == 5
        # First individual should match base
        assert torch.allclose(evo.get_individual(0), base)

    def test_set_fitness(self):
        evo = self._make_evo()
        evo.initialize_population()
        evo.set_fitness(0, 0.9)
        assert evo._fitnesses[0] == pytest.approx(0.9)

    def test_set_all_fitnesses(self):
        evo = self._make_evo(pop_size=4)
        evo.initialize_population()
        fits = [0.1, 0.2, 0.3, 0.4]
        evo.set_all_fitnesses(fits)
        assert evo._fitnesses == fits

    def test_best_fitness_property(self):
        evo = self._make_evo()
        evo.initialize_population()
        evo.set_all_fitnesses([0.1, 0.5, 0.3, 0.2, 0.4])
        assert evo.best_fitness == pytest.approx(0.5)

    def test_average_fitness_property(self):
        evo = self._make_evo(pop_size=4)
        evo.initialize_population()
        evo.set_all_fitnesses([0.0, 1.0, 0.0, 1.0])
        assert evo.average_fitness == pytest.approx(0.5)

    def test_generation_starts_at_zero(self):
        evo = self._make_evo()
        assert evo.generation == 0

    def test_evolve_increments_generation(self):
        evo = self._make_evo()
        evo.initialize_population()
        evo.set_all_fitnesses([float(i) for i in range(5)])
        evo.evolve()
        assert evo.generation == 1

    def test_evolve_preserves_population_size(self):
        evo = self._make_evo(pop_size=6)
        evo.initialize_population()
        evo.set_all_fitnesses([float(i) for i in range(6)])
        evo.evolve()
        assert len(evo.get_population()) == 6

    def test_get_best_returns_highest_fitness(self):
        evo = self._make_evo(pop_size=4)
        evo.initialize_population()
        evo.set_all_fitnesses([0.1, 0.9, 0.5, 0.3])
        _, best_fit = evo.get_best()
        assert best_fit == pytest.approx(0.9)

    def test_save_and_load_state(self, tmp_path):
        evo = self._make_evo(pop_size=3)
        evo.initialize_population()
        evo.set_all_fitnesses([0.1, 0.5, 0.3])
        evo._generation = 7

        path = str(tmp_path / "style_evo.pt")
        evo.save_state(path)

        evo2 = self._make_evo(pop_size=3)
        evo2.load_state(path)

        assert evo2.generation == 7
        assert evo2.population_size == 3
        assert len(evo2.get_population()) == 3

    def test_individual_shape_correct(self):
        evo = self._make_evo(style_dim=64, num_tokens=8)
        evo.initialize_population()
        ind = evo.get_individual(0)
        assert ind.shape == (8, 64)


# ── MultiStyleEvolution ───────────────────────────────────────────────────────

class TestMultiStyleEvolution:

    def test_initialize_all_creates_evolutions(self):
        from genesis.tts.style_evolution import MultiStyleEvolution
        configs = {
            "prosody": {"dim": 32, "num_tokens": 4},
            "emotion": {"dim": 64, "num_tokens": 8},
        }
        mse = MultiStyleEvolution(style_configs=configs, population_size=4)
        assert "prosody" in mse.evolutions
        assert "emotion" in mse.evolutions

    def test_initialize_all_no_base(self):
        from genesis.tts.style_evolution import MultiStyleEvolution
        configs = {"style": {"dim": 16, "num_tokens": 2}}
        mse = MultiStyleEvolution(style_configs=configs, population_size=3)
        mse.initialize_all()
        assert len(mse.evolutions["style"].get_population()) == 3

    def test_evolve_all(self):
        from genesis.tts.style_evolution import MultiStyleEvolution
        configs = {"style": {"dim": 16, "num_tokens": 2}}
        mse = MultiStyleEvolution(style_configs=configs, population_size=3)
        mse.initialize_all()
        for evo in mse.evolutions.values():
            evo.set_all_fitnesses([float(i) for i in range(3)])
        mse.evolve_all()
        assert mse.evolutions["style"].generation == 1

    def test_get_combined_style(self):
        from genesis.tts.style_evolution import MultiStyleEvolution
        configs = {"a": {"dim": 8, "num_tokens": 2}, "b": {"dim": 16, "num_tokens": 3}}
        mse = MultiStyleEvolution(style_configs=configs, population_size=4)
        mse.initialize_all()
        combined = mse.get_combined_style({"a": 0, "b": 1})
        assert "a" in combined and "b" in combined
        assert combined["a"].shape == (2, 8)
        assert combined["b"].shape == (3, 16)


# ── TTSChild ──────────────────────────────────────────────────────────────────

class TestTTSChild:

    def test_default_initialization(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        assert child.fitness == 0.0
        assert child.generation == 0
        assert child.parent_ids == []
        assert child.id is not None

    def test_style_tokens_initialized(self):
        from genesis.tts.tts_child import TTSChild, TTSConfig
        cfg = TTSConfig(style_dim=64)
        child = TTSChild(config=cfg)
        assert child.style_tokens.shape == (10, 64)

    def test_speaker_embeddings_initialized(self):
        from genesis.tts.tts_child import TTSChild, TTSConfig
        cfg = TTSConfig(speaker_dim=128, num_speakers=2)
        child = TTSChild(config=cfg)
        assert child.speaker_embeddings.shape == (2, 128)

    def test_custom_style_tokens(self):
        from genesis.tts.tts_child import TTSChild
        tokens = torch.randn(5, 32)
        child = TTSChild(style_tokens=tokens)
        assert torch.allclose(child.style_tokens, tokens)

    def test_clone_is_independent(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        child.fitness = 0.7
        clone = child.clone()
        clone.style_tokens[0, 0] = 999.0
        assert child.style_tokens[0, 0] != 999.0

    def test_clone_copies_fitness(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        child.fitness = 0.8
        clone = child.clone()
        assert clone.fitness == 0.8

    def test_clone_parent_id_set(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        clone = child.clone()
        assert child.id in clone.parent_ids

    def test_mutate_returns_different_child(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        # With mutation_rate=1.0, mutation will always happen
        mutated = child.mutate(mutation_rate=1.0, mutation_scale=0.1)
        assert mutated is not child

    def test_mutate_increments_generation(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        child.generation = 3
        mutated = child.mutate(mutation_rate=1.0)
        assert mutated.generation == 4

    def test_crossover_child_generation(self):
        from genesis.tts.tts_child import TTSChild
        p1 = TTSChild()
        p1.generation = 2
        p2 = TTSChild()
        p2.generation = 4
        child = p1.crossover(p2)
        assert child.generation == 5  # max(2,4) + 1

    def test_crossover_child_has_both_parents(self):
        from genesis.tts.tts_child import TTSChild
        p1 = TTSChild()
        p2 = TTSChild()
        child = p1.crossover(p2)
        assert p1.id in child.parent_ids
        assert p2.id in child.parent_ids

    def test_crossover_interpolates_style_tokens(self):
        from genesis.tts.tts_child import TTSChild, TTSConfig
        cfg = TTSConfig(style_dim=8)
        p1 = TTSChild(config=cfg, style_tokens=torch.zeros(10, 8))
        p2 = TTSChild(config=cfg, style_tokens=torch.ones(10, 8))
        child = p1.crossover(p2)
        # Child should be between 0 and 1
        assert (child.style_tokens >= 0).all()
        assert (child.style_tokens <= 1).all()

    def test_get_and_set_evolved_params(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        params = child.get_evolved_params()
        assert "style_tokens" in params
        assert "speaker_embeddings" in params
        # Set new params
        new_tokens = torch.zeros_like(params["style_tokens"])
        child.set_evolved_params({"style_tokens": new_tokens})
        assert torch.allclose(child.style_tokens, new_tokens)

    def test_save_and_load_roundtrip(self, tmp_path):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        child.fitness = 0.65
        child.generation = 7
        path = str(tmp_path / "child.pt")
        child.save(path)

        loaded = TTSChild.load(path)
        assert loaded.id == child.id
        assert loaded.fitness == pytest.approx(0.65)
        assert loaded.generation == 7
        assert torch.allclose(loaded.style_tokens, child.style_tokens)

    def test_synthesize_raises_without_model(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            child.synthesize("hello", device="cpu")

    def test_repr(self):
        from genesis.tts.tts_child import TTSChild
        child = TTSChild()
        r = repr(child)
        assert "TTSChild" in r
        assert "fitness" in r
