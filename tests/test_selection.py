"""Tests for all selection strategies."""

import pytest
import numpy as np

from genesis.core.population import Individual
from genesis.core.selection import (
    ElitismSelection,
    TournamentSelection,
    RouletteWheelSelection,
    RankSelection,
    TruncationSelection,
    BoltzmannSelection,
    SteadyStateSelection,
    create_selection_strategy,
)


# ── Helper ───────────────────────────────────────────────────────────────────────

def _pop(fitnesses: list[float]) -> list[Individual]:
    """Create a population list with the given fitness values."""
    return [Individual(state_dict={}, fitness=f) for f in fitnesses]


# ── ElitismSelection ─────────────────────────────────────────────────────────────

class TestElitismSelection:

    def test_selects_correct_count(self):
        sel = ElitismSelection()
        pop = _pop([0.1, 0.9, 0.5, 0.7, 0.3])
        result = sel.select(pop, 3)
        assert len(result) == 3

    def test_selects_top_by_fitness(self):
        sel = ElitismSelection()
        pop = _pop([0.1, 0.9, 0.5, 0.7, 0.3])
        result = sel.select(pop, 2)
        fitnesses = sorted([ind.fitness for ind in result], reverse=True)
        assert fitnesses[0] == pytest.approx(0.9)
        assert fitnesses[1] == pytest.approx(0.7)

    def test_select_one(self):
        sel = ElitismSelection()
        pop = _pop([0.2, 0.8, 0.5])
        result = sel.select(pop, 1)
        assert result[0].fitness == pytest.approx(0.8)

    def test_callable_interface(self):
        sel = ElitismSelection()
        pop = _pop([0.3, 0.7])
        result = sel(pop, 1)
        assert len(result) == 1


# ── TournamentSelection ──────────────────────────────────────────────────────────

class TestTournamentSelection:

    def test_selects_correct_count(self):
        sel = TournamentSelection(tournament_size=3)
        pop = _pop([0.1, 0.5, 0.9, 0.3, 0.7])
        result = sel.select(pop, 4)
        assert len(result) == 4

    def test_winner_is_highest_fitness_in_tournament(self):
        """With a population of one distinct best, it should always be picked."""
        sel = TournamentSelection(tournament_size=5)
        pop = _pop([0.0, 0.0, 0.0, 0.0, 1.0])
        for _ in range(20):
            result = sel.select(pop, 1)
            assert result[0].fitness == pytest.approx(1.0)

    def test_tournament_size_clamped_to_population(self):
        """tournament_size > pop size must not raise."""
        sel = TournamentSelection(tournament_size=10)
        pop = _pop([0.5, 0.8])
        result = sel.select(pop, 2)
        assert len(result) == 2

    def test_single_individual_population(self):
        sel = TournamentSelection(tournament_size=3)
        pop = _pop([0.6])
        result = sel.select(pop, 1)
        assert result[0].fitness == pytest.approx(0.6)


# ── RouletteWheelSelection ───────────────────────────────────────────────────────

class TestRouletteWheelSelection:

    def test_selects_correct_count(self):
        sel = RouletteWheelSelection()
        pop = _pop([0.2, 0.4, 0.6, 0.8])
        result = sel.select(pop, 3)
        assert len(result) == 3

    def test_handles_negative_fitness(self):
        sel = RouletteWheelSelection()
        pop = _pop([-1.0, -0.5, 0.0, 0.5])
        result = sel.select(pop, 2)
        assert len(result) == 2

    def test_rank_scaling(self):
        sel = RouletteWheelSelection(scaling="rank")
        pop = _pop([0.1, 0.5, 0.9])
        result = sel.select(pop, 2)
        assert len(result) == 2

    def test_sigma_scaling(self):
        sel = RouletteWheelSelection(scaling="sigma")
        pop = _pop([0.1, 0.5, 0.9])
        result = sel.select(pop, 2)
        assert len(result) == 2

    def test_all_results_are_from_population(self):
        sel = RouletteWheelSelection()
        pop = _pop([0.1, 0.3, 0.6])
        pop_ids = {id(ind) for ind in pop}
        for ind in sel.select(pop, 10):
            assert id(ind) in pop_ids


# ── RankSelection ────────────────────────────────────────────────────────────────

class TestRankSelection:

    def test_selects_correct_count(self):
        sel = RankSelection(selection_pressure=1.5)
        pop = _pop([0.2, 0.4, 0.6, 0.8, 1.0])
        result = sel.select(pop, 3)
        assert len(result) == 3

    def test_probabilities_sum_to_one(self):
        """Indirectly: all selected must come from the population."""
        sel = RankSelection()
        pop = _pop([0.1, 0.5, 0.9])
        pop_ids = {id(ind) for ind in pop}
        for ind in sel.select(pop, 20):
            assert id(ind) in pop_ids

    def test_higher_fitness_selected_more_often(self):
        np.random.seed(0)
        sel = RankSelection(selection_pressure=2.0)
        fitnesses = [0.0, 0.0, 0.0, 0.0, 1.0]  # only last one is best
        pop = _pop(fitnesses)
        best = max(pop, key=lambda x: x.fitness)
        selected = sel.select(pop, 100)
        best_count = sum(1 for ind in selected if ind is best)
        # Best individual should be selected significantly more often
        assert best_count > 20


# ── TruncationSelection ──────────────────────────────────────────────────────────

class TestTruncationSelection:

    def test_selects_correct_count(self):
        sel = TruncationSelection(truncation_rate=0.5)
        pop = _pop([0.1, 0.3, 0.5, 0.7, 0.9])
        result = sel.select(pop, 4)
        assert len(result) == 4

    def test_only_top_half_selected(self):
        sel = TruncationSelection(truncation_rate=0.5)
        pop = _pop([0.1, 0.2, 0.8, 0.9])
        result = sel.select(pop, 20)
        for ind in result:
            assert ind.fitness >= 0.8

    def test_truncation_rate_one_uses_whole_population(self):
        sel = TruncationSelection(truncation_rate=1.0)
        pop = _pop([0.1, 0.5, 0.9])
        pop_ids = {id(ind) for ind in pop}
        for ind in sel.select(pop, 10):
            assert id(ind) in pop_ids


# ── BoltzmannSelection ────────────────────────────────────────────────────────────

class TestBoltzmannSelection:

    def test_selects_correct_count(self):
        sel = BoltzmannSelection(initial_temperature=5.0)
        pop = _pop([0.1, 0.4, 0.7, 1.0])
        result = sel.select(pop, 3)
        assert len(result) == 3

    def test_cool_down_reduces_temperature(self):
        sel = BoltzmannSelection(initial_temperature=10.0, cooling_rate=0.5,
                                 min_temperature=0.1)
        t1 = sel.temperature
        sel.cool_down()
        assert sel.temperature < t1

    def test_cool_down_respects_minimum(self):
        sel = BoltzmannSelection(initial_temperature=0.1, min_temperature=0.1,
                                 cooling_rate=0.5)
        sel.cool_down()
        assert sel.temperature >= 0.1

    def test_all_selected_from_population(self):
        sel = BoltzmannSelection(initial_temperature=1.0)
        pop = _pop([0.2, 0.5, 0.8])
        pop_ids = {id(ind) for ind in pop}
        for ind in sel.select(pop, 10):
            assert id(ind) in pop_ids


# ── SteadyStateSelection ──────────────────────────────────────────────────────────

class TestSteadyStateSelection:

    def test_selects_correct_count(self):
        sel = SteadyStateSelection(replacement_rate=0.4)
        pop = _pop([0.1, 0.3, 0.5, 0.7, 0.9])
        result = sel.select(pop, 5)
        assert len(result) == 5

    def test_survivors_are_high_fitness(self):
        sel = SteadyStateSelection(replacement_rate=0.2)
        fitnesses = [0.1, 0.2, 0.3, 0.4, 1.0]
        pop = _pop(fitnesses)
        result = sel.select(pop, 5)
        # The best individual must be in survivors (top 80%)
        best = max(pop, key=lambda x: x.fitness)
        assert any(ind is best for ind in result)

    def test_uses_custom_parent_selection(self):
        parent_sel = ElitismSelection()
        sel = SteadyStateSelection(replacement_rate=0.5,
                                   parent_selection=parent_sel)
        pop = _pop([0.1, 0.5, 0.9, 0.3, 0.7])
        result = sel.select(pop, 5)
        assert len(result) == 5


# ── Factory function ─────────────────────────────────────────────────────────────

class TestCreateSelectionStrategy:

    @pytest.mark.parametrize("name,cls", [
        ("elitism", ElitismSelection),
        ("tournament", TournamentSelection),
        ("roulette", RouletteWheelSelection),
        ("rank", RankSelection),
        ("truncation", TruncationSelection),
        ("boltzmann", BoltzmannSelection),
        ("steady_state", SteadyStateSelection),
    ])
    def test_factory_returns_correct_type(self, name, cls):
        strategy = create_selection_strategy(name)
        assert isinstance(strategy, cls)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError):
            create_selection_strategy("unknown_strategy")

    def test_factory_passes_kwargs(self):
        sel = create_selection_strategy("tournament", tournament_size=7)
        assert sel.tournament_size == 7
