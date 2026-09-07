"""Tests for simulate/pool.py: graceful degradation without popularity data,
CSV loading, and pool-field simulation mechanics."""

from __future__ import annotations

import polars as pl
import pytest

from survivor.simulate.pool import (
    load_popularity,
    popularity_csv_path,
    simulate_pool,
    simulate_pool_many_trials,
)


def test_load_popularity_missing_file_returns_none(tmp_path, monkeypatch):
    from survivor import config as config_module

    monkeypatch.setattr(config_module.CONFIG.paths, "popularity", str(tmp_path))
    assert load_popularity(2099, 1) is None


def test_load_popularity_renormalizes(tmp_path, monkeypatch):
    from survivor import config as config_module

    monkeypatch.setattr(config_module.CONFIG.paths, "popularity", str(tmp_path))
    path = popularity_csv_path(2024, 5)
    pl.DataFrame({"team": ["KC", "SF"], "pick_pct": [40.0, 10.0]}).write_csv(path)
    pop = load_popularity(2024, 5)
    assert pop["KC"] == pytest.approx(0.8)
    assert pop["SF"] == pytest.approx(0.2)


def test_load_popularity_missing_columns_raises(tmp_path, monkeypatch):
    from survivor import config as config_module

    monkeypatch.setattr(config_module.CONFIG.paths, "popularity", str(tmp_path))
    path = popularity_csv_path(2024, 6)
    pl.DataFrame({"team": ["KC"], "pct": [50.0]}).write_csv(path)
    with pytest.raises(ValueError):
        load_popularity(2024, 6)


# ---------------------------------------------------------------------------
# simulate_pool mechanics
# ---------------------------------------------------------------------------


def test_simulate_pool_sole_survivor_when_field_certain_to_lose():
    """You survive for certain, and the whole field is guaranteed to lose
    every week -> you must be the sole survivor."""
    weeks = [1, 2]
    your_probs = {1: 1.0, 2: 1.0}
    popularity = {1: {"BAD": 1.0}, 2: {"BAD": 1.0}}
    survive = {1: {"BAD": 0.0}, 2: {"BAD": 0.0}}
    result = simulate_pool(your_probs, popularity, survive, weeks, n_field_entries=500, seed=0)
    assert result.p_sole_survivor == 1.0
    assert result.p_share_win == 0.0
    assert result.p_you_survive == 1.0


def test_simulate_pool_share_win_when_field_also_certain_to_survive():
    weeks = [1, 2]
    your_probs = {1: 1.0, 2: 1.0}
    popularity = {1: {"GOOD": 1.0}, 2: {"GOOD": 1.0}}
    survive = {1: {"GOOD": 1.0}, 2: {"GOOD": 1.0}}
    result = simulate_pool(your_probs, popularity, survive, weeks, n_field_entries=500, seed=0)
    assert result.p_sole_survivor == 0.0
    assert result.p_share_win == 1.0


def test_simulate_pool_you_lose_means_neither_sole_nor_share():
    weeks = [1]
    your_probs = {1: 0.0}
    popularity = {1: {"X": 1.0}}
    survive = {1: {"X": 0.0}}
    result = simulate_pool(your_probs, popularity, survive, weeks, n_field_entries=100, seed=0)
    assert result.p_you_survive == 0.0
    assert result.p_sole_survivor == 0.0
    assert result.p_share_win == 0.0


def test_simulate_pool_missing_popularity_week_field_stays_alive():
    """A week with no popularity data for the field should not crash and
    should leave the field's alive/dead status untouched that week."""
    weeks = [1, 2]
    your_probs = {1: 1.0, 2: 1.0}
    popularity = {1: {"X": 1.0}}  # no entry for week 2
    survive = {1: {"X": 1.0}, 2: {"X": 1.0}}
    result = simulate_pool(your_probs, popularity, survive, weeks, n_field_entries=50, seed=0)
    assert result.p_you_survive == 1.0


def test_simulate_pool_many_trials_averages_are_valid_probabilities():
    weeks = [1, 2, 3]
    your_probs = {1: 0.7, 2: 0.6, 3: 0.5}
    popularity = {w: {"A": 0.6, "B": 0.4} for w in weeks}
    survive = {w: {"A": 0.55, "B": 0.45} for w in weeks}
    result = simulate_pool_many_trials(your_probs, popularity, survive, weeks, n_trials=50, n_field_entries=200, seed=0)
    assert 0.0 <= result["p_sole_survivor"] <= 1.0
    assert 0.0 <= result["p_share_win"] <= 1.0
    assert result["p_sole_survivor"] + result["p_share_win"] <= result["p_you_survive"] + 1e-9
