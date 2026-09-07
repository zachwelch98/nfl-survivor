"""Tests for models/montecarlo.py: log5 combination, OT tie resolution,
simulator sanity (plan's "correctness tests that matter" list)."""

from __future__ import annotations

import numpy as np
import pytest

from survivor.features.drive_rates import OUTCOME_CATEGORIES
from survivor.models.montecarlo import (
    DriveSimConfig,
    _resolve_win_probabilities,
    apply_logit_shifts,
    combine_matchup_rates,
    combined_n_drives,
    multiplier_to_category_shifts,
    simulate_game,
)

_RAW_LEAGUE = {
    "td": 0.236, "fg": 0.175, "punt": 0.381, "turnover_int": 0.063,
    "turnover_fumble": 0.055, "turnover_on_downs": 0.058, "missed_fg": 0.029, "safety": 0.004,
}
_league_total = sum(_RAW_LEAGUE.values())  # these real-data-derived figures were rounded to 3dp and don't sum to exactly 1
LEAGUE = {k: v / _league_total for k, v in _RAW_LEAGUE.items()}


# ---------------------------------------------------------------------------
# combine_matchup_rates
# ---------------------------------------------------------------------------


def test_combine_matchup_rates_neutral_matchup_returns_league():
    combined = combine_matchup_rates(LEAGUE, LEAGUE, LEAGUE)
    for k in OUTCOME_CATEGORIES:
        assert combined[k] == pytest.approx(LEAGUE[k], abs=1e-6)
    assert sum(combined.values()) == pytest.approx(1.0, abs=1e-9)


def test_combine_matchup_rates_strong_offense_and_weak_defense_compound():
    """A good offense against a bad defense should produce a higher TD rate
    than either team's raw rate alone (the log5 compounding effect)."""
    strong_off = dict(LEAGUE, td=0.35, punt=0.27)
    weak_def = dict(LEAGUE, td=0.32, punt=0.30)
    combined = combine_matchup_rates(strong_off, weak_def, LEAGUE)
    assert combined["td"] > strong_off["td"]
    assert combined["td"] > weak_def["td"]
    assert sum(combined.values()) == pytest.approx(1.0, abs=1e-9)


def test_combine_matchup_rates_weak_offense_and_strong_defense_suppress():
    weak_off = dict(LEAGUE, td=0.15, punt=0.48)
    strong_def = dict(LEAGUE, td=0.17, punt=0.46)
    combined = combine_matchup_rates(weak_off, strong_def, LEAGUE)
    assert combined["td"] < weak_off["td"]
    assert combined["td"] < strong_def["td"]


def test_combine_matchup_rates_always_sums_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = rng.dirichlet(np.ones(len(OUTCOME_CATEGORIES)))
        off = dict(zip(OUTCOME_CATEGORIES, raw))
        raw2 = rng.dirichlet(np.ones(len(OUTCOME_CATEGORIES)))
        deff = dict(zip(OUTCOME_CATEGORIES, raw2))
        combined = combine_matchup_rates(off, deff, LEAGUE)
        assert sum(combined.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= v <= 1.0 for v in combined.values())


# ---------------------------------------------------------------------------
# apply_logit_shifts / multiplier_to_category_shifts
# ---------------------------------------------------------------------------


def test_apply_logit_shifts_zero_shift_is_noop():
    shifted = apply_logit_shifts(LEAGUE, dict.fromkeys(OUTCOME_CATEGORIES, 0.0))
    for k in OUTCOME_CATEGORIES:
        assert shifted[k] == pytest.approx(LEAGUE[k], abs=1e-6)


def test_multiplier_to_category_shifts_below_one_hurts_scoring():
    shifts = multiplier_to_category_shifts(0.9)
    assert shifts["td"] < 0
    assert shifts["fg"] < 0
    assert shifts["turnover_int"] > 0
    adjusted = apply_logit_shifts(LEAGUE, shifts)
    assert adjusted["td"] < LEAGUE["td"]
    assert adjusted["turnover_int"] > LEAGUE["turnover_int"]


def test_multiplier_to_category_shifts_neutral_at_one():
    shifts = multiplier_to_category_shifts(1.0)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in shifts.values())


# ---------------------------------------------------------------------------
# OT tie resolution
# ---------------------------------------------------------------------------


def test_resolve_win_probabilities_reduces_tie_mass():
    home_scores = np.array([21, 21, 21, 24, 17] * 200)
    away_scores = np.array([21, 21, 21, 17, 24] * 200)
    raw_tie_frac = float(np.mean(home_scores == away_scores))  # 0.6
    home_p, away_p, tie_p = _resolve_win_probabilities(home_scores, away_scores, ot_still_tied_rate=0.05)
    assert tie_p == pytest.approx(raw_tie_frac * 0.05, abs=1e-9)
    assert home_p + away_p + tie_p == pytest.approx(1.0, abs=1e-9)
    assert tie_p < raw_tie_frac


def test_resolve_win_probabilities_splits_tied_mass_by_decisive_share():
    """If home wins decisive games far more often than away, the resolved
    tied mass should favor home too, not split it 50/50."""
    home_scores = np.array([21] * 10 + [30] * 90 + [10] * 10)
    away_scores = np.array([21] * 10 + [20] * 90 + [30] * 10)
    home_p, away_p, tie_p = _resolve_win_probabilities(home_scores, away_scores, ot_still_tied_rate=0.05)
    assert home_p > away_p  # home was decisive winner 90/100 of the time


def test_resolve_win_probabilities_degenerate_all_tied():
    home_scores = np.array([14, 14, 14])
    away_scores = np.array([14, 14, 14])
    home_p, away_p, tie_p = _resolve_win_probabilities(home_scores, away_scores, ot_still_tied_rate=0.05)
    assert home_p == pytest.approx(0.475)
    assert away_p == pytest.approx(0.475)
    assert tie_p == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# simulate_game: sanity checks (plan's "simulator sanity" correctness test)
# ---------------------------------------------------------------------------


def test_simulate_game_symmetric_matchup_near_fifty_fifty():
    config = DriveSimConfig(n_simulations=20000, seed=1)
    result = simulate_game(LEAGUE, LEAGUE, n_drives=10, config=config)
    assert result.home_win_prob == pytest.approx(0.5, abs=0.03)
    assert result.away_win_prob == pytest.approx(0.5, abs=0.03)
    assert result.simulated_spread == pytest.approx(0.0, abs=1.0)


def test_simulate_game_tie_rate_matches_empirical_nfl_rate():
    """The whole reason for _resolve_win_probabilities: final tie_prob
    should land near real NFL's ~0.2-0.3% rate, not the much higher
    regulation-only tie rate a discrete scoring process produces."""
    config = DriveSimConfig(n_simulations=30000, seed=2)
    result = simulate_game(LEAGUE, LEAGUE, n_drives=10, config=config)
    assert 0.0005 < result.tie_prob < 0.01


def test_simulate_game_mean_total_matches_real_nfl_scoring_range():
    """A simulator that produces wildly unrealistic scores (e.g. 5-point or
    90-point games on average) is broken regardless of its win probabilities."""
    config = DriveSimConfig(n_simulations=20000, seed=3)
    result = simulate_game(LEAGUE, LEAGUE, n_drives=10, config=config)
    assert 35 < result.simulated_total < 55
    assert 15 < result.mean_home_score < 30
    assert 15 < result.mean_away_score < 30


def test_simulate_game_stronger_offense_favors_that_team():
    strong_off = dict(LEAGUE, td=0.40, punt=0.22)
    config = DriveSimConfig(n_simulations=20000, seed=4)
    result = simulate_game(strong_off, LEAGUE, n_drives=10, config=config)
    assert result.home_win_prob > 0.65
    assert result.simulated_spread > 3


def test_simulate_game_deterministic_with_same_seed():
    config = DriveSimConfig(n_simulations=5000, seed=99)
    r1 = simulate_game(LEAGUE, LEAGUE, n_drives=10, config=config)
    r2 = simulate_game(LEAGUE, LEAGUE, n_drives=10, config=config)
    assert r1.home_win_prob == r2.home_win_prob
    assert r1.simulated_total == r2.simulated_total


def test_simulate_game_rejects_zero_drives():
    with pytest.raises(ValueError):
        simulate_game(LEAGUE, LEAGUE, n_drives=0)


def test_simulate_game_probabilities_sum_to_one():
    config = DriveSimConfig(n_simulations=5000, seed=7)
    result = simulate_game(LEAGUE, LEAGUE, n_drives=10, config=config)
    assert result.home_win_prob + result.away_win_prob + result.tie_prob == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# combined_n_drives
# ---------------------------------------------------------------------------


def test_combined_n_drives_league_average_teams_get_league_average():
    assert combined_n_drives(9.7, 9.7, 9.7) == 10  # rounds to nearest int


def test_combined_n_drives_fast_pace_teams_get_more_drives():
    fast_vs_fast = combined_n_drives(11.0, 11.0, 9.7)
    slow_vs_slow = combined_n_drives(8.5, 8.5, 9.7)
    assert fast_vs_fast > slow_vs_slow


def test_combined_n_drives_asymmetric_pace_averages():
    result = combined_n_drives(11.0, 8.4, 9.7)  # one fast, one slow team
    league_avg_case = combined_n_drives(9.7, 9.7, 9.7)
    assert result == pytest.approx(league_avg_case, abs=1)  # roughly cancels out


def test_combined_n_drives_never_below_one():
    assert combined_n_drives(0.0, 0.0, 9.7) >= 1
