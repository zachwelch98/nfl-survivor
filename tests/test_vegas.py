"""Tests for Method 1 (Vegas baseline): spread logistic, de-vig, tie modeling.

Per plan's correctness-test list, de-vig and tie handling get dedicated
coverage here (leakage is covered in test_clock.py; optimizer and simulator
sanity get their own files once those modules exist).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from survivor.models.base import assert_valid_output
from survivor.models.vegas import (
    ConstantTieModel,
    SpreadLogisticModel,
    VegasModel,
    devig_proportional,
    devig_shin,
    moneyline_to_implied_prob,
    sigma_from_games,
    spread_to_prob_normal,
)


# ---------------------------------------------------------------------------
# Moneyline -> raw implied probability
# ---------------------------------------------------------------------------


def test_moneyline_implied_prob_known_values():
    # +150 -> 100/250 = 0.4 ; -150 -> 150/250 = 0.6 ; even money either sign -> 0.5
    got = moneyline_to_implied_prob(np.array([150.0, -150.0, 100.0, -100.0]))
    np.testing.assert_allclose(got, [0.4, 0.6, 0.5, 0.5], atol=1e-9)


# ---------------------------------------------------------------------------
# De-vig
# ---------------------------------------------------------------------------


def test_devig_proportional_sums_to_one():
    home, away = devig_proportional(np.array([-200.0, 110.0]), np.array([170.0, -130.0]))
    np.testing.assert_allclose(home + away, [1.0, 1.0], atol=1e-9)


def test_devig_proportional_symmetric_odds_give_fifty_fifty():
    home, away = devig_proportional(np.array([-110.0]), np.array([-110.0]))
    np.testing.assert_allclose([home[0], away[0]], [0.5, 0.5], atol=1e-9)


def test_devig_shin_sums_to_one():
    home, away = devig_shin(np.array([-200.0, 110.0, -110.0]), np.array([170.0, -130.0, -110.0]))
    np.testing.assert_allclose(home + away, [1.0, 1.0, 1.0], atol=1e-6)


def test_devig_shin_matches_proportional_on_fair_odds():
    """At zero overround (a "fair" two-way market), Shin's z -> 0 and the
    method must reduce to plain proportional normalization."""
    home_ml = np.array([100.0])
    away_ml = np.array([-100.0])  # implied probs 0.5/0.5, no vig
    prop_h, prop_a = devig_proportional(home_ml, away_ml)
    shin_h, shin_a = devig_shin(home_ml, away_ml)
    np.testing.assert_allclose([shin_h[0], shin_a[0]], [prop_h[0], prop_a[0]], atol=1e-6)


def test_devig_shin_corrects_favorite_longshot_bias_direction():
    """On an asymmetric, vig-inflated market, Shin's method should assign the
    favorite a *higher* true probability than proportional normalization,
    and the underdog a lower one — the well-documented correction direction
    for the favorite-longshot bias (verified numerically here rather than
    derived, since the direction is the empirically load-bearing claim)."""
    home_ml, away_ml = np.array([-200.0]), np.array([170.0])  # home is the favorite
    prop_h, prop_a = devig_proportional(home_ml, away_ml)
    shin_h, shin_a = devig_shin(home_ml, away_ml)
    assert shin_h[0] > prop_h[0]
    assert shin_a[0] < prop_a[0]


def test_devig_handles_multiple_rows_independently():
    home, away = devig_shin(np.array([-200.0, -110.0, 300.0]), np.array([170.0, -110.0, -400.0]))
    assert home.shape == (3,)
    np.testing.assert_allclose(home + away, [1.0, 1.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Tie modeling
# ---------------------------------------------------------------------------


def _synthetic_games(margins: list[int], spreads: list[float]) -> pl.DataFrame:
    n = len(margins)
    return pl.DataFrame(
        {
            "season": [2020] * n,
            "week": list(range(1, n + 1)),
            "game_type": ["REG"] * n,
            "home_score": [10 + max(m, 0) for m in margins],
            "away_score": [10 + max(-m, 0) for m in margins],
            "spread_line": spreads,
        }
    )


def test_constant_tie_model_matches_empirical_rate():
    # 2 ties out of 10 games -> rate should be exactly 0.2
    margins = [0, 0, 3, -3, 7, -7, 10, -10, 1, -1]
    games = _synthetic_games(margins, [0.0] * 10)
    model = ConstantTieModel().fit(games)
    assert model.tie_rate == pytest.approx(0.2)
    preds = model.predict(5)
    assert preds.shape == (5,)
    assert (preds == 0.2).all()


def test_constant_tie_model_zero_when_no_completed_games():
    games = pl.DataFrame({"season": [2026], "week": [5], "game_type": ["REG"], "home_score": [None], "away_score": [None], "spread_line": [3.0]})
    model = ConstantTieModel().fit(games)
    assert model.tie_rate == 0.0


# ---------------------------------------------------------------------------
# Spread -> win probability
# ---------------------------------------------------------------------------


def test_spread_logistic_requires_minimum_sample_size():
    games = _synthetic_games([3, -3, 7], [2.0, -2.0, 5.0])
    with pytest.raises(ValueError):
        SpreadLogisticModel().fit(games)


def test_spread_logistic_positive_coefficient_and_monotonic_predictions():
    """A bigger home favorite (higher spread_line) must predict a higher win
    share — this is the core sanity check on sign convention."""
    rng = np.random.default_rng(0)
    n = 500
    spreads = rng.uniform(-14, 14, n)
    # simulate margin = spread + noise, home wins iff margin > 0
    margin = spreads + rng.normal(0, 13, n)
    margin = np.where(margin == 0, 1, margin)  # avoid accidental ties in synthetic data
    games = pl.DataFrame(
        {
            "season": [2020] * n,
            "week": (np.arange(n) % 18 + 1).tolist(),
            "game_type": ["REG"] * n,
            "home_score": np.where(margin > 0, 20 + np.abs(margin), 20).astype(int).tolist(),
            "away_score": np.where(margin > 0, 20, 20 + np.abs(margin)).astype(int).tolist(),
            "spread_line": spreads.tolist(),
        }
    )
    model = SpreadLogisticModel().fit(games)
    assert model.clf.coef_[0][0] > 0

    test_spreads = np.array([-10.0, -3.0, 0.0, 3.0, 10.0])
    preds = model.predict_win_share(test_spreads)
    assert np.all(np.diff(preds) > 0), "win share must increase monotonically with spread_line"


def test_spread_to_prob_normal_symmetric_around_zero():
    p = spread_to_prob_normal(np.array([-7.0, 0.0, 7.0]), sigma=13.5)
    assert p[1] == pytest.approx(0.5, abs=1e-9)
    assert p[0] == pytest.approx(1 - p[2], abs=1e-9)


def test_sigma_from_games_reasonable_on_real_data():
    from survivor.data.nflverse import load_schedules

    games = load_schedules()
    sigma = sigma_from_games(games)
    # NFL margin-vs-spread residual std is well documented to sit ~10-16 pts.
    assert 10.0 < sigma < 16.0


def test_sigma_from_games_fallback_on_insufficient_data():
    games = pl.DataFrame({"season": [], "week": [], "game_type": [], "home_score": [], "away_score": [], "spread_line": []})
    assert sigma_from_games(games) == 13.5


# ---------------------------------------------------------------------------
# Combined VegasModel, walk-forward against real data
# ---------------------------------------------------------------------------


def test_vegas_model_output_is_valid_probability_distribution():
    from survivor.clock import AsOf
    from survivor.data.nflverse import load_schedules

    games = load_schedules()
    asof = AsOf(season=2024, week=10, game_type="REG")
    train = asof.visible(games)
    target = games.filter((pl.col("season") == 2024) & (pl.col("week") == 10) & (pl.col("game_type") == "REG"))

    model = VegasModel().fit(train)
    preds = model.predict_proba(target)
    assert_valid_output(preds)
    assert preds.height == target.height
    assert preds["home_win_prob"].is_not_nan().all()


def test_vegas_model_falls_back_to_spread_only_pre_moneyline_era():
    """Before 2010, moneyline is sparse/absent — model must not crash or NaN."""
    from survivor.clock import AsOf
    from survivor.data.nflverse import load_schedules

    games = load_schedules()
    asof = AsOf(season=2005, week=10, game_type="REG")
    train = asof.visible(games)
    target = games.filter((pl.col("season") == 2005) & (pl.col("week") == 10) & (pl.col("game_type") == "REG"))

    model = VegasModel().fit(train)
    preds = model.predict_proba(target)
    assert_valid_output(preds)
