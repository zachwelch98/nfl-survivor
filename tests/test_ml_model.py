"""Tests for models/ml.py — Method 3, the supervised classifier.

Two things here are more than smoke tests. `test_fit_refuses_when_pbp_does_not
_cover_training_seasons` pins the guard that stops the model silently
training on median-imputed junk (measured at 96% of rows when pbp is
narrower than the schedule). `test_predict_proba_does_not_refit` pins the
season-cadence contract that everything downstream assumes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from survivor.clock import AsOf
from survivor.features.drive_rates import build_drive_table
from survivor.features.team_ratings import build_team_game_table
from survivor.models.base import assert_valid_output
from survivor.models.ml import CLASSIFIER_NAMES, MLWinProbModel

FAST = dict(calibration_method="none", min_train_games=100)


@pytest.fixture(scope="module")
def real_data():
    """Four seasons of pbp: enough real training rows for the model to be
    meaningful, small enough to keep the suite quick."""
    from survivor.data.nflverse import load_pbp, load_schedules

    schedules = load_schedules()
    pbp = load_pbp([2021, 2022, 2023, 2024])
    return schedules, build_team_game_table(pbp), build_drive_table(pbp)


def _train_test_split(schedules, train_through_season: int):
    """Expanding-window split: fit on everything before `train_through_season
    + 1`, score that next season. Same shape as the season-cadence pattern
    the real call sites use."""
    as_of = AsOf(train_through_season + 1, week=1)
    train = as_of.visible(schedules).filter(
        (pl.col("game_type") == "REG")
        & (pl.col("season") >= 2022)
        & pl.col("home_score").is_not_null()
    )
    target = schedules.filter(
        (pl.col("season") == train_through_season + 1)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") <= 6)
    )
    return train, target


def test_fit_refuses_below_minimum_usable_games(real_data):
    schedules, team_game_table, drive_table = real_data
    train, _ = _train_test_split(schedules, 2023)
    with pytest.raises(ValueError, match="at least 100000"):
        MLWinProbModel(team_game_table, drive_table, schedules, min_train_games=100000).fit(train)


def test_fit_refuses_when_pbp_does_not_cover_training_seasons(real_data):
    """The guard that matters: a schedule spanning 1999+ with pbp covering
    only 2021+ leaves almost every training row with no team-strength
    features at all. Median-imputing those would teach the model that
    average-looking features predict the base rate; refusing is correct.
    """
    schedules, team_game_table, drive_table = real_data
    full_history = AsOf(2024, week=1).visible(schedules).filter(
        (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
    )
    assert full_history.height > 5000  # mostly seasons with no pbp loaded

    with pytest.raises(ValueError, match="doesn't cover the training seasons"):
        MLWinProbModel(team_game_table, drive_table, schedules, min_train_games=2000).fit(full_history)


def test_fit_drops_unusable_rows_rather_than_imputing_them(real_data):
    schedules, team_game_table, drive_table = real_data
    full_history = AsOf(2024, week=1).visible(schedules).filter(
        (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
    )
    model = MLWinProbModel(team_game_table, drive_table, schedules, **FAST).fit(full_history)
    # Only the seasons with real play-by-play survive.
    assert model.n_train_games_ < model.n_candidate_games_ / 2
    assert model.n_train_games_ >= 100


@pytest.mark.parametrize("classifier", CLASSIFIER_NAMES)
def test_predict_proba_is_a_valid_distribution(real_data, classifier):
    schedules, team_game_table, drive_table = real_data
    train, target = _train_test_split(schedules, 2023)
    model = MLWinProbModel(team_game_table, drive_table, schedules, classifier=classifier, **FAST).fit(train)

    out = model.predict_proba(target)
    assert_valid_output(out)
    assert out.height == target.height
    assert out["home_win_prob"].min() > 0.0
    assert out["home_win_prob"].max() < 1.0


@pytest.mark.parametrize("calibration", ["sigmoid", "isotonic"])
def test_calibration_wrappers_produce_valid_output(real_data, calibration):
    schedules, team_game_table, drive_table = real_data
    train, target = _train_test_split(schedules, 2023)
    model = MLWinProbModel(
        team_game_table,
        drive_table,
        schedules,
        classifier="logistic",
        calibration_method=calibration,
        min_train_games=100,
    ).fit(train)
    assert_valid_output(model.predict_proba(target))


def test_predict_proba_before_fit_raises(real_data):
    schedules, team_game_table, drive_table = real_data
    model = MLWinProbModel(team_game_table, drive_table, schedules)
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict_proba(schedules.head(5))


def test_predict_proba_does_not_refit(real_data):
    """The season-cadence contract: within a season the model must be reused,
    never silently refit per week. Downstream call sites cache one fitted
    model per season on exactly this assumption."""
    schedules, team_game_table, drive_table = real_data
    train, target = _train_test_split(schedules, 2023)
    model = MLWinProbModel(team_game_table, drive_table, schedules, **FAST).fit(train)

    estimator_before = model.estimator_
    n_train_before = model.n_train_games_
    first = model.predict_proba(target)["home_win_prob"].to_list()
    second = model.predict_proba(target)["home_win_prob"].to_list()

    assert model.estimator_ is estimator_before
    assert model.n_train_games_ == n_train_before
    assert first == second


def test_model_favors_the_deterministically_stronger_team(real_data):
    """Directional sanity on real data: over a full slate, the side the model
    likes should win more often than not. A model that inverted its features
    would fail this while still producing valid-looking probabilities."""
    schedules, team_game_table, drive_table = real_data
    train, _ = _train_test_split(schedules, 2023)
    scored = schedules.filter(
        (pl.col("season") == 2024) & (pl.col("game_type") == "REG") & pl.col("home_score").is_not_null()
    )
    model = MLWinProbModel(team_game_table, drive_table, schedules, **FAST).fit(train)

    out = model.predict_proba(scored)
    decisive = out.filter(pl.col("home_score") != pl.col("away_score"))
    picked_home = decisive["home_win_prob"] > decisive["away_win_prob"]
    home_won = decisive["home_score"] > decisive["away_score"]
    accuracy = float((picked_home == home_won).mean())
    assert accuracy > 0.55, f"model picked the winner only {accuracy:.1%} of the time"


def test_feature_matrix_order_is_pinned_to_the_model_not_the_table(real_data):
    """Guards against a reordered feature table feeding column i's values
    into column j's coefficient."""
    schedules, team_game_table, drive_table = real_data
    train, target = _train_test_split(schedules, 2023)
    model = MLWinProbModel(team_game_table, drive_table, schedules, **FAST).fit(train)

    table = model._features(target, drop_ties=False)
    shuffled = table.select(list(reversed(table.columns)))
    np.testing.assert_array_equal(model._matrix(table), model._matrix(shuffled))

    with pytest.raises(ValueError, match="missing"):
        model._matrix(table.drop("proj_epa_diff"))
