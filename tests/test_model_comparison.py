"""Tests for evaluate/model_comparison.py: walk-forward blend leakage safety
and a real-data smoke test of the full collection pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from survivor.evaluate.model_comparison import add_walkforward_blend, collect_predictions


def _synthetic_predictions(n_weeks: int, games_per_week: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, n_weeks + 1):
        for _ in range(games_per_week):
            true_p = rng.uniform(0.2, 0.8)
            outcome = int(rng.uniform(0, 1) < true_p)
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "home_team": "AAA",
                    "away_team": "BBB",
                    "home_win": outcome,
                    "vegas_home_win_prob": float(np.clip(true_p + rng.normal(0, 0.05), 0.02, 0.98)),
                    "mc_home_win_prob": float(np.clip(true_p + rng.normal(0, 0.08), 0.02, 0.98)),
                }
            )
    return pl.DataFrame(rows)


def test_walkforward_blend_no_leakage_from_future_weeks():
    """Blend predictions for early weeks must be identical whether or not
    later weeks' data is present in the input at all."""
    full = _synthetic_predictions(n_weeks=20, games_per_week=10, seed=0)
    truncated = full.filter(pl.col("week") <= 12)

    full_result = add_walkforward_blend(full, min_train_games=50)
    truncated_result = add_walkforward_blend(truncated, min_train_games=50)

    full_early = full_result.filter(pl.col("week") <= 12).sort(["week", "home_team"])
    truncated_early = truncated_result.sort(["week", "home_team"])

    full_vals = full_early["blend_home_win_prob"].to_numpy()
    trunc_vals = truncated_early["blend_home_win_prob"].to_numpy()

    both_notnan = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)
    assert both_notnan.sum() > 0
    np.testing.assert_allclose(full_vals[both_notnan], trunc_vals[both_notnan], atol=1e-9)


def test_walkforward_blend_early_weeks_below_min_train_are_null():
    preds = _synthetic_predictions(n_weeks=5, games_per_week=5, seed=1)
    result = add_walkforward_blend(preds, min_train_games=1000)  # impossible to reach
    assert result["blend_home_win_prob"].is_null().all()


def test_walkforward_blend_improves_over_noisy_input_eventually():
    preds = _synthetic_predictions(n_weeks=25, games_per_week=14, seed=2)
    result = add_walkforward_blend(preds, min_train_games=100)
    evaluated = result.filter(pl.col("blend_home_win_prob").is_not_null())
    assert evaluated.height > 0

    from survivor.evaluate.metrics import brier_score

    bs_blend = brier_score(evaluated["blend_home_win_prob"].to_numpy(), evaluated["home_win"].to_numpy().astype(float))
    bs_vegas = brier_score(evaluated["vegas_home_win_prob"].to_numpy(), evaluated["home_win"].to_numpy().astype(float))
    # blend shouldn't be dramatically worse than the better input
    assert bs_blend < bs_vegas + 0.02


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------


def test_collect_predictions_real_data_smoke():
    from survivor.data.nflverse import load_pbp, load_schedules
    from survivor.models.montecarlo import DriveSimConfig

    pbp = load_pbp([2023, 2024])
    schedules = load_schedules()
    weeks = [(2024, w) for w in range(9, 11)]  # 2 weeks, keeps runtime small

    preds = collect_predictions(pbp, schedules, weeks, mc_config=DriveSimConfig(n_simulations=2000, seed=0))
    assert preds.height > 0
    assert set(preds.columns) >= {"season", "week", "home_team", "away_team", "home_win", "vegas_home_win_prob", "mc_home_win_prob"}
    assert preds["vegas_home_win_prob"].is_between(0, 1).all()
    assert preds["mc_home_win_prob"].is_between(0, 1).all()
    assert preds["home_win"].is_in([0, 1]).all()


def test_summarize_comparison_structure_and_significance_flags():
    from survivor.evaluate.model_comparison import summarize_comparison

    # Vegas clearly better, MC pure noise -> vegas should come out significantly better
    n = 1500
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.2, 0.8, n)
    outcome = (rng.uniform(0, 1, n) < true_p).astype(int)
    vegas = np.clip(true_p + rng.normal(0, 0.03, n), 0.02, 0.98)
    mc = np.clip(rng.uniform(0.3, 0.7, n), 0.02, 0.98)
    weeks = (np.arange(n) % 18 + 1)

    preds = pl.DataFrame(
        {
            "season": [2024] * n,
            "week": weeks.tolist(),
            "home_team": ["AAA"] * n,
            "away_team": ["BBB"] * n,
            "home_win": outcome.tolist(),
            "vegas_home_win_prob": vegas.tolist(),
            "mc_home_win_prob": mc.tolist(),
        }
    )
    summary = summarize_comparison(preds, n_boot=500, seed=0)
    assert summary["n_games"] == n
    assert summary["mc_vs_vegas_bootstrap"]["vegas_significantly_better"] is True
    assert summary["mc_vs_vegas_bootstrap"]["mc_significantly_better"] is False
    assert set(summary["vegas"].keys()) >= {"brier", "log_loss", "reliability", "high_confidence_slice", "by_week_bucket"}
    assert "blend" not in summary  # no blend column supplied


def test_summarize_comparison_includes_blend_when_present():
    from survivor.evaluate.model_comparison import add_walkforward_blend, summarize_comparison

    preds = _synthetic_predictions(n_weeks=25, games_per_week=14, seed=5)
    preds = add_walkforward_blend(preds, min_train_games=100)
    summary = summarize_comparison(preds, n_boot=300, seed=1)
    assert "blend" in summary
    assert "blend_vs_vegas_bootstrap" in summary["blend"]
