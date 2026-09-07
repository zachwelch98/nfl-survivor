"""Walk-forward comparison of Method 1 (Vegas) vs Method 2 (Monte Carlo) vs
the fitted Blend (plan §3) — the step that actually decides which method
(or blend) the optimizer should trust, rather than assuming the answer.

Two passes, each walk-forward on its own terms:
  1. `collect_predictions` produces one row per decisive REG game with both
     methods' home-win probability, each computed using only data visible
     strictly before that game's week (the same AsOf discipline as
     everywhere else in this project).
  2. `add_walkforward_blend` fits BlendModel at each week using only
     *already-collected* (Vegas, MC, outcome) triples from strictly earlier
     weeks, and predicts on the current week — i.e. the blend itself is
     evaluated without ever seeing its own test games during fitting. This
     is a second, independent walk-forward discipline layered on top of the
     first (the raw method predictions being walk-forward doesn't
     automatically make a *blend fit on top of them* walk-forward too).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from survivor.clock import AsOf
from survivor.features.drive_rates import build_drive_table, compute_defense_profiles, compute_offense_profiles, league_average_profile
from survivor.models.blend import BlendModel
from survivor.models.montecarlo import DriveSimConfig, combine_matchup_rates, combined_n_drives, simulate_game
from survivor.models.vegas import VegasModel


def collect_predictions(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    weeks: list[tuple[int, int]],
    shrink_k: float = 5.0,
    prior_blend_prev_season: float = 0.6,
    prior_blend_league_mean: float = 0.4,
    mc_config: DriveSimConfig | None = None,
    min_train_games: int = 20,
) -> pl.DataFrame:
    """One row per decisive REG game across the given (season, week) list:
    vegas_home_win_prob, mc_home_win_prob, home_win (0/1). Ties are
    excluded (both methods' win-share metrics are only meaningful among
    decisive outcomes; tie handling is evaluated separately — see
    models/base.py's pick_survival_prob).
    """
    mc_config = mc_config or DriveSimConfig()
    drive_table = build_drive_table(pbp)
    rows = []

    for season, week in weeks:
        as_of = AsOf(season=season, week=week, game_type="REG")
        train_games = as_of.visible(schedules)
        target = schedules.filter(
            (pl.col("season") == season) & (pl.col("week") == week) & (pl.col("game_type") == "REG")
        ).filter(pl.col("home_score").is_not_null() & (pl.col("home_score") != pl.col("away_score")))
        if target.height == 0:
            continue

        decisive_train = train_games.filter(
            pl.col("home_score").is_not_null() & pl.col("spread_line").is_not_null() & (pl.col("home_score") != pl.col("away_score"))
        )
        if decisive_train.height < min_train_games:
            continue
        vegas = VegasModel().fit(train_games)
        vegas_preds = vegas.predict_proba(target)

        off = compute_offense_profiles(drive_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean)
        deff = compute_defense_profiles(drive_table, as_of, shrink_k, prior_blend_prev_season, prior_blend_league_mean)
        league = league_average_profile(drive_table, as_of)

        for row, vegas_row in zip(target.iter_rows(named=True), vegas_preds.iter_rows(named=True)):
            home, away = row["home_team"], row["away_team"]
            if home not in off or away not in off or home not in deff or away not in deff:
                continue
            home_rates = combine_matchup_rates(off[home].rates, deff[away].rates, league.rates)
            away_rates = combine_matchup_rates(off[away].rates, deff[home].rates, league.rates)
            n_drives = combined_n_drives(off[home].drives_per_game, off[away].drives_per_game, league.drives_per_game)
            sim = simulate_game(home_rates, away_rates, n_drives, mc_config)
            mc_home_share = sim.home_win_prob / (sim.home_win_prob + sim.away_win_prob)
            vegas_home_share = vegas_row["home_win_prob"] / (vegas_row["home_win_prob"] + vegas_row["away_win_prob"])

            rows.append(
                {
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_win": int(row["home_score"] > row["away_score"]),
                    "vegas_home_win_prob": vegas_home_share,
                    "mc_home_win_prob": mc_home_share,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8,
                "home_win": pl.Int64, "vegas_home_win_prob": pl.Float64, "mc_home_win_prob": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


def add_walkforward_blend(predictions: pl.DataFrame, min_train_games: int = 100) -> pl.DataFrame:
    """Adds a `blend_home_win_prob` column, fitting BlendModel at each
    distinct (season, week) using only strictly earlier rows. Rows before
    `min_train_games` is reached get null blend predictions (excluded from
    blend evaluation, not silently zero-filled).
    """
    predictions = predictions.sort(["season", "week"])
    weeks = predictions.select(["season", "week"]).unique().sort(["season", "week"]).iter_rows()

    blend_preds = np.full(predictions.height, np.nan)
    row_season = predictions["season"].to_numpy()
    row_week = predictions["week"].to_numpy()
    vegas_arr = predictions["vegas_home_win_prob"].to_numpy()
    mc_arr = predictions["mc_home_win_prob"].to_numpy()
    outcome_arr = predictions["home_win"].to_numpy()

    for season, week in weeks:
        train_mask = (row_season < season) | ((row_season == season) & (row_week < week))
        test_mask = (row_season == season) & (row_week == week)
        if train_mask.sum() < min_train_games:
            continue
        model = BlendModel().fit(
            {"vegas": vegas_arr[train_mask], "mc": mc_arr[train_mask]}, outcome_arr[train_mask]
        )
        blend_preds[test_mask] = model.predict({"vegas": vegas_arr[test_mask], "mc": mc_arr[test_mask]})

    # NaN (unset — not enough training data yet) must become an actual
    # polars null, not float NaN, so `.is_null()`/`.is_not_null()` behave as
    # callers expect (polars treats NaN and null as distinct).
    blend_series = pl.Series("blend_home_win_prob", blend_preds)
    blend_series = pl.when(blend_series.is_nan()).then(None).otherwise(blend_series).alias("blend_home_win_prob")
    return predictions.with_columns(blend_series)


def summarize_comparison(predictions: pl.DataFrame, n_boot: int = 3000, seed: int = 0) -> dict:
    """Package the full plan §3 comparison into one JSON-serializable dict:
    overall Brier/log loss/BSS + bootstrap CI, week-bucket breakdown,
    high-confidence-band slice, and (where present) the blend's numbers.
    This is the payload the frontend's "Historical analysis" view and the
    weekly-operation method choice both consume — see plan §3 and §9.
    """
    from survivor.evaluate.calibration import high_confidence_slice, reliability_curve
    from survivor.evaluate.metrics import bootstrap_metric_diff, brier_score, brier_skill_score, log_loss, metrics_by_group

    outcome = predictions["home_win"].to_numpy().astype(float)
    vegas = predictions["vegas_home_win_prob"].to_numpy()
    mc = predictions["mc_home_win_prob"].to_numpy()
    weeks = predictions["week"].to_numpy()

    week_bucket = np.select(
        [weeks <= 4, weeks <= 9, weeks <= 14],
        ["wk1-4", "wk5-9", "wk10-14"],
        default="wk15+",
    )

    boot = bootstrap_metric_diff(mc, vegas, outcome, metric_fn=brier_score, n_boot=n_boot, seed=seed)

    result = {
        "n_games": int(len(outcome)),
        "vegas": {
            "brier": brier_score(vegas, outcome),
            "log_loss": log_loss(vegas, outcome),
            "reliability": [b.__dict__ for b in reliability_curve(vegas, outcome)],
            "high_confidence_slice": high_confidence_slice(vegas, outcome).__dict__,
            "by_week_bucket": metrics_by_group(vegas, outcome, week_bucket),
        },
        "montecarlo": {
            "brier": brier_score(mc, outcome),
            "log_loss": log_loss(mc, outcome),
            "reliability": [b.__dict__ for b in reliability_curve(mc, outcome)],
            "high_confidence_slice": high_confidence_slice(mc, outcome).__dict__,
            "by_week_bucket": metrics_by_group(mc, outcome, week_bucket),
        },
        "mc_vs_vegas_brier_skill_score": brier_skill_score(mc, outcome, reference_prob=vegas),
        "mc_vs_vegas_bootstrap": {
            "point_estimate": boot.point_estimate,
            "ci_low": boot.ci_low,
            "ci_high": boot.ci_high,
            "mc_significantly_better": boot.excludes_zero and boot.point_estimate > 0,
            "vegas_significantly_better": boot.excludes_zero and boot.point_estimate < 0,
        },
    }

    if "blend_home_win_prob" in predictions.columns:
        blend_eval = predictions.filter(pl.col("blend_home_win_prob").is_not_null())
        if blend_eval.height > 0:
            b_outcome = blend_eval["home_win"].to_numpy().astype(float)
            b_vegas = blend_eval["vegas_home_win_prob"].to_numpy()
            b_blend = blend_eval["blend_home_win_prob"].to_numpy()
            boot_blend = bootstrap_metric_diff(b_blend, b_vegas, b_outcome, metric_fn=brier_score, n_boot=n_boot, seed=seed + 1)
            result["blend"] = {
                "n_games": int(blend_eval.height),
                "brier": brier_score(b_blend, b_outcome),
                "log_loss": log_loss(b_blend, b_outcome),
                "vegas_on_same_subset_brier": brier_score(b_vegas, b_outcome),
                "blend_vs_vegas_bootstrap": {
                    "point_estimate": boot_blend.point_estimate,
                    "ci_low": boot_blend.ci_low,
                    "ci_high": boot_blend.ci_high,
                    "blend_significantly_better": boot_blend.excludes_zero and boot_blend.point_estimate > 0,
                },
            }

    return result
