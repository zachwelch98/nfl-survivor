"""Walk-forward model selection for Method 3 (models/ml.py).

Which classifier and which probability calibration? This module answers that
with evidence rather than assumption — the same standard `config.yaml`'s
`primary_method` comment already holds Vegas-vs-Monte-Carlo to.

## The validation scheme

Expanding window at season granularity, exactly the schedule the model is
trained on in production:

    fit on 2010..2014  ->  score every game of 2015
    fit on 2010..2015  ->  score every game of 2016
    ...
    fit on 2010..2024  ->  score every game of 2025

No game is ever scored by a model that saw its season. There is no random
train/test split anywhere in this file, and the per-game features are
independently as-of gated (features/ml_features.py), so the two halves of
the leakage story are separately enforced.

## Why features are built once

Every feature row is as-of by construction: the row for a 2018 week 3 game
is identical whether it was built while evaluating 2019 or 2023. So the
whole span's features are built once and sliced per fold, rather than
rebuilt 9 × 11 times. The only thing that varies per fold is which rows are
train and which are test.

## What decides the winner

Ranked, in order:
  1. Lowest pooled Brier across all validation games (pooled, not averaged
     per season, so one short season can't distort the ranking).
  2. A bootstrap CI on the difference vs. the runner-up that excludes zero —
     otherwise the "win" is noise and the tie-break below applies.
  3. Calibration in the 0.85-0.99 band, which is where survivor picks
     actually live. This is the criterion that settled Vegas-vs-Monte-Carlo,
     and a model that is excellent on average but overconfident in that band
     is useless for this application specifically.
  4. Simplicity (logistic < random_forest < hist_gb) when the above tie.

Log loss, ROC-AUC and accuracy are reported alongside but do not select:
accuracy in particular is close to meaningless for a survivor pool, where
the difference between an 88% and a 94% favorite matters enormously and
both round to the same "pick".
"""

from __future__ import annotations

import numpy as np
import polars as pl

from survivor.clock import AsOf
from survivor.evaluate.calibration import high_confidence_slice, reliability_curve
from survivor.evaluate.metrics import bootstrap_metric_diff, brier_score, log_loss
from survivor.features.ml_features import build_feature_table, feature_columns
from survivor.models.ml import CALIBRATION_METHODS, CLASSIFIER_NAMES, CORE_FEATURES, build_estimator

# Ordered simplest-first; used as the final tie-break.
_COMPLEXITY_ORDER = {"logistic": 0, "random_forest": 1, "hist_gb": 2}


def build_walkforward_features(
    schedules: pl.DataFrame,
    team_game_table: pl.DataFrame,
    drive_table: pl.DataFrame,
    depth_charts: pl.DataFrame | None,
    train_floor_season: int,
    last_season: int,
    rolling_windows: tuple[int, ...] = (3, 5),
    **feature_kwargs,
) -> pl.DataFrame:
    """Features for every completed, decisive REG game in
    [train_floor_season, last_season] — built once, sliced per fold."""
    games = schedules.filter(
        (pl.col("game_type") == "REG")
        & (pl.col("season") >= train_floor_season)
        & (pl.col("season") <= last_season)
        & pl.col("home_score").is_not_null()
    )
    table = build_feature_table(
        games,
        team_game_table,
        drive_table,
        schedules,
        depth_charts,
        rolling_windows=rolling_windows,
        drop_ties=True,
        **feature_kwargs,
    )
    if table.height == 0:
        return table
    # Same guard MLWinProbModel.fit applies: a row with no team-strength
    # signal is not a training example, it's an absence of one.
    return table.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in CORE_FEATURES]))


def walkforward_predictions(
    features: pl.DataFrame,
    validate_seasons: list[int],
    classifier: str,
    calibration_method: str,
    rolling_windows: tuple[int, ...] = (3, 5),
    min_train_games: int = 200,
    calibration_cv: int = 5,
    random_state: int = 0,
) -> pl.DataFrame:
    """One row per scored game: season, week, game_id, home_win, ml_home_win_prob.

    Each validation season is scored by a single model fit on every earlier
    season in `features` — the season-cadence pattern production uses, so
    what is measured here is what actually ships.
    """
    feature_names = list(feature_columns(rolling_windows))
    out = []

    for season in sorted(validate_seasons):
        train = features.filter(pl.col("season") < season)
        test = features.filter(pl.col("season") == season)
        if train.height < min_train_games or test.height == 0:
            continue

        x_train = train.select(feature_names).to_numpy().astype(float)
        y_train = train["home_win"].to_numpy().astype(int)
        estimator = build_estimator(classifier, calibration_method, calibration_cv, random_state).fit(x_train, y_train)

        x_test = test.select(feature_names).to_numpy().astype(float)
        out.append(
            test.select(["game_id", "season", "week", "home_team", "away_team", "home_win"]).with_columns(
                pl.Series("ml_home_win_prob", estimator.predict_proba(x_test)[:, 1]),
                pl.lit(train.height).alias("n_train_games"),
            )
        )

    if not out:
        return pl.DataFrame()
    return pl.concat(out)


def score_predictions(predictions: pl.DataFrame, reference_prob: np.ndarray | None = None) -> dict:
    """Brier/log loss/AUC/accuracy plus reliability and the high-confidence
    slice, over whatever set of predictions it's given."""
    prob = predictions["ml_home_win_prob"].to_numpy()
    outcome = predictions["home_win"].to_numpy().astype(float)

    result = {
        "n_games": int(len(outcome)),
        "brier": brier_score(prob, outcome),
        "log_loss": log_loss(prob, outcome),
        "roc_auc": _roc_auc(prob, outcome),
        "accuracy": float(((prob >= 0.5).astype(float) == outcome).mean()),
        "reliability": [
            {
                "bin_low": b.bin_low,
                "bin_high": b.bin_high,
                "n": b.n,
                "mean_predicted": b.mean_predicted,
                "empirical_rate": b.empirical_rate,
            }
            for b in reliability_curve(prob, outcome)
        ],
        "high_confidence_slice": _slice_dict(high_confidence_slice(prob, outcome)),
        "by_season": {
            int(season): {
                "n_games": int(group.height),
                "brier": brier_score(group["ml_home_win_prob"].to_numpy(), group["home_win"].to_numpy().astype(float)),
                "log_loss": log_loss(group["ml_home_win_prob"].to_numpy(), group["home_win"].to_numpy().astype(float)),
                "n_train_games": int(group["n_train_games"][0]) if "n_train_games" in group.columns else None,
            }
            for (season,), group in predictions.group_by(["season"], maintain_order=True)
        },
    }
    if reference_prob is not None:
        from survivor.evaluate.metrics import brier_skill_score

        result["brier_skill_score_vs_reference"] = brier_skill_score(prob, outcome, reference_prob)
    return result


def _slice_dict(s) -> dict:
    return {
        "threshold_low": s.threshold_low,
        "threshold_high": s.threshold_high,
        "n": s.n,
        "brier": s.brier,
        "log_loss": s.log_loss,
        "empirical_win_rate": s.empirical_win_rate,
        "mean_predicted": s.mean_predicted,
    }


def _roc_auc(prob: np.ndarray, outcome: np.ndarray) -> float | None:
    """Rank-based AUC via the Mann-Whitney U identity — no sklearn.metrics
    import needed, and it handles ties in `prob` correctly via average ranks."""
    pos = prob[outcome == 1]
    neg = prob[outcome == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks within ties so a constant predictor scores 0.5, not 1.0
    values = np.concatenate([pos, neg])
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run_model_selection_grid(
    features: pl.DataFrame,
    validate_seasons: list[int],
    classifiers: tuple[str, ...] = CLASSIFIER_NAMES,
    calibration_methods: tuple[str, ...] = CALIBRATION_METHODS,
    rolling_windows: tuple[int, ...] = (3, 5),
    min_train_games: int = 200,
    n_boot: int = 2000,
    seed: int = 0,
    progress: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Score every (classifier, calibration) combination walk-forward.

    Returns `(summary_table, detail)` — one summary row per combination
    sorted best-Brier-first, and a dict of the full per-combination scoring
    (reliability curves, per-season breakdown) keyed by "classifier+calibration".
    """
    rows = []
    detail: dict[str, dict] = {}
    predictions_by_combo: dict[str, pl.DataFrame] = {}

    for classifier in classifiers:
        for calibration in calibration_methods:
            key = f"{classifier}+{calibration}"
            if progress:
                print(f"  fitting {key} ...", flush=True)
            preds = walkforward_predictions(
                features,
                validate_seasons,
                classifier,
                calibration,
                rolling_windows=rolling_windows,
                min_train_games=min_train_games,
            )
            if preds.height == 0:
                continue
            scored = score_predictions(preds)
            detail[key] = scored
            predictions_by_combo[key] = preds
            hcs = scored["high_confidence_slice"]
            rows.append(
                {
                    "combo": key,
                    "classifier": classifier,
                    "calibration": calibration,
                    "n_games": scored["n_games"],
                    "brier": scored["brier"],
                    "log_loss": scored["log_loss"],
                    "roc_auc": scored["roc_auc"],
                    "accuracy": scored["accuracy"],
                    "high_conf_n": hcs["n"],
                    "high_conf_predicted": hcs["mean_predicted"],
                    "high_conf_actual": hcs["empirical_win_rate"],
                    "high_conf_gap": (
                        None
                        if hcs["mean_predicted"] is None or hcs["empirical_win_rate"] is None
                        else hcs["mean_predicted"] - hcs["empirical_win_rate"]
                    ),
                    "complexity": _COMPLEXITY_ORDER[classifier],
                }
            )

    if not rows:
        return pl.DataFrame(), detail

    summary = pl.DataFrame(rows).sort("brier")

    # Bootstrap the leader against the runner-up on their common games: a
    # point-estimate win that a CI can't separate from zero is not a win, and
    # in that case the simpler model should be preferred instead.
    if summary.height >= 2:
        best_key, runner_key = summary["combo"][0], summary["combo"][1]
        best, runner = predictions_by_combo[best_key], predictions_by_combo[runner_key]
        joined = best.select(["game_id", "home_win", "ml_home_win_prob"]).join(
            runner.select(["game_id", pl.col("ml_home_win_prob").alias("runner_prob")]), on="game_id", how="inner"
        )
        boot = bootstrap_metric_diff(
            joined["ml_home_win_prob"].to_numpy(),
            joined["runner_prob"].to_numpy(),
            joined["home_win"].to_numpy().astype(float),
            brier_score,
            n_boot=n_boot,
            seed=seed,
        )
        detail["leader_vs_runner_up"] = {
            "leader": best_key,
            "runner_up": runner_key,
            # brier_score is a loss, so a *negative* difference means the
            # leader really is better; "significant" = the CI excludes zero.
            "brier_diff": boot.point_estimate,
            "ci_low": boot.ci_low,
            "ci_high": boot.ci_high,
            "leader_significantly_better": bool(boot.excludes_zero and boot.point_estimate < 0),
        }

    return summary, detail


def recommend_combination(summary: pl.DataFrame, detail: dict) -> dict:
    """Apply the selection rules in order and explain which one decided it."""
    if summary.height == 0:
        return {"classifier": None, "calibration_method": None, "reason": "no combinations produced predictions"}

    leader = summary.row(0, named=True)
    if summary.height == 1:
        return {**_pick(leader), "reason": "only one combination scored"}

    comparison = detail.get("leader_vs_runner_up", {})
    if comparison.get("leader_significantly_better"):
        return {
            **_pick(leader),
            "reason": (
                f"lowest pooled Brier ({leader['brier']:.4f}) and the bootstrap CI vs "
                f"{comparison['runner_up']} excludes zero "
                f"([{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}])"
            ),
        }

    # Not separable on Brier: fall to high-confidence calibration, then simplicity.
    contenders = summary.filter(pl.col("brier") <= leader["brier"] + 0.005)
    with_gap = contenders.filter(pl.col("high_conf_gap").is_not_null())
    if with_gap.height > 0:
        best_calibrated = with_gap.with_columns(pl.col("high_conf_gap").abs().alias("abs_gap")).sort(
            ["abs_gap", "complexity"]
        ).row(0, named=True)
        return {
            **_pick(best_calibrated),
            "reason": (
                "Brier differences among the leaders are not statistically separable, so this was decided on "
                f"calibration in the 0.85-0.99 band (predicted {best_calibrated['high_conf_predicted']:.3f} "
                f"vs actual {best_calibrated['high_conf_actual']:.3f})"
            ),
        }

    simplest = contenders.sort(["complexity", "brier"]).row(0, named=True)
    return {**_pick(simplest), "reason": "statistically indistinguishable on every criterion; preferred the simplest model"}


def _pick(row: dict) -> dict:
    return {"classifier": row["classifier"], "calibration_method": row["calibration"], "brier": row["brier"]}
