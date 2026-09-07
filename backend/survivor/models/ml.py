"""Method 3 (challenger): a supervised, calibrated win-probability model.

Where Method 1 reads the market and Method 2 simulates drives from a fixed
log5 formula, Method 3 *fits* the mapping from team/situational state to
outcome. It sees much of what Method 2 sees (the same matchup-resolved
drive-rate distributions) plus what Method 2 structurally cannot use:
opponent-adjusted ridge EPA, recent form, trajectory, rest/travel/weather,
and starting-QB continuity — see features/ml_features.py.

It deliberately does NOT take `spread_line`/moneyline as a feature. With
the line included it would largely learn to imitate Vegas, and a noisier
reconstruction of a signal Method 1 already reads directly is worth
nothing in a blend. More decisively: a future week's posted line cannot be
used at all in this project (optimize/projection.py's docstring covers
why), and future weeks are the entire reason Method 3 exists. A model that
needed a line would be useless exactly where it is needed.

## Refit cadence is the caller's job, not this class's

`fit(games)` fits whatever it is handed, exactly like `VegasModel` — this
class has no notion of a "season." The season-cadence discipline
(refit once per season on all strictly-prior seasons, reuse for every week
within it) is a call-site pattern:

    if season != fitted_for_season:
        train = AsOf(season, week=1).visible(schedules)
        model = MLWinProbModel(...).fit(train)
        fitted_for_season = season

That mirrors how `blend_model` is already threaded through
backtest/harness.py as a caller-supplied, pre-fit object, and it is why
this class is expensive-dependency-injected at construction rather than
through the `WinProbModel` protocol (whose `fit(games)` carries only a
schedules-shaped frame and could never pass play-by-play through).

Refitting weekly would be more current but wasteful: team ratings move
slowly within a season relative to what a retrain costs, and a 16-season
backtest would pay ~288 fits instead of ~16.

## Ties

Reuses `vegas.ConstantTieModel` rather than modeling ties again: ties are
~0.2% of games with no meaningful relationship to matchup strength, and
fitting a second curve to ~15 events would be fitting noise. The
classifier's output splits the remaining `1 - tie_prob` mass, structurally
identical to how VegasModel does it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from survivor.features.ml_features import build_feature_table, feature_columns
from survivor.models.vegas import ConstantTieModel

ClassifierName = Literal["logistic", "random_forest", "hist_gb"]
CalibrationMethod = Literal["none", "sigmoid", "isotonic"]

CLASSIFIER_NAMES: tuple[ClassifierName, ...] = ("logistic", "random_forest", "hist_gb")
CALIBRATION_METHODS: tuple[CalibrationMethod, ...] = ("none", "sigmoid", "isotonic")

# A row whose opponent-adjusted and naive EPA differentials are both null has
# no team-strength signal at all -- it happens when no play-by-play history
# exists for that as-of point, which is the normal state of affairs for games
# before the earliest season the caller loaded pbp for. Such rows must be
# dropped, not imputed: median-filling them would teach the model that
# "average-looking features" predict the raw home-win base rate, and with a
# narrow pbp load that junk can be 96% of the training set (measured). This
# is the guard that turns a silently-bad model into a loud error.
CORE_FEATURES: tuple[str, ...] = ("proj_epa_diff", "epa_per_play_diff")


def _make_classifier(name: ClassifierName, random_state: int):
    """Hyperparameters are deliberately conservative rather than tuned.

    ~5,000 training games against ~107 features is a regime where an
    unconstrained tree ensemble will happily memorize; the depth and
    min-samples-leaf floors below matter more for honest probabilities than
    any grid search would. Tuning these is a follow-up that should be
    judged by the same walk-forward evidence as everything else here, not
    by in-sample fit.
    """
    if name == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=random_state, n_jobs=-1
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=random_state
        )
    raise ValueError(f"unknown classifier {name!r}; expected one of {CLASSIFIER_NAMES}")


def build_estimator(
    classifier: ClassifierName,
    calibration_method: CalibrationMethod = "isotonic",
    calibration_cv: int = 5,
    random_state: int = 0,
):
    """The full unfitted estimator: impute -> scale -> classify, optionally
    wrapped in probability calibration.

    Exposed separately from MLWinProbModel so evaluate/ml_model_selection.py
    can score the grid against a precomputed feature table without rebuilding
    features once per combination — the model selection harness and the
    production model must be the same estimator, not two definitions that can
    drift.

    Calibration wraps the whole pipeline with `cv` folds taken *within*
    whatever training data it is given. Every caller passes strictly-prior
    seasons, so the calibration curve never sees the season being scored.
    """
    pipeline = Pipeline(
        [
            # Median imputation, not mean: several features are legitimately
            # null (no travel data for a neutral-site game, no opponent-
            # adjusted rating for an unseen team) and the median is the more
            # robust filler for the skewed ones.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", _make_classifier(classifier, random_state)),
        ]
    )
    if calibration_method == "none":
        return pipeline
    return CalibratedClassifierCV(pipeline, method=calibration_method, cv=calibration_cv)


class MLWinProbModel:
    """Implements the `WinProbModel` protocol (models/base.py)."""

    name = "ml"

    def __init__(
        self,
        team_game_table: pl.DataFrame,
        drive_table: pl.DataFrame,
        schedules: pl.DataFrame,
        depth_charts: pl.DataFrame | None = None,
        classifier: ClassifierName = "hist_gb",
        calibration_method: CalibrationMethod = "isotonic",
        shrink_k: float = 5.0,
        prior_blend_prev_season: float = 0.6,
        prior_blend_league_mean: float = 0.4,
        opp_adj_alpha: float = 40.0,
        rolling_windows: tuple[int, ...] = (3, 5),
        rolling_min_games_floor: int = 3,
        min_train_games: int = 200,
        calibration_cv: int = 5,
        random_state: int = 0,
    ) -> None:
        self.team_game_table = team_game_table
        self.drive_table = drive_table
        self.schedules = schedules
        self.depth_charts = depth_charts
        self.classifier = classifier
        self.calibration_method = calibration_method
        self.min_train_games = min_train_games
        self.calibration_cv = calibration_cv
        self.random_state = random_state
        self.rolling_windows = tuple(rolling_windows)
        self._feature_kwargs = dict(
            shrink_k=shrink_k,
            prior_blend_prev_season=prior_blend_prev_season,
            prior_blend_league_mean=prior_blend_league_mean,
            opp_adj_alpha=opp_adj_alpha,
            rolling_windows=self.rolling_windows,
            rolling_min_games_floor=rolling_min_games_floor,
        )
        self.feature_names = feature_columns(self.rolling_windows)
        self.estimator_ = None
        self.tie_model_ = ConstantTieModel()
        self.n_train_games_: int = 0
        self.n_candidate_games_: int = 0

    # -- internals ----------------------------------------------------------

    def _features(self, games: pl.DataFrame, drop_ties: bool) -> pl.DataFrame:
        return build_feature_table(
            games,
            self.team_game_table,
            self.drive_table,
            self.schedules,
            self.depth_charts,
            drop_ties=drop_ties,
            **self._feature_kwargs,
        )

    def _matrix(self, table: pl.DataFrame) -> np.ndarray:
        """Feature columns as a float matrix, in `self.feature_names` order.

        The order is pinned to the model's own recorded list rather than the
        table's column order so that a table built with different options —
        or simply a different polars version's ordering — can never silently
        feed column i's values into column j's coefficient.
        """
        missing = [c for c in self.feature_names if c not in table.columns]
        if missing:
            raise ValueError(f"feature table is missing {len(missing)} expected columns, e.g. {missing[:5]}")
        return (
            table.select([pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names])
            .to_numpy()
            .astype(float)
        )

    # -- WinProbModel protocol ---------------------------------------------

    def fit(self, games: pl.DataFrame) -> "MLWinProbModel":
        """`games` must already be `as_of.visible(schedules)` — same contract
        as every other model here (models/base.py: enforcing no-lookahead is
        `AsOf`'s job, not the model interface's)."""
        table = self._features(games, drop_ties=True)
        if table.height == 0:
            raise ValueError("no usable training rows (empty feature table)")

        labelled = table.filter(pl.col("home_win").is_not_null())
        usable = labelled.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in CORE_FEATURES])
        )
        self.n_candidate_games_ = labelled.height
        self.n_train_games_ = usable.height

        if usable.height < self.min_train_games:
            raise ValueError(
                f"need at least {self.min_train_games} decisive games with usable features to fit "
                f"MLWinProbModel, got {usable.height} of {labelled.height} candidates. "
                "If most rows were dropped, the play-by-play passed in likely doesn't cover the "
                "training seasons -- MLWinProbModel needs pbp back to the earliest training season, "
                "unlike Method 2, which only ever looks back one season from each as-of point."
            )
        labelled = usable

        x = self._matrix(labelled)
        y = labelled["home_win"].to_numpy().astype(int)

        self.estimator_ = build_estimator(
            self.classifier, self.calibration_method, self.calibration_cv, self.random_state
        ).fit(x, y)

        self.tie_model_.fit(games)
        return self

    def predict_proba(self, games: pl.DataFrame) -> pl.DataFrame:
        """Adds home_win_prob/away_win_prob/tie_prob to `games`.

        Handles training rows and never-played future rows identically —
        `resolve_starter`'s null check is what absorbs the difference (see
        features/ml_features.py).
        """
        if self.estimator_ is None:
            raise RuntimeError("call fit() before predict_proba()")

        n = games.height
        tie_prob = self.tie_model_.predict(n)
        win_mass = 1.0 - tie_prob

        table = self._features(games, drop_ties=False)
        home_share = np.full(n, 0.5)

        if table.height > 0:
            predicted = self.estimator_.predict_proba(self._matrix(table))[:, 1]
            # build_feature_table can legitimately return fewer/reordered rows
            # than it was given, so map back by game_id rather than trusting
            # positional alignment.
            by_game = dict(zip(table["game_id"].to_list(), predicted))
            home_share = np.array([by_game.get(gid, 0.5) for gid in games["game_id"].to_list()])

        return games.with_columns(
            [
                pl.Series("home_win_prob", win_mass * home_share),
                pl.Series("away_win_prob", win_mass * (1.0 - home_share)),
                pl.Series("tie_prob", tie_prob),
            ]
        )
