"""Load config.yaml into a typed, importable object.

Single source of truth for paths and model/pool parameters — every other
module imports `CONFIG` from here rather than re-reading the YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

# Repo root is two levels up from this file: backend/survivor/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


class SeasonConfig(BaseModel):
    current: int
    moneyline_start: int
    spread_start: int
    backtest_seasons: tuple[int, int]


class PoolConfig(BaseModel):
    tie_counts_as_loss: bool
    allow_two_picks_late_season: bool


class VegasModelConfig(BaseModel):
    devig_method: Literal["proportional", "shin"]


class MonteCarloModelConfig(BaseModel):
    n_simulations: int
    seed: int
    rating_shrink_k: float
    prior_blend_prev_season: float
    prior_blend_league_mean: float


class BlendModelConfig(BaseModel):
    enabled: bool


class MLModelConfig(BaseModel):
    classifier: Literal["logistic", "random_forest", "hist_gb"]
    calibration_method: Literal["none", "sigmoid", "isotonic"]
    rating_shrink_k: float
    prior_blend_prev_season: float
    prior_blend_league_mean: float
    opp_adj_alpha: float
    rolling_windows: tuple[int, ...]
    rolling_min_games_floor: int
    min_train_games: int
    train_floor_season: int


class ModelConfig(BaseModel):
    primary_method: Literal["vegas", "montecarlo", "blend", "ml", "blend3"]
    vegas: VegasModelConfig
    montecarlo: MonteCarloModelConfig
    blend: BlendModelConfig
    ml: MLModelConfig


class OptimizerConfig(BaseModel):
    future_week_shrink_to_half: bool


class SimulationConfig(BaseModel):
    n_season_sims: int
    n_pool_field_entries: int


class PathsConfig(BaseModel):
    raw: str
    features: str
    cache: str
    popularity: str
    frontend_data: str

    def resolve(self, key: str) -> Path:
        return REPO_ROOT / getattr(self, key)


class Config(BaseModel):
    season: SeasonConfig
    pool: PoolConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    simulation: SimulationConfig
    paths: PathsConfig

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


CONFIG = Config.load()
