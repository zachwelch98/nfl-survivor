"""Fitted ensemble of Method 1 (Vegas) and Method 2 (Monte Carlo) — plan §2/§3.

A logistic-regression stack on the two methods' logit-transformed home-win
probabilities: `logit(p_blend) = a + b1*logit(p_vegas) + b2*logit(p_mc)`,
fit on prior seasons only (walk-forward, same discipline as every other
model here). Whether this blend actually beats either input alone is an
empirical question answered by evaluate/model_comparison.py — it is not
assumed here, and CONFIG.model.blend.enabled exists specifically so a
caller can turn blending off if the evidence doesn't support it.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit_array(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class BlendModel:
    name = "blend"

    def __init__(self) -> None:
        self.clf: LogisticRegression | None = None

    def fit(self, vegas_home_win_prob: np.ndarray, mc_home_win_prob: np.ndarray, home_win_outcome: np.ndarray) -> "BlendModel":
        """`home_win_outcome` must be 0/1 over *decisive* games only (ties
        excluded) — same convention as SpreadLogisticModel."""
        n = len(home_win_outcome)
        if n < 20:
            raise ValueError(f"need at least 20 decisive games to fit BlendModel, got {n}")
        x = np.column_stack([_logit_array(vegas_home_win_prob), _logit_array(mc_home_win_prob)])
        y = np.asarray(home_win_outcome, dtype=int)
        self.clf = LogisticRegression()
        self.clf.fit(x, y)
        return self

    def predict(self, vegas_home_win_prob: np.ndarray, mc_home_win_prob: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("call fit() before predict()")
        x = np.column_stack([_logit_array(vegas_home_win_prob), _logit_array(mc_home_win_prob)])
        return self.clf.predict_proba(x)[:, 1]
