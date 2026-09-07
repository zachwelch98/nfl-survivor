"""Fitted ensemble of the win-probability methods — plan §2/§3.

A logistic-regression stack on each method's logit-transformed home-win
probability: `logit(p_blend) = a + Σ b_i * logit(p_i)`, fit on prior
seasons only (walk-forward, same discipline as every other model here).

Inputs are named rather than positional, so the same class serves the
original Vegas+Monte-Carlo pair and the three-way blend that includes
Method 3 (models/ml.py) without a second implementation. The fitted key
set is remembered and re-checked at predict time — with N inputs stacked
into an unlabeled matrix, silently swapping two columns between fit and
predict would produce plausible-looking garbage rather than an error.

Whether blending actually beats its best single input is an empirical
question answered by evaluate/model_comparison.py — it is not assumed
here, and CONFIG.model.blend.enabled exists specifically so a caller can
turn it off if the evidence doesn't support it.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from survivor.models._prob_utils import logit_array as _logit_array

# Minimum decisive games required *per input model*. Two inputs need 40,
# three need 60: each additional input is another coefficient to estimate,
# and a flat floor that's adequate for two would quietly over-fit at four.
MIN_GAMES_PER_INPUT = 20


class BlendModel:
    name = "blend"

    def __init__(self) -> None:
        self.clf: LogisticRegression | None = None
        self.input_keys: tuple[str, ...] | None = None

    def fit(self, probs: dict[str, np.ndarray], home_win_outcome: np.ndarray) -> "BlendModel":
        """`probs` maps a model name to its home-win probability array, e.g.
        `{"vegas": ..., "mc": ...}` or `{"vegas": ..., "mc": ..., "ml": ...}`.

        `home_win_outcome` must be 0/1 over *decisive* games only (ties
        excluded) — same convention as SpreadLogisticModel.
        """
        if not probs:
            raise ValueError("BlendModel needs at least one input model")

        n = len(home_win_outcome)
        minimum = MIN_GAMES_PER_INPUT * len(probs)
        if n < minimum:
            raise ValueError(
                f"need at least {minimum} decisive games to fit BlendModel on {len(probs)} inputs, got {n}"
            )

        self.input_keys = tuple(probs.keys())
        x = self._design_matrix(probs)
        y = np.asarray(home_win_outcome, dtype=int)
        self.clf = LogisticRegression()
        self.clf.fit(x, y)
        return self

    def predict(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        if self.clf is None or self.input_keys is None:
            raise RuntimeError("call fit() before predict()")
        if set(probs.keys()) != set(self.input_keys):
            raise ValueError(f"BlendModel was fit on inputs {self.input_keys}, got {tuple(probs.keys())}")
        return self.clf.predict_proba(self._design_matrix(probs))[:, 1]

    def _design_matrix(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Columns ordered by `self.input_keys`, never by dict iteration
        order — that ordering is what predict-time key checking protects."""
        assert self.input_keys is not None
        return np.column_stack([_logit_array(probs[key]) for key in self.input_keys])
