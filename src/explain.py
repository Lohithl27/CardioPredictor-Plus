"""
explain.py
Per-prediction explanations. This replaces the earlier behaviour where the
API returned the same global "top 5 features" list for every patient.

For each incoming reading we compute how much EACH feature pushed THIS
patient's risk up or down relative to a typical (background) patient:

  * If the `shap` package is installed, SHAP's model-agnostic Permutation
    explainer is run on the calibrated model's probability output.
  * Otherwise a transparent fallback is used: Shapley-style sampling over
    random feature orderings (the same idea SHAP's permutation explainer
    uses), implemented here with numpy/pandas only.

Either way the result is a list of {feature, value, contribution, direction}
sorted by absolute contribution, where `contribution` is in probability
points (e.g. +0.031 means this feature raised the predicted risk by 3.1
percentage points versus the background average).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BACKGROUND = 100
N_PERMUTATIONS = 30  # fallback sampler only


def make_background(X_train: pd.DataFrame, n: int = N_BACKGROUND, seed: int = 42) -> pd.DataFrame:
    """Small, reproducible sample of training rows used as the reference."""
    n = min(n, len(X_train))
    return X_train.sample(n=n, random_state=seed).reset_index(drop=True)


def _predict_fn(model, columns):
    def f(data):
        df = pd.DataFrame(np.asarray(data), columns=columns)
        return model.predict_proba(df)[:, 1]
    return f


def _shapley_sampling(model, row: pd.DataFrame, background: pd.DataFrame,
                      n_permutations: int = N_PERMUTATIONS, seed: int = 0) -> np.ndarray:
    """Monte-Carlo Shapley values for one row against a background set.

    For each random ordering of features, features are switched from a
    background row to the patient's value one at a time; each feature's
    contribution is the change in predicted probability when it is switched.
    """
    rng = np.random.default_rng(seed)
    cols = list(row.columns)
    x = row.iloc[0].to_numpy(dtype=float)
    B = background.to_numpy(dtype=float)
    d = len(cols)
    phi = np.zeros(d)

    for _ in range(n_permutations):
        order = rng.permutation(d)
        current = B[rng.integers(0, len(B))].copy()
        # Build the whole chain of d+1 rows for this ordering, then score
        # them in ONE model call (much faster than d+1 separate calls).
        chain = np.empty((d + 1, d))
        chain[0] = current
        for step, j in enumerate(order, start=1):
            current[j] = x[j]
            chain[step] = current
        probs = model.predict_proba(pd.DataFrame(chain, columns=cols))[:, 1]
        for step, j in enumerate(order, start=1):
            phi[j] += probs[step] - probs[step - 1]
    return phi / n_permutations


class Explainer:
    """Wraps whichever backend is available behind one `explain()` call."""

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.background = background
        self.columns = list(background.columns)
        self.method = "shapley_sampling"
        self._shap_explainer = None
        try:
            import shap  # noqa: WPS433
            masker = shap.maskers.Independent(background.to_numpy(dtype=float), max_samples=N_BACKGROUND)
            self._shap_explainer = shap.Explainer(
                _predict_fn(model, self.columns), masker, algorithm="permutation",
            )
            self.method = "shap"
        except Exception:
            self._shap_explainer = None

    def explain(self, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
        row = row[self.columns]
        if self._shap_explainer is not None:
            values = self._shap_explainer(row.to_numpy(dtype=float), max_evals=2 * len(self.columns) + 1)
            phi = np.asarray(values.values)[0]
        else:
            phi = _shapley_sampling(self.model, row, self.background)

        items = []
        for col, contrib in zip(self.columns, phi):
            items.append({
                "feature": col,
                "value": float(row.iloc[0][col]),
                "contribution": round(float(contrib), 4),
                "direction": "increases risk" if contrib > 0 else "decreases risk",
            })
        items.sort(key=lambda it: -abs(it["contribution"]))
        return items[:top_k]
