"""Bounded random-candidate optimization of preference-informed acquisitions."""
from __future__ import annotations

import numpy as np

from .acquisition import acquisition_scores
from .conditioned_gp import conditioned_moments


def propose(
    acquisition: str,
    prior,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: np.ndarray,
    seed: int,
    n_raw: int = 2048,
) -> np.ndarray:
    """Return the best acquisition point among reproducible uniform draws."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape (2, dimension)")
    if np.any(bounds[0] >= bounds[1]) or not np.all(np.isfinite(bounds)):
        raise ValueError("each lower bound must be finite and below its upper bound")
    if n_raw < 1:
        raise ValueError("n_raw must be positive")
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(bounds[0], bounds[1], size=(n_raw, bounds.shape[1]))
    mean, variance = conditioned_moments(prior, candidates, train_x, train_y)
    scores = acquisition_scores(acquisition, mean, variance, train_y)
    return candidates[int(np.argmax(scores))]
