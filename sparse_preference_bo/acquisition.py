"""Utility-space EI and confidence-bound acquisitions for minimization BO."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .conditioned_gp import objective_to_utility


def acquisition_scores(
    name: str,
    mean: np.ndarray,
    variance: np.ndarray,
    observed_y: np.ndarray,
    *,
    xi: float = 0.01,
    delta: float = 0.1,
) -> np.ndarray:
    """Return scores to maximize.

    The public name ``lcb`` follows the experiments' objective-space naming.
    Internally the model represents higher-is-better utility ``-objective``, so
    the equivalent score is an upper confidence bound on utility.
    """
    if name not in {"ei", "lcb"}:
        raise ValueError("name must be 'ei' or 'lcb'")
    mean = np.asarray(mean, dtype=float).reshape(-1)
    variance = np.asarray(variance, dtype=float).reshape(-1)
    if mean.shape != variance.shape or not np.all(np.isfinite(mean)):
        raise ValueError("mean and variance must be finite vectors of equal length")
    if np.any(variance < 0.0):
        raise ValueError("variance must be nonnegative")
    std = np.sqrt(np.maximum(variance, 1e-10))
    observed_y = np.asarray(observed_y, dtype=float).reshape(-1)
    if name == "lcb":
        count = max(2, len(observed_y))
        beta = 2.0 * math.log(count**2 * math.pi**2 / (6.0 * delta))
        return mean + math.sqrt(beta) * std
    utility = objective_to_utility(observed_y)
    best = float(np.max(utility)) if len(utility) else 0.0
    improvement = mean - best - xi
    z = improvement / std
    return improvement * ndtr(z) + std * np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
