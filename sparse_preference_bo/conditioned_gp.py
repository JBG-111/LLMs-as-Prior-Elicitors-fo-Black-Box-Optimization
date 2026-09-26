"""Condition a preference prior on direct black-box observations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _rbf(left: np.ndarray, right: np.ndarray, lengthscale: float) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    squared_distance = np.maximum(
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )
    return np.exp(-0.5 * squared_distance / lengthscale**2)


def _factor(matrix: np.ndarray):
    matrix = 0.5 * (matrix + matrix.T)
    eye = np.eye(len(matrix))
    for power in range(8):
        try:
            return cho_factor(
                matrix + 1e-8 * 10.0**power * eye,
                lower=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("could not stabilize observation covariance")


@dataclass(frozen=True)
class RBFZeroPrior:
    """Zero-mean RBF prior used for preference-free GP baselines."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    lengthscale: float

    @classmethod
    def fit(cls, support_x: np.ndarray, seed: int = 0) -> "RBFZeroPrior":
        x = np.asarray(support_x, dtype=float)
        if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
            raise ValueError("support_x must be a finite 2D array with at least two rows")
        feature_mean = x.mean(axis=0)
        feature_std = x.std(axis=0)
        feature_std[feature_std < 1e-12] = 1.0
        scaled = (x - feature_mean) / feature_std
        rng = np.random.default_rng(seed)
        sample = scaled[rng.choice(len(scaled), size=min(1024, len(scaled)), replace=False)]
        distances = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
        nonzero = distances[distances > 1e-12]
        lengthscale = max(float(np.median(nonzero)) if len(nonzero) else 1.0, 1e-6)
        return cls(feature_mean, feature_std, lengthscale)

    def _scaled(self, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != len(self.feature_mean):
            raise ValueError("x has the wrong feature dimension")
        return (points - self.feature_mean) / self.feature_std

    def mean(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(self._scaled(x)))

    def variance(self, x: np.ndarray) -> np.ndarray:
        return np.ones(len(self._scaled(x)))

    def mean_and_variance(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.mean(x), self.variance(x)

    def covariance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return _rbf(self._scaled(left), self._scaled(right), self.lengthscale)


def objective_to_utility(observed_y: np.ndarray) -> np.ndarray:
    """Convert minimization objectives to standardized higher-is-better utility."""
    values = -np.asarray(observed_y, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("observed_y must be finite")
    if len(values) == 0:
        return values
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def conditioned_moments(
    prior,
    candidate_x: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    observation_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return latent utility moments after direct objective observations."""
    candidate_x = np.atleast_2d(np.asarray(candidate_x, dtype=float))
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float).reshape(-1)
    if train_x.size == 0:
        train_x = np.empty((0, candidate_x.shape[1]), dtype=float)
    train_x = np.atleast_2d(train_x)
    if len(train_x) != len(train_y):
        raise ValueError("train_x and train_y must contain the same number of rows")
    if observation_noise <= 0.0:
        raise ValueError("observation_noise must be positive")
    if hasattr(prior, "mean_and_variance"):
        candidate_mean, candidate_variance = prior.mean_and_variance(candidate_x)
    else:
        candidate_mean = prior.mean(candidate_x)
        candidate_variance = prior.variance(candidate_x)
    if len(train_y) == 0:
        return candidate_mean, np.maximum(candidate_variance, 1e-10)
    utility = objective_to_utility(train_y)
    train_mean = prior.mean(train_x)
    cross = prior.covariance(candidate_x, train_x)
    covariance = prior.covariance(train_x, train_x)
    factor = _factor(covariance + observation_noise**2 * np.eye(len(train_x)))
    conditional_mean = candidate_mean + cross @ cho_solve(
        factor, utility - train_mean, check_finite=False
    )
    solved = cho_solve(factor, cross.T, check_finite=False)
    conditional_variance = candidate_variance - np.sum(cross * solved.T, axis=1)
    return conditional_mean, np.maximum(conditional_variance, 1e-10)
