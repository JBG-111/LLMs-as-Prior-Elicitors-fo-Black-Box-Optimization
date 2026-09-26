"""All-label inducing-point probit preference posterior.

This module deliberately omits Feature-CV, filtering, reliability weighting,
and flip-noise. Every supplied pairwise label has unit weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr


def _rbf(left: np.ndarray, right: np.ndarray, lengthscale: float) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    # The Gram-matrix identity avoids an n-by-m-by-d broadcast temporary.
    squared_distance = np.maximum(
        np.sum(left**2, axis=1)[:, None]
        + np.sum(right**2, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )
    return np.exp(-0.5 * squared_distance / (lengthscale**2))


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / np.sqrt(2.0 * np.pi)


def _stable_cholesky(matrix: np.ndarray, base_jitter: float = 1e-6) -> tuple[np.ndarray, float]:
    matrix = 0.5 * (matrix + matrix.T)
    eye = np.eye(matrix.shape[0])
    for power in range(8):
        jitter = base_jitter * 10.0**power
        try:
            return np.linalg.cholesky(matrix + jitter * eye), jitter
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Inducing kernel could not be stabilized by jitter.")


@dataclass(frozen=True)
class SparsePreferencePosterior:
    """Nyström inducing-point GP posterior over a latent preference utility.

    ``mean_weights`` and ``variances`` parameterize a whitened inducing
    posterior. ``variances`` is a diagonal Laplace approximation, which keeps
    memory linear in the number of inducing points after the kernel factor.
    """

    feature_mean: np.ndarray
    feature_std: np.ndarray
    inducing_x_scaled: np.ndarray
    inducing_cholesky: np.ndarray
    lengthscale: float
    mean_weights: np.ndarray
    variances: np.ndarray
    metadata: dict

    @property
    def dimension(self) -> int:
        return int(len(self.feature_mean))

    @classmethod
    def fit(
        cls,
        support_x: np.ndarray,
        pairs: np.ndarray,
        labels: np.ndarray,
        *,
        n_inducing: int = 5000,
        inducing_x: np.ndarray | None = None,
        tau: float = 1.0,
        epochs: int = 10,
        batch_size: int = 512,
        learning_rate: float = 0.03,
        seed: int = 0,
        lengthscale_sample: int = 2048,
    ) -> "SparsePreferencePosterior":
        """Fit a unit-weight, all-label sparse probit preference posterior.

        Each ``pairs[i] = [a, b]`` indexes ``support_x``. ``labels[i] = +1``
        means a is preferred to b, and ``-1`` reverses that order.
        """
        x = np.asarray(support_x, dtype=float)
        pair_array = np.asarray(pairs, dtype=int)
        label_array = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
            raise ValueError("support_x must be a finite 2D array with at least two rows.")
        if pair_array.ndim != 2 or pair_array.shape[1] != 2 or len(pair_array) == 0:
            raise ValueError("pairs must be a non-empty array with shape (n_pairs, 2).")
        if len(label_array) != len(pair_array) or not np.all(np.isin(label_array, (-1.0, 1.0))):
            raise ValueError("labels must contain exactly one +1 or -1 value per pair.")
        if np.any(pair_array < 0) or np.any(pair_array >= len(x)):
            raise ValueError("pairs contains an index outside support_x.")
        if np.any(pair_array[:, 0] == pair_array[:, 1]):
            raise ValueError("A preference pair must contain two different candidate indices.")
        if n_inducing < 2 or tau <= 0.0 or epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
            raise ValueError("n_inducing >= 2, tau > 0, epochs >= 1, batch_size >= 1, and learning_rate > 0 are required.")

        feature_mean = x.mean(axis=0)
        feature_std = x.std(axis=0)
        feature_std[feature_std < 1e-12] = 1.0
        x_scaled = (x - feature_mean) / feature_std
        rng = np.random.default_rng(seed)
        sample_count = min(len(x_scaled), max(2, int(lengthscale_sample)))
        sample = x_scaled[rng.choice(len(x_scaled), size=sample_count, replace=False)]
        pairwise = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
        nonzero = pairwise[pairwise > 1e-12]
        lengthscale = max(float(np.median(nonzero)) if len(nonzero) else 1.0, 1e-6)

        if inducing_x is None:
            count = min(int(n_inducing), len(x_scaled))
            inducing_indices = np.sort(rng.choice(len(x_scaled), size=count, replace=False))
            inducing_scaled = x_scaled[inducing_indices]
            inducing_source = "uniform_without_replacement"
        else:
            supplied = np.asarray(inducing_x, dtype=float)
            if supplied.ndim != 2 or supplied.shape[1] != x.shape[1] or len(supplied) < 2 or not np.all(np.isfinite(supplied)):
                raise ValueError("inducing_x must be a finite 2D array with matching dimension and at least two rows.")
            inducing_indices = np.array([], dtype=int)
            inducing_scaled = (supplied - feature_mean) / feature_std
            inducing_source = "user_supplied"

        inducing_kernel = _rbf(inducing_scaled, inducing_scaled, lengthscale)
        inducing_cholesky, kernel_jitter = _stable_cholesky(inducing_kernel)
        dimension = len(inducing_scaled)

        def features(points: np.ndarray) -> np.ndarray:
            points_scaled = (np.asarray(points, dtype=float) - feature_mean) / feature_std
            kernel_zx = _rbf(inducing_scaled, points_scaled, lengthscale)
            return solve_triangular(inducing_cholesky, kernel_zx, lower=True, check_finite=False).T

        # A support-feature cache turns repeated pair batches into indexed row
        # lookups. It is especially important when m is close to n_support.
        cache_bytes = len(x) * dimension * np.dtype(float).itemsize
        cache_limit_bytes = 512 * 1024**2
        support_features = features(x) if cache_bytes <= cache_limit_bytes else None

        # Stochastic MAP on the complete label set. The N/B likelihood factor
        # makes each mini-batch gradient an unbiased estimate of the full loss.
        weights = np.zeros(dimension)
        first_moment = np.zeros(dimension)
        second_moment = np.zeros(dimension)
        step = 0
        total = len(pair_array)
        for _ in range(int(epochs)):
            for start in range(0, total, int(batch_size)):
                order = rng.permutation(total) if start == 0 else order
                batch_indices = order[start : start + int(batch_size)]
                batch_pairs = pair_array[batch_indices]
                if support_features is None:
                    difference = features(x[batch_pairs[:, 0]]) - features(x[batch_pairs[:, 1]])
                else:
                    difference = support_features[batch_pairs[:, 0]] - support_features[batch_pairs[:, 1]]
                signed_margin = label_array[batch_indices] * (difference @ weights) / tau
                cdf = np.maximum(ndtr(signed_margin), 1e-12)
                inverse_mills = _normal_pdf(signed_margin) / cdf
                gradient = weights - (total / len(batch_indices)) * (
                    difference.T @ (label_array[batch_indices] * inverse_mills / tau)
                )
                step += 1
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient**2
                corrected_first = first_moment / (1.0 - 0.9**step)
                corrected_second = second_moment / (1.0 - 0.999**step)
                weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

        # Diagonal Laplace curvature uses every preference once more; it is not
        # sampled or filtered, so all labels determine the reported uncertainty.
        hessian_diagonal = np.ones(dimension)
        for start in range(0, total, int(batch_size)):
            batch_pairs = pair_array[start : start + int(batch_size)]
            if support_features is None:
                difference = features(x[batch_pairs[:, 0]]) - features(x[batch_pairs[:, 1]])
            else:
                difference = support_features[batch_pairs[:, 0]] - support_features[batch_pairs[:, 1]]
            signed_margin = label_array[start : start + int(batch_size)] * (difference @ weights) / tau
            cdf = np.maximum(ndtr(signed_margin), 1e-12)
            inverse_mills = _normal_pdf(signed_margin) / cdf
            curvature = inverse_mills * (signed_margin + inverse_mills) / (tau**2)
            hessian_diagonal += np.sum(curvature[:, None] * difference**2, axis=0)
        variances = 1.0 / np.maximum(hessian_diagonal, 1e-10)

        return cls(
            feature_mean=feature_mean,
            feature_std=feature_std,
            inducing_x_scaled=inducing_scaled,
            inducing_cholesky=inducing_cholesky,
            lengthscale=lengthscale,
            mean_weights=weights,
            variances=variances,
            metadata={
                "n_support": int(len(x)),
                "n_preferences_used": int(total),
                "n_inducing": int(dimension),
                "n_inducing_requested": int(n_inducing),
                "inducing_source": inducing_source,
                "inducing_indices": inducing_indices,
                "tau": float(tau),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "kernel_lengthscale": float(lengthscale),
                "kernel_jitter": float(kernel_jitter),
                "support_features_cached": support_features is not None,
                "support_feature_cache_bytes": int(cache_bytes),
            },
        )

    def _features(self, x: np.ndarray) -> np.ndarray:
        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.all(np.isfinite(points)):
            raise ValueError(f"x must be finite with shape (n, {self.dimension}).")
        scaled = (points - self.feature_mean) / self.feature_std
        kernel_zx = _rbf(self.inducing_x_scaled, scaled, self.lengthscale)
        return solve_triangular(self.inducing_cholesky, kernel_zx, lower=True, check_finite=False).T

    def mean(self, x: np.ndarray) -> np.ndarray:
        return self._features(x) @ self.mean_weights

    def variance(self, x: np.ndarray) -> np.ndarray:
        phi = self._features(x)
        base_residual = 1.0 - np.sum(phi**2, axis=1)
        uncertainty = np.sum(phi**2 * self.variances, axis=1)
        return np.maximum(base_residual + uncertainty, 1e-10)

    def mean_and_variance(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return marginal moments while computing inducing features only once."""
        phi = self._features(x)
        mean = phi @ self.mean_weights
        base_residual = 1.0 - np.sum(phi**2, axis=1)
        uncertainty = np.sum(phi**2 * self.variances, axis=1)
        return mean, np.maximum(base_residual + uncertainty, 1e-10)

    def covariance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if left_array.ndim == 1:
            left_array = left_array[None, :]
        if right_array.ndim == 1:
            right_array = right_array[None, :]
        left_scaled = (left_array - self.feature_mean) / self.feature_std
        right_scaled = (right_array - self.feature_mean) / self.feature_std
        phi_left = self._features(left_array)
        phi_right = self._features(right_array)
        residual = _rbf(left_scaled, right_scaled, self.lengthscale) - phi_left @ phi_right.T
        return residual + (phi_left * self.variances) @ phi_right.T

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self.mean(x)
        covariance = self.covariance(x, x)
        covariance = 0.5 * (covariance + covariance.T)
        covariance[np.diag_indices_from(covariance)] = np.maximum(np.diag(covariance), 1e-10)
        return mean, covariance
