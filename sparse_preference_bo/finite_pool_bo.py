"""Generic Bayesian optimization over a finite candidate pool."""
from __future__ import annotations

import numpy as np

from .acquisition import acquisition_scores
from .conditioned_gp import conditioned_moments


def run_finite_pool_bo(
    prior,
    x: np.ndarray,
    objective: np.ndarray,
    initial_indices,
    n_steps: int,
    acquisition: str,
    seed: int,
) -> dict[str, object]:
    """Select unobserved pool points and reveal their packaged objectives."""
    x = np.asarray(x, dtype=float)
    objective = np.asarray(objective, dtype=float).reshape(-1)
    selected = [int(index) for index in initial_indices]
    if x.ndim != 2 or len(x) != len(objective):
        raise ValueError("x and objective must describe the same finite pool")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("initial_indices must be non-empty and distinct")
    if min(selected) < 0 or max(selected) >= len(x):
        raise ValueError("initial_indices contains an invalid pool index")
    if n_steps < 0 or len(selected) + n_steps > len(x):
        raise ValueError("n_steps exceeds the unobserved pool")
    rng = np.random.default_rng(seed)
    best_objective: list[float] = []
    for _ in range(n_steps):
        observed = np.asarray(selected, dtype=int)
        available = np.setdiff1d(np.arange(len(x)), observed, assume_unique=False)
        mean, variance = conditioned_moments(
            prior, x[available], x[observed], objective[observed]
        )
        scores = acquisition_scores(acquisition, mean, variance, objective[observed])
        ties = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-14))
        selected.append(int(available[int(rng.choice(ties))]))
        best_objective.append(float(np.min(objective[np.asarray(selected, dtype=int)])))
    return {"selected_indices": selected, "best_objective": np.asarray(best_objective)}
