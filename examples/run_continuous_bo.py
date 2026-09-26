"""Minimal bounded continuous proposal using a sparse preference prior."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_preference_bo import SparsePreferencePosterior
from sparse_preference_bo.continuous_bo import propose


def main() -> None:
    rng = np.random.default_rng(0)
    support = rng.uniform(-1.0, 1.0, size=(64, 2))
    latent_objective = np.sum((support - np.asarray([0.2, -0.3])) ** 2, axis=1)
    pairs = rng.integers(0, len(support), size=(100, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    labels = np.where(
        latent_objective[pairs[:, 0]] < latent_objective[pairs[:, 1]], 1.0, -1.0
    )
    prior = SparsePreferencePosterior.fit(
        support, pairs, labels, n_inducing=32, epochs=10, seed=0
    )
    train_x = support[[0, 1, 2]]
    train_y = latent_objective[[0, 1, 2]]
    candidate = propose(
        "ei",
        prior,
        train_x,
        train_y,
        bounds=np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
        seed=1,
    )
    print(candidate.tolist())


if __name__ == "__main__":
    main()
