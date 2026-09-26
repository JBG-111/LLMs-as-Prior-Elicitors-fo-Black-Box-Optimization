"""Minimal synthetic finite-pool run using a sparse preference prior."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_preference_bo import SparsePreferencePosterior
from sparse_preference_bo.finite_pool_bo import run_finite_pool_bo


def main() -> None:
    x = np.linspace(-2.0, 2.0, 41)[:, None]
    objective = (x[:, 0] - 0.35) ** 2
    pairs = np.asarray([[i + 1, i] for i in range(20)], dtype=int)
    labels = np.where(objective[pairs[:, 0]] < objective[pairs[:, 1]], 1.0, -1.0)
    prior = SparsePreferencePosterior.fit(
        x, pairs, labels, n_inducing=20, epochs=10, seed=0
    )
    result = run_finite_pool_bo(
        prior,
        x,
        objective,
        initial_indices=[0, len(x) - 1],
        n_steps=5,
        acquisition="lcb",
        seed=0,
    )
    print(result["selected_indices"])


if __name__ == "__main__":
    main()
