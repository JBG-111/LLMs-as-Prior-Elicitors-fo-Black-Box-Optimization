"""Exact lookup evaluator for the user-supplied 608-point COF 14D pool."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def _load_data() -> tuple[np.ndarray, np.ndarray]:
    candidates_path = DATA / "candidates.npy"
    objectives_path = DATA / "objective_values.npy"
    missing = [path.name for path in (candidates_path, objectives_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "COF arrays are not distributed with this public release. "
            "Provide benchmarks/cof14d/data/candidates.npy and "
            "benchmarks/cof14d/data/objective_values.npy from an authorized source; "
            "see benchmarks/cof14d/data/README.md."
        )
    return np.load(candidates_path), np.load(objectives_path)


def evaluate(x: Sequence[float]) -> float:
    candidates, objectives = _load_data()
    try:
        values = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a numeric vector") from exc
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise ValueError("x must contain exactly 14 finite values")
    matches = np.flatnonzero(
        np.all(np.isclose(candidates, values, rtol=1e-10, atol=1e-12), axis=1)
    )
    if matches.size != 1:
        raise ValueError("x must match exactly one point in the published candidate pool")
    return float(objectives[int(matches[0])])


def evaluate_index(candidate_index: int) -> float:
    _, objectives = _load_data()
    if isinstance(candidate_index, bool) or not isinstance(
        candidate_index, (int, np.integer)
    ):
        raise ValueError("candidate_index must be an integer")
    index = int(candidate_index)
    if not 0 <= index < len(objectives):
        raise ValueError(f"candidate_index must be between 0 and {len(objectives) - 1}")
    return float(objectives[index])
