"""Exact lookup evaluator for the 5,000-point Cologne3 SUMO pool."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


DIMENSION = 11
DATA = Path(__file__).resolve().parent / "data" / "pool_objectives.jsonl"


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(
                    (int(row["pool_index"]), row["x"], float(row["objective_value"]))
                )
    if len(rows) != 5000:
        raise ValueError(f"expected 5000 successful candidates, found {len(rows)}")
    indices = np.asarray([row[0] for row in rows], dtype=int)
    points = np.asarray([row[1] for row in rows], dtype=float)
    objectives = np.asarray([row[2] for row in rows], dtype=float)
    if points.shape != (5000, DIMENSION) or not np.all(np.isfinite(points)):
        raise ValueError("invalid candidate pool")
    return indices, points, objectives


_INDICES, _POINTS, _OBJECTIVES = _load()


def evaluate(x: Sequence[float]) -> float:
    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a numeric vector") from exc
    if values.shape != (DIMENSION,) or not np.all(np.isfinite(values)):
        raise ValueError(f"x must contain exactly {DIMENSION} finite values")
    matches = np.flatnonzero(
        np.all(np.isclose(_POINTS, values, rtol=1e-10, atol=1e-12), axis=1)
    )
    if matches.size != 1:
        raise ValueError("x must match exactly one point in the archived candidate pool")
    return float(_OBJECTIVES[int(matches[0])])


def evaluate_index(pool_index: int) -> float:
    if isinstance(pool_index, bool) or not isinstance(pool_index, (int, np.integer)):
        raise ValueError("pool_index must be an integer")
    matches = np.flatnonzero(_INDICES == int(pool_index))
    if matches.size != 1:
        raise ValueError("pool_index is not present in the archived candidate pool")
    return float(_OBJECTIVES[int(matches[0])])
