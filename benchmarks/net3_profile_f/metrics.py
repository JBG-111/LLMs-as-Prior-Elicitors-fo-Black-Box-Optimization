"""Pure objective metrics for Net3 Profile F."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def affine_normalize(value: float, minimum: float, maximum: float) -> float:
    if not all(math.isfinite(item) for item in (value, minimum, maximum)) or maximum <= minimum:
        raise ValueError("normalization values must be finite with maximum > minimum")
    return (float(value) - minimum) / (maximum - minimum)


def lower_tail_pressure_loss(pressure: np.ndarray, fraction: float) -> float:
    values = np.asarray(pressure, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or not 0.0 < fraction <= 1.0:
        raise ValueError("finite pressure and a tail fraction in (0, 1] are required")
    count = max(1, int(np.ceil(fraction * values.size)))
    return -float(np.partition(values, count - 1)[:count].mean())


def cyclic_speed_variation(speeds: Sequence[Sequence[float]], lower: float, upper: float) -> float:
    values = np.asarray(speeds, dtype=float)
    differences = values - np.roll(values, 1, axis=1)
    return float(np.mean((differences / (upper - lower)) ** 2))
