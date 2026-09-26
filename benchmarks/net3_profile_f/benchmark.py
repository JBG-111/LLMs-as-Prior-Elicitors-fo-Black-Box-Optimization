"""Portable 24D Net3 Profile F minimization objective."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import affine_normalize, cyclic_speed_variation
from .simulation import simulate


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def evaluate(x: Sequence[float]) -> float:
    return float(evaluate_detailed(x)["objective"])


def evaluate_detailed(x: Sequence[float]) -> dict[str, object]:
    definition = json.loads((ROOT / "definition.json").read_text(encoding="utf-8"))
    normalization = json.loads((DATA / "normalization_F.json").read_text(encoding="utf-8"))
    try:
        values = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a numeric vector") from exc
    lower = float(definition["speed_lower_bound"])
    upper = float(definition["speed_upper_bound"])
    if values.shape != (24,) or not np.all(np.isfinite(values)):
        raise ValueError("x must contain exactly 24 finite values")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("x values must lie in [0.7, 1.2]")
    speeds = values.reshape(2, 12)
    try:
        raw = simulate(speeds, definition)
        status = "ok"
    except Exception as exc:
        bounds = normalization["bounds"]
        raw = {
            "energy_cost": 2 * bounds["energy"]["maximum"] - bounds["energy"]["minimum"],
            "tail_pressure_loss": 2 * bounds["tail"]["maximum"] - bounds["tail"]["minimum"],
            "minimum_pressure": math.nan,
            "final_tank_levels": {},
        }
        status = f"failed:{type(exc).__name__}"
    bounds = normalization["bounds"]
    normalized_energy = affine_normalize(
        float(raw["energy_cost"]), float(bounds["energy"]["minimum"]), float(bounds["energy"]["maximum"])
    )
    normalized_tail = affine_normalize(
        float(raw["tail_pressure_loss"]), float(bounds["tail"]["minimum"]), float(bounds["tail"]["maximum"])
    )
    variation = cyclic_speed_variation(speeds, lower, upper)
    objective = 0.75 * normalized_energy + 0.05 * normalized_tail + 0.20 * variation
    return {
        "objective": float(objective),
        "raw_energy": float(raw["energy_cost"]),
        "raw_tail_loss": float(raw["tail_pressure_loss"]),
        "normalized_energy": float(normalized_energy),
        "normalized_tail_loss": float(normalized_tail),
        "variation": float(variation),
        "simulation_status": status,
        "minimum_pressure": float(raw["minimum_pressure"]),
        "final_tank_levels": raw["final_tank_levels"],
    }
