"""Unified scalar interface for the six minimization benchmarks."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Sequence


BENCHMARKS = {
    "cof14d": "benchmarks.cof14d.benchmark",
    "net3_profile_f": "benchmarks.net3_profile_f.benchmark",
    "evcharging_caltech12d": "benchmarks.evcharging_caltech12d.benchmark",
    "evcharging_jpl12d": "benchmarks.evcharging_jpl12d.benchmark",
    "tsc_cologne3_11d": "benchmarks.tsc_cologne3_11d.benchmark",
    "tsc_cologne8_25d": "benchmarks.tsc_cologne8_25d.benchmark",
}


def evaluate(benchmark_name: str, x: Sequence[float]) -> float:
    """Evaluate one decision vector and return a finite minimization objective."""
    try:
        module_name = BENCHMARKS[benchmark_name]
    except KeyError as exc:
        choices = ", ".join(BENCHMARKS)
        raise ValueError(
            f"unknown benchmark {benchmark_name!r}; choose from {choices}"
        ) from exc
    value = float(import_module(module_name).evaluate(x))
    if not math.isfinite(value):
        raise ValueError(f"{benchmark_name} returned a non-finite objective")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=tuple(BENCHMARKS))
    parser.add_argument("--x-file", required=True, type=Path)
    args = parser.parse_args()
    try:
        x = json.loads(args.x_file.read_text(encoding="utf-8"))
        if not isinstance(x, list):
            raise ValueError("x JSON must be a list")
        value = evaluate(args.benchmark, x)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(repr(value))


if __name__ == "__main__":
    main()
