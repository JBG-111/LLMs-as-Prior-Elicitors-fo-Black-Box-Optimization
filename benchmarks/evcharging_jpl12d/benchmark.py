"""Standalone runtime copied into each packaged 12D EV-charging benchmark."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


E_MAX_KWH = 32 * 208 / 1000 * 5 / 60
TIME_BLOCK_ENDS = (96, 132, 180)
PARAMETER_BOUNDS = tuple(
    bound
    for _ in range(4)
    for bound in ((-4.0, 2.0), (0.0, 15.0), (0.0, 30.0))
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(package_root: Path, config: Mapping[str, Any]) -> None:
    for item in config.get("input_files", []):
        path = package_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少仿真输入文件: {path}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"输入文件大小不匹配: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"输入文件 SHA-256 不匹配: {path}")


def _validate_vector(x: Sequence[float]) -> np.ndarray:
    vector = np.asarray(x, dtype=np.float64)
    if vector.shape != (12,):
        raise ValueError("决策变量必须恰好包含 12 个数")
    if not np.all(np.isfinite(vector)):
        raise ValueError("所有决策变量必须为有限数值")
    lows = np.array([low for low, _ in PARAMETER_BOUNDS])
    highs = np.array([high for _, high in PARAMETER_BOUNDS])
    if np.any(vector < lows) or np.any(vector > highs):
        raise ValueError("决策变量超出 benchmark_config.json 规定的范围")
    return vector


def _controller_action(
    observation: Mapping[str, np.ndarray], timestep: int, x: np.ndarray
) -> np.ndarray:
    block = int(np.searchsorted(TIME_BLOCK_ENDS, timestep, side="right"))
    b, k_u, k_c = x.reshape(4, 3)[block]
    demands = np.asarray(observation["demands"], dtype=np.float64)
    departures = np.asarray(observation["est_departures"], dtype=np.float64)
    forecast = np.asarray(observation["forecasted_moer"], dtype=np.float64)
    required_rate = demands / (E_MAX_KWH * np.maximum(departures, 1.0))
    current_moer = float(np.asarray(observation["prev_moer"]).reshape(-1)[0])
    carbon_gap = current_moer - float(np.min(forecast[:12]))
    logits = np.clip(b + k_u * required_rate - k_c * carbon_gap, -60.0, 60.0)
    action = 1.0 / (1.0 + np.exp(-logits))
    return np.where(demands > 0.0, action, 0.0).astype(np.float32)


class _RawEvaluator:
    def __init__(self, site: str, package_root: Path):
        from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator

        self.site = site
        inputs = package_root / "data"
        old_directory = Path.cwd()
        old_moer_directory = GMMsTraceGenerator.MOER_DATA_DIR
        try:
            os.chdir(inputs)
            GMMsTraceGenerator.MOER_DATA_DIR = str((inputs / "moer").resolve())
            self.generator = GMMsTraceGenerator(site, "Summer 2021", seed=46)
        finally:
            os.chdir(old_directory)
            GMMsTraceGenerator.MOER_DATA_DIR = old_moer_directory
        self.env = EVChargingEnv(self.generator, project_action_in_env=False)
        self._closed = False

    def evaluate(self, x: np.ndarray) -> dict[str, float]:
        observation, _ = self.env.reset(seed=46)
        carbon_kg = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = _controller_action(observation, self.env.t, x)
            observation, _, terminated, truncated, _ = self.env.step(action)
            current = float(
                np.sum(self.env._simulator.charging_rates[:, self.env.t - 1])
            )
            delivered_step = current * self.env.A_PERS_TO_KWH
            carbon_kg += delivered_step * float(self.env.moer[self.env.t, 0])

        requested_kwh = 0.0
        delivered_kwh = 0.0
        for ev in self.env._evs:
            requested = float(ev.requested_energy)
            delivered = min(float(ev.energy_delivered), requested)
            requested_kwh += requested
            delivered_kwh += max(delivered, 0.0)
        return {
            "completion_rate": delivered_kwh / requested_kwh,
            "carbon_kg": carbon_kg,
            "requested_kwh": requested_kwh,
            "delivered_kwh": delivered_kwh,
        }

    def close(self) -> None:
        if not self._closed:
            self.env.close()
            self._closed = True


EvaluatorFactory = Callable[[str, Path], Any]


class EVChargingBenchmark:
    """Fixed-bound, deterministic 12D EV-charging black-box benchmark."""

    dimension = 12

    def __init__(
        self,
        package_root: Path | None = None,
        evaluator_factory: EvaluatorFactory = _RawEvaluator,
    ):
        self.package_root = (
            Path(__file__).resolve().parent
            if package_root is None else Path(package_root).resolve()
        )
        config_path = self.package_root / "definition.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(self.config["dimension"]) != self.dimension:
            raise ValueError("benchmark_config.json 的决策维度不是 12")
        configured_bounds = [
            (float(item["lower"]), float(item["upper"]))
            for item in self.config["parameters"]
        ]
        if tuple(configured_bounds) != PARAMETER_BOUNDS:
            raise ValueError("benchmark_config.json 的参数顺序或范围不正确")
        self.bounds = configured_bounds
        self.normalization_bounds = self.config["normalization_bounds"]
        c_min = float(self.normalization_bounds["completion_rate_min"])
        c_max = float(self.normalization_bounds["completion_rate_max"])
        m_min = float(self.normalization_bounds["carbon_kg_min"])
        m_max = float(self.normalization_bounds["carbon_kg_max"])
        if not c_max > c_min or not m_max > m_min:
            raise ValueError("benchmark_config.json 的归一化范围无效")
        _validate_inputs(self.package_root, self.config)
        self._evaluator = evaluator_factory(self.config["site"], self.package_root)

    def evaluate(self, x: Sequence[float]) -> dict[str, float]:
        vector = _validate_vector(x)
        raw = self._evaluator.evaluate(vector)
        bounds = self.normalization_bounds
        completion_normalized = (
            raw["completion_rate"] - float(bounds["completion_rate_min"])
        ) / (
            float(bounds["completion_rate_max"])
            - float(bounds["completion_rate_min"])
        )
        carbon_normalized = (
            raw["carbon_kg"] - float(bounds["carbon_kg_min"])
        ) / (
            float(bounds["carbon_kg_max"])
            - float(bounds["carbon_kg_min"])
        )
        return {
            **raw,
            "completion_rate_normalized": completion_normalized,
            "carbon_kg_normalized": carbon_normalized,
            "obj": carbon_normalized - completion_normalized,
        }

    def close(self) -> None:
        self._evaluator.close()

    def __enter__(self) -> "EVChargingBenchmark":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


_DEFAULT_BENCHMARK: EVChargingBenchmark | None = None


def _default_benchmark() -> EVChargingBenchmark:
    global _DEFAULT_BENCHMARK
    if _DEFAULT_BENCHMARK is None:
        _DEFAULT_BENCHMARK = EVChargingBenchmark(Path(__file__).resolve().parent)
    return _DEFAULT_BENCHMARK


def evaluate_detailed(x: Sequence[float]) -> dict[str, float]:
    result = _default_benchmark().evaluate(x)
    result["objective"] = float(result["obj"])
    return result


def evaluate(x: Sequence[float]) -> float:
    return float(evaluate_detailed(x)["objective"])


def close() -> None:
    global _DEFAULT_BENCHMARK
    if _DEFAULT_BENCHMARK is not None:
        _DEFAULT_BENCHMARK.close()
        _DEFAULT_BENCHMARK = None


atexit.register(close)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", nargs=12, type=float, required=True,
                        metavar="VALUE")
    args = parser.parse_args()
    with EVChargingBenchmark() as benchmark:
        print(json.dumps(benchmark.evaluate(args.x), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
