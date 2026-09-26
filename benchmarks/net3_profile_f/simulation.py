"""Self-contained one-day EPANET simulation for Net3 Profile F."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import wntr
from wntr.network.base import LinkStatus

from .metrics import lower_tail_pressure_loss


WATER_DENSITY = 1_000.0
GRAVITY = 9.80665


def simulate(speeds: np.ndarray, definition: dict[str, object]) -> dict[str, object]:
    data_path = Path(__file__).resolve().parent / "data" / "Net3.inp"
    network = wntr.network.WaterNetworkModel(data_path)
    for control_name in definition["removed_control_names"]:
        network.remove_control(control_name)
    network.options.time.duration = int(definition["duration_seconds"])
    network.options.time.report_timestep = int(definition["report_timestep_seconds"])
    network.options.time.pattern_timestep = int(definition["control_period_seconds"])
    network.options.quality.parameter = "NONE"

    demand_nodes: list[str] = []
    for node_name in network.junction_name_list:
        junction = network.get_node(node_name)
        if any(demand.base_value > 0.0 for demand in junction.demand_timeseries_list):
            demand_nodes.append(node_name)

    for pump_index, pump_name in enumerate(definition["pump_names"]):
        pump = network.get_link(pump_name)
        pump.initial_status = LinkStatus.Open
        pattern_name = f"__profile_f_speed_{pump_index}"
        network.add_pattern(pattern_name, speeds[pump_index].tolist())
        pump.speed_pattern_name = pattern_name

    with tempfile.TemporaryDirectory(prefix="net3-profile-f-") as directory:
        results = wntr.sim.EpanetSimulator(network).run_sim(
            file_prefix=str(Path(directory) / "epanet")
        )
        pressure = results.node["pressure"].loc[:, demand_nodes].iloc[1:].to_numpy(dtype=float)
        energy = _energy_cost(results, network, definition)
        tank_levels = {
            name: float(results.node["pressure"].iloc[-1][name])
            for name in network.tank_name_list
        }
    return {
        "energy_cost": energy,
        "tail_pressure_loss": lower_tail_pressure_loss(pressure, float(definition["tail_fraction"])),
        "minimum_pressure": float(np.min(pressure)),
        "final_tank_levels": tank_levels,
    }


def _energy_cost(results, network, definition: dict[str, object]) -> float:
    times = results.link["flowrate"].index.to_numpy(dtype=float)
    intervals = np.diff(times)
    prices = np.asarray(definition["tariff_per_kwh"], dtype=float)[
        (times[:-1] // int(definition["control_period_seconds"])).astype(int)
    ]
    pump_names = definition["pump_names"]
    power = np.empty((len(times), len(pump_names)), dtype=float)
    for pump_index, pump_name in enumerate(pump_names):
        pump = network.get_link(pump_name)
        flow = results.link["flowrate"][pump_name].to_numpy(dtype=float)
        head_gain = (
            results.node["head"][pump.end_node_name].to_numpy(dtype=float)
            - results.node["head"][pump.start_node_name].to_numpy(dtype=float)
        )
        power[:, pump_index] = WATER_DENSITY * GRAVITY * np.maximum(flow, 0.0) * np.maximum(head_gain, 0.0)
    energy_kwh = power[:-1] * intervals[:, None] / 3_600_000.0
    return float(np.sum(energy_kwh * prices[:, None]))
