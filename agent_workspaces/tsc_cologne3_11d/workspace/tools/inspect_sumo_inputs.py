"""Extract neutral, reproducible facts from static SUMO input files."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _context_file(context_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else context_path.parent / path


def _split_files(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _config_value(root: ET.Element, name: str) -> str | None:
    element = root.find(f".//{name}")
    return element.get("value") if element is not None else None


def _signal_programs(root: ET.Element, source: str) -> list[dict]:
    programs = []
    for logic in root.findall(".//tlLogic"):
        programs.append(
            {
                "id": logic.get("id"),
                "program_id": logic.get("programID"),
                "type": logic.get("type"),
                "offset": float(logic.get("offset", 0)),
                "source": source,
                "phases": [
                    {
                        key: float(value) if key in {"duration", "minDur", "maxDur"} else value
                        for key, value in phase.attrib.items()
                    }
                    for phase in logic.findall("phase")
                ],
            }
        )
    return programs


def inspect_sumo_inputs(context_path: str | Path) -> dict:
    """Return structural facts without calculating scores or preferences."""

    context_path = Path(context_path)
    context = json.loads(context_path.read_text(encoding="utf-8-sig"))
    simulator = context.get("simulator")
    if not isinstance(simulator, dict):
        raise ValueError("problem context is missing simulator")

    config_path = _context_file(context_path, simulator["config_file"])
    config_root = ET.parse(config_path).getroot()
    config_dir = config_path.parent

    configured_network = simulator.get("network_file")
    network_path = (
        _context_file(context_path, configured_network)
        if configured_network
        else config_dir / str(_config_value(config_root, "net-file"))
    )
    route_values = simulator.get("route_files") or _split_files(
        _config_value(config_root, "route-files")
    )
    additional_values = simulator.get("additional_files") or _split_files(
        _config_value(config_root, "additional-files")
    )
    route_paths = [_context_file(context_path, value) for value in route_values]
    additional_paths = [_context_file(context_path, value) for value in additional_values]

    network_root = ET.parse(network_path).getroot()
    edges = [edge for edge in network_root.findall("edge") if edge.get("id")]
    junctions = [item for item in network_root.findall("junction") if item.get("id")]
    traffic_light_ids = sorted(
        {
            str(item.get("id"))
            for item in junctions
            if item.get("type") == "traffic_light"
        }
        | {
            str(item.get("id"))
            for item in network_root.findall("tlLogic")
            if item.get("id")
        }
    )
    signals = _signal_programs(network_root, network_path.name)
    for path in additional_paths:
        signals.extend(_signal_programs(ET.parse(path).getroot(), path.name))

    demand = {
        "route_count": 0,
        "vehicle_count": 0,
        "trip_count": 0,
        "flow_count": 0,
        "vehicles_with_shared_route": 0,
        "vehicles_with_embedded_route": 0,
        "depart_time_range": None,
    }
    departure_times: list[float] = []
    for path in route_paths:
        root = ET.parse(path).getroot()
        demand["route_count"] += len(root.findall("route"))
        vehicles = root.findall("vehicle")
        trips = root.findall("trip")
        flows = root.findall("flow")
        demand["vehicle_count"] += len(vehicles)
        demand["trip_count"] += len(trips)
        demand["flow_count"] += len(flows)
        demand["vehicles_with_shared_route"] += sum(
            1 for item in vehicles if item.get("route")
        )
        demand["vehicles_with_embedded_route"] += sum(
            1 for item in vehicles if item.find("route") is not None
        )
        for item in [*vehicles, *trips]:
            try:
                departure_times.append(float(item.get("depart", "")))
            except ValueError:
                pass
    if departure_times:
        demand["depart_time_range"] = [min(departure_times), max(departure_times)]

    begin_value = _config_value(config_root, "begin")
    end_value = _config_value(config_root, "end")
    time_window = simulator.get("time_window_seconds", [0, None])
    begin = float(begin_value) if begin_value is not None else float(time_window[0])
    end = float(end_value) if end_value is not None else (
        float(time_window[1]) if time_window[1] is not None else None
    )

    return {
        "problem_id": context.get("problem_id"),
        "files": {
            "config": str(config_path),
            "network": str(network_path),
            "routes": [str(path) for path in route_paths],
            "additional": [str(path) for path in additional_paths],
        },
        "simulation_time": {"begin": begin, "end": end},
        "network": {
            "edge_count": len(edges),
            "lane_count": len(network_root.findall(".//lane")),
            "junction_count": len(junctions),
            "connection_count": len(network_root.findall("connection")),
            "traffic_light_ids": traffic_light_ids,
        },
        "traffic_signals": signals,
        "demand": demand,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="input/problem_context.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    summary = inspect_sumo_inputs(args.context)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
