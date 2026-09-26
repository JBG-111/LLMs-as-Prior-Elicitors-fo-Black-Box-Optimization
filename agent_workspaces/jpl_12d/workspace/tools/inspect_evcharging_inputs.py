"""Inspect static EV-charging inputs without evaluating candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def inspect_inputs(context_path: str | Path) -> dict[str, object]:
    context_path = Path(context_path).resolve()
    context = json.loads(context_path.read_text(encoding="utf-8"))
    files = context["available_inputs"]["files"]
    sessions = _read_csv(context_path.parent / files["sessions"])
    moer = _read_csv(context_path.parent / files["moer"])
    interface = json.loads(
        (context_path.parent / files["charging_interface"]).read_text(encoding="utf-8")
    )

    requested = [float(row["requested_kwh"]) for row in sessions]
    arrivals = [int(row["arrival_step"]) for row in sessions]
    departures = [int(row["departure_step"]) for row in sessions]
    moer_values = [float(row["moer"]) for row in moer]
    return {
        "problem_id": context.get("problem_id"),
        "session_count": len(sessions),
        "total_requested_kwh": sum(requested),
        "arrival_step_min_max": [min(arrivals), max(arrivals)] if arrivals else None,
        "departure_step_min_max": [min(departures), max(departures)] if departures else None,
        "moer_rows": len(moer),
        "moer_min_max": [min(moer_values), max(moer_values)] if moer_values else None,
        "evse_count": interface.get("evse_count"),
        "timestep_minutes": interface.get("timestep_minutes"),
        "maximum_pilot_a": interface.get("maximum_pilot_a"),
        "project_action_in_env": interface.get("project_action_in_env"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=Path("input/problem_context.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(inspect_inputs(args.context), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
