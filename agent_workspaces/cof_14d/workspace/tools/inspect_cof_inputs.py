"""Summarize public COF benchmark metadata without estimating objectives."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "input" / "simulation" / "public_benchmark_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "candidate_count": metadata["candidate_count"],
        "descriptor_count": metadata["descriptor_count"],
        "simulation_conditions": metadata["simulation_conditions"],
        "descriptors": [item["name"] for item in metadata["descriptor_schema"]],
        "information_boundary": metadata["information_boundary"],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
