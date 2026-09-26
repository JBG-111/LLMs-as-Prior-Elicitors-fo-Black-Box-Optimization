"""Neutral parser for static EPANET INP facts; it does not score candidates."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse(path):
    sections, current = {}, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.upper()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line.split())
    return sections

if __name__ == "__main__":
    data = parse(ROOT / "input/simulation/network.inp")
    print(json.dumps({
        "counts": {key.strip("[]").lower(): len(value) for key, value in data.items()},
        "pumps": data.get("[PUMPS]", []),
        "curves": data.get("[CURVES]", []),
        "patterns": data.get("[PATTERNS]", []),
        "junctions": data.get("[JUNCTIONS]", []),
    }, indent=2))
