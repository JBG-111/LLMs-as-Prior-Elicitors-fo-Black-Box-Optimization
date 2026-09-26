"""Decode visible COF descriptors without scoring or ranking candidates."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    context = json.loads(
        (ROOT / "input" / "problem_context.json").read_text(encoding="utf-8")
    )
    path = ROOT / "input" / "candidates.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    decoded = []
    for row in rows:
        decoded.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "descriptors": {
                    item["name"]: float(row[item["name"]])
                    for item in context["decision_variables"]
                },
            }
        )
    print(json.dumps(decoded, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
