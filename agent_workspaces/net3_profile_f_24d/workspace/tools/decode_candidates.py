"""Decode candidate rows into pump/time assignments without scoring them."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
context = json.loads((ROOT / "input/problem_context.json").read_text(encoding="utf-8"))
with (ROOT / "input/candidates.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
decoded = []
for row in rows:
    decoded.append({
        "candidate_index": int(row["candidate_index"]),
        "assignments": [
            {**variable, "value": float(row[variable["name"]])}
            for variable in context["decision_variables"]
        ],
    })
print(json.dumps(decoded, ensure_ascii=False, indent=2))
