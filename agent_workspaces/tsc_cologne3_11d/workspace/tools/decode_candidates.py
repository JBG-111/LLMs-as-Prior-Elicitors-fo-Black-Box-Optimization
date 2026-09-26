"""Map candidate table values to simulator objects described by problem context."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def decode_candidates(
    context_path: str | Path, candidates_path: str | Path
) -> list[dict]:
    context = json.loads(Path(context_path).read_text(encoding="utf-8-sig"))
    variables = context.get("decision_variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("problem context is missing decision_variables")

    with Path(candidates_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    plans = []
    for row in rows:
        decoded = []
        for variable in variables:
            name = variable["name"]
            if name not in row:
                raise ValueError(f"candidates.csv is missing decision variable: {name}")
            value = float(row[name])
            lower, upper = map(float, variable["bounds"])
            decoded.append(
                {
                    "name": name,
                    "value": value,
                    "type": variable.get("type"),
                    "unit": variable.get("unit"),
                    "bounds": [lower, upper],
                    "within_bounds": lower <= value <= upper,
                    "simulator_mapping": variable["simulator_mapping"],
                }
            )
        plans.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "candidate_id": row.get("candidate_id"),
                "variables": decoded,
            }
        )
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="input/problem_context.json")
    parser.add_argument("--candidates", default="input/candidates.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    plans = decode_candidates(args.context, args.candidates)
    text = json.dumps(plans, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
