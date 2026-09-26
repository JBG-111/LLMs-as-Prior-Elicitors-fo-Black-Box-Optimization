"""Decode EV-charging controller candidates without estimating outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def decode_candidates(
    context_path: str | Path, candidates_path: str | Path
) -> list[dict[str, object]]:
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    variables = context["decision_variables"]
    with Path(candidates_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    decoded: list[dict[str, object]] = []
    for row in rows:
        parameters: dict[str, float] = {}
        for variable in variables:
            name = variable["name"]
            value = float(row[name])
            lower, upper = map(float, variable["bounds"])
            if not lower <= value <= upper:
                raise ValueError(
                    f"candidate {row['candidate_index']} has {name}={value} "
                    f"outside [{lower}, {upper}]"
                )
            parameters[name] = value
        decoded.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "controller_parameters": parameters,
            }
        )
    return decoded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=Path("input/problem_context.json"))
    parser.add_argument("--candidates", type=Path, default=Path("input/candidates.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(
        decode_candidates(args.context, args.candidates), ensure_ascii=False, indent=2
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
