"""Read candidate tables and convert Agent outputs to hard preference arrays."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np


PREFERENCES = ("A", "B", "tie", "insufficient_information")


@dataclass(frozen=True)
class HardPreferenceData:
    """Pair rows usable by the sparse probit model plus output diagnostics."""

    pairs: np.ndarray
    labels: np.ndarray
    pair_ids: list[str]
    counts: dict[str, int]


def load_candidate_matrix(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a candidate CSV while preserving its explicit candidate indices."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not reader.fieldnames:
            raise ValueError("candidate CSV is missing a header")
        if reader.fieldnames[0] != "candidate_index" or len(reader.fieldnames) < 2:
            raise ValueError("candidate CSV must start with candidate_index and include features")
        feature_columns = reader.fieldnames[1:]
        rows = list(reader)
    if not rows:
        raise ValueError("candidate CSV contains no candidates")
    try:
        indices = np.asarray([int(row["candidate_index"]) for row in rows], dtype=int)
        x = np.asarray(
            [[float(row[column]) for column in feature_columns] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("candidate CSV contains an invalid index or feature value") from exc
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("candidate_index values must be unique")
    if not np.all(np.isfinite(x)):
        raise ValueError("candidate features must be finite")
    return indices, x, list(feature_columns)


def _read_records(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                records.append(record)
        return records
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_hard_preferences(
    path: str | Path, candidate_indices: np.ndarray
) -> HardPreferenceData:
    """Map A/B outputs to +1/-1 and omit abstentions from model fitting.

    ``+1`` means the first candidate is preferred; ``-1`` means the second.
    Ties and insufficient-information records are counted but excluded because
    the canonical sparse posterior uses a binary probit likelihood.
    """
    records = _read_records(Path(path))
    row_by_index = {int(value): row for row, value in enumerate(candidate_indices)}
    counts = {preference: 0 for preference in PREFERENCES}
    pairs: list[list[int]] = []
    labels: list[float] = []
    pair_ids: list[str] = []
    seen_ids: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        try:
            pair_id = str(record["pair_id"])
            a_index = int(record["a_candidate_index"])
            b_index = int(record["b_candidate_index"])
            preference = str(record["preference"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"preference row {row_number} is missing required fields") from exc
        if pair_id in seen_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen_ids.add(pair_id)
        if preference not in counts:
            raise ValueError(f"invalid preference {preference!r}")
        if a_index not in row_by_index or b_index not in row_by_index:
            raise ValueError(f"{pair_id}: unknown candidate index")
        if a_index == b_index:
            raise ValueError(f"{pair_id}: a pair must reference two different candidates")
        counts[preference] += 1
        if preference in ("A", "B"):
            pairs.append([row_by_index[a_index], row_by_index[b_index]])
            labels.append(1.0 if preference == "A" else -1.0)
            pair_ids.append(pair_id)
    if not pairs:
        raise ValueError("no hard A/B preferences are available for model fitting")
    return HardPreferenceData(
        pairs=np.asarray(pairs, dtype=int),
        labels=np.asarray(labels, dtype=float),
        pair_ids=pair_ids,
        counts=counts,
    )
