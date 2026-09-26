"""Fit the sparse posterior from a populated Agent workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_preference_bo import (
    SparsePreferencePosterior,
    load_candidate_matrix,
    load_hard_preferences,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates")
    parser.add_argument("preferences")
    parser.add_argument("--n-inducing", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    candidate_ids, x, columns = load_candidate_matrix(args.candidates)
    data = load_hard_preferences(args.preferences, candidate_ids)
    posterior = SparsePreferencePosterior.fit(
        x,
        data.pairs,
        data.labels,
        n_inducing=min(args.n_inducing, len(x)),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "features": columns,
                "preference_counts": data.counts,
                "posterior": posterior.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
