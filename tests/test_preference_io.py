import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PreferenceIOTests(unittest.TestCase):
    def test_loads_candidates_and_converts_a_b_to_signed_hard_labels(self):
        from sparse_preference_bo.preference_io import (
            load_candidate_matrix,
            load_hard_preferences,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.csv"
            candidates.write_text(
                "candidate_index,x0,x1\n10,0.0,1.0\n20,1.0,0.0\n30,2.0,2.0\n",
                encoding="utf-8",
            )
            preferences = root / "preferences.jsonl"
            rows = [
                {"pair_id": "p0", "a_candidate_index": 10, "b_candidate_index": 20, "preference": "A"},
                {"pair_id": "p1", "a_candidate_index": 20, "b_candidate_index": 30, "preference": "B"},
                {"pair_id": "p2", "a_candidate_index": 10, "b_candidate_index": 30, "preference": "tie"},
                {"pair_id": "p3", "a_candidate_index": 30, "b_candidate_index": 10, "preference": "insufficient_information"},
            ]
            preferences.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            indices, x, columns = load_candidate_matrix(candidates)
            data = load_hard_preferences(preferences, indices)

        np.testing.assert_array_equal(indices, [10, 20, 30])
        np.testing.assert_allclose(x, [[0, 1], [1, 0], [2, 2]])
        self.assertEqual(columns, ["x0", "x1"])
        np.testing.assert_array_equal(data.pairs, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(data.labels, [1.0, -1.0])
        self.assertEqual(data.pair_ids, ["p0", "p1"])
        self.assertEqual(data.counts, {"A": 1, "B": 1, "tie": 1, "insufficient_information": 1})

    def test_rejects_unknown_candidate_reference(self):
        from sparse_preference_bo.preference_io import load_hard_preferences

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pair_id", "a_candidate_index", "b_candidate_index", "preference"],
                )
                writer.writeheader()
                writer.writerow(
                    {"pair_id": "p0", "a_candidate_index": 1, "b_candidate_index": 99, "preference": "A"}
                )
            with self.assertRaisesRegex(ValueError, "unknown candidate"):
                load_hard_preferences(path, np.asarray([1, 2]))


if __name__ == "__main__":
    unittest.main()
