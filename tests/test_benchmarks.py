import importlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BENCHMARKS = {
    "cof14d": 14,
    "net3_profile_f": 24,
    "evcharging_jpl12d": 12,
    "evcharging_caltech12d": 12,
    "tsc_cologne3_11d": 11,
    "tsc_cologne8_25d": 25,
}


class BenchmarkTests(unittest.TestCase):
    def test_all_six_benchmark_packages_and_definitions_exist(self):
        for name, dimension in BENCHMARKS.items():
            package = ROOT / "benchmarks" / name
            self.assertTrue((package / "benchmark.py").is_file(), name)
            definition = json.loads(
                (package / "definition.json").read_text(encoding="utf-8")
            )
            if name == "net3_profile_f":
                observed_dimension = 2 * 12
            else:
                observed_dimension = int(definition["dimension"])
            self.assertEqual(observed_dimension, dimension, name)

    def test_lookup_benchmarks_return_packaged_objectives(self):
        cases = {
            "tsc_cologne3_11d": "candidate_0.json",
            "tsc_cologne8_25d": "candidate_0.json",
        }
        for name, filename in cases.items():
            module = importlib.import_module(f"benchmarks.{name}.benchmark")
            x = json.loads(
                (ROOT / "benchmarks" / name / "examples" / filename).read_text(
                    encoding="utf-8"
                )
            )
            value = float(module.evaluate(x))
            self.assertTrue(math.isfinite(value), name)
            self.assertEqual(value, float(module.evaluate_index(0)), name)

    def test_cof_benchmark_explains_missing_user_supplied_data(self):
        from evaluate import evaluate

        x_file = ROOT / "benchmarks" / "cof14d" / "examples" / "candidate_0.json"
        x = json.loads(x_file.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(FileNotFoundError, "not distributed"):
            evaluate("cof14d", x)

    def test_unified_dispatcher_and_cli(self):
        from evaluate import BENCHMARKS as registry, evaluate

        self.assertEqual(set(registry), set(BENCHMARKS))
        x_file = ROOT / "benchmarks" / "tsc_cologne3_11d" / "examples" / "candidate_0.json"
        x = json.loads(x_file.read_text(encoding="utf-8"))
        expected = evaluate("tsc_cologne3_11d", x)
        completed = subprocess.run(
            [sys.executable, str(ROOT / "evaluate.py"), "tsc_cologne3_11d", "--x-file", str(x_file)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertAlmostEqual(float(completed.stdout.strip()), expected)

    def test_continuous_bounds_are_documented(self):
        net3 = json.loads(
            (ROOT / "benchmarks/net3_profile_f/definition.json").read_text(encoding="utf-8")
        )
        self.assertEqual(net3["speed_lower_bound"], 0.7)
        self.assertEqual(net3["speed_upper_bound"], 1.2)
        for name in ("evcharging_jpl12d", "evcharging_caltech12d"):
            definition = json.loads(
                (ROOT / "benchmarks" / name / "definition.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(definition["parameters"]), 12)
            self.assertTrue(all(item["lower"] < item["upper"] for item in definition["parameters"]))


if __name__ == "__main__":
    unittest.main()
