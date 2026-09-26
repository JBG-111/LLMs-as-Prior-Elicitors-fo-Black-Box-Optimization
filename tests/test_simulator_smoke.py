import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SimulatorSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("wntr"), "WNTR is not installed")
    def test_net3_profile_f_returns_a_finite_objective(self):
        from benchmarks.net3_profile_f.benchmark import evaluate

        self.assertTrue(math.isfinite(evaluate([0.7] * 24)))

    @unittest.skipUnless(
        importlib.util.find_spec("sustaingym") and importlib.util.find_spec("acnportal"),
        "SustainGym and ACNPortal are not installed",
    )
    def test_ev_charging_examples_return_finite_objectives(self):
        for name in ("evcharging_jpl12d", "evcharging_caltech12d"):
            module = __import__(f"benchmarks.{name}.benchmark", fromlist=["evaluate"])
            x = json.loads(
                (ROOT / "benchmarks" / name / "examples" / "candidate_0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(math.isfinite(module.evaluate(x)), name)


if __name__ == "__main__":
    unittest.main()
