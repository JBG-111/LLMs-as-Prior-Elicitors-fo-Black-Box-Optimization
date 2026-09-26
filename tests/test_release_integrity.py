import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_required_documentation_and_examples_exist(self):
        for relative in (
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "requirements.txt",
            "examples/fit_preference_prior.py",
            "examples/run_finite_pool_bo.py",
            "examples/run_continuous_bo.py",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_release_contains_no_runtime_or_experiment_artifacts(self):
        forbidden_names = {
            "preferences.jsonl",
            "preference_pairs.csv",
            "run.json",
            "evaluation.json",
            "events.jsonl",
            "stderr.txt",
        }
        # Python may create bytecode caches while this very test is running;
        # cache removal is checked again immediately before packaging.
        forbidden_suffixes = {".png", ".pdf", ".svg"}
        for path in ROOT.rglob("*"):
            self.assertNotIn(path.name, forbidden_names, path)
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), forbidden_suffixes, path)

    def test_readme_explains_information_boundary_and_both_bo_modes(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for phrase in (
            "information boundary",
            "agent_workspaces",
            "finite-pool",
            "continuous",
            "sparse preference posterior",
            "net3 profile f",
        ):
            self.assertIn(phrase, text)

    def test_cof_arrays_are_not_redistributed(self):
        data_dir = ROOT / "benchmarks" / "cof14d" / "data"
        self.assertFalse((data_dir / "candidates.npy").exists())
        self.assertFalse((data_dir / "objective_values.npy").exists())
        notice = (data_dir / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("not distributed", notice)
        self.assertIn("candidates.npy", notice)
        self.assertIn("objective_values.npy", notice)
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("cof arrays are not included", root_readme)

    def test_self_contained_examples_run_from_release_root(self):
        for name in ("run_finite_pool_bo.py", "run_continuous_bo.py"):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "examples" / name)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, (name, completed.stderr))
            self.assertTrue(completed.stdout.strip(), name)


if __name__ == "__main__":
    unittest.main()
