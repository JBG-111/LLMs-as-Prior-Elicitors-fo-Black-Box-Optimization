import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent_workspaces"
SCENARIOS = {
    "cof_14d",
    "net3_profile_f_24d",
    "jpl_12d",
    "caltech_12d",
    "tsc_cologne3_11d",
    "tsc_cologne8_25d",
}


class AgentWorkspaceTests(unittest.TestCase):
    def test_six_workspaces_use_identical_generic_agent_instructions(self):
        self.assertEqual(
            {path.name for path in AGENT_ROOT.iterdir() if path.is_dir()},
            SCENARIOS,
        )
        paths = [AGENT_ROOT / name / "workspace" / "AGENTS.md" for name in SCENARIOS]
        hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(len(hashes), 1)
        text = paths[0].read_text(encoding="utf-8").lower()
        for term in ("cologne", "caltech", "jpl", "wntr", "cof", "charging"):
            self.assertNotIn(term, text)

    def test_agent_instructions_match_research_annotation_protocol(self):
        path = AGENT_ROOT / "caltech_12d" / "workspace" / "AGENTS.md"
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        self.assertTrue(text.startswith("ROLE\n"))
        for required in (
            "You are an autonomous research-oriented preference annotation agent",
            "EVIDENCE TYPES",
            "input/problem_context.json",
            "Analyze all candidates in a batch before comparing pairs.",
            "REQUIRED ANALYSIS ARTIFACTS",
            "output/preferences.jsonl",
            "insufficient_information require label=null and weight=0",
            "Do not use the Internet or launch the black-box simulator.",
        ):
            self.assertIn(" ".join(required.split()), normalized)

    def test_candidate_and_pair_interfaces_are_header_only(self):
        for name in SCENARIOS:
            input_dir = AGENT_ROOT / name / "workspace" / "input"
            with (input_dir / "candidates.template.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                candidate_rows = list(csv.reader(handle))
            with (input_dir / "pairs.template.tsv").open(
                encoding="utf-8", newline=""
            ) as handle:
                pair_rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(len(candidate_rows), 1, name)
            self.assertEqual(candidate_rows[0][0], "candidate_index", name)
            self.assertEqual(
                pair_rows,
                [["pair_id", "a_candidate_index", "b_candidate_index"]],
                name,
            )

    def test_agent_layer_contains_no_evaluator_truth_or_outputs(self):
        forbidden = {
            "objective_values.npy",
            "pool_objectives.jsonl",
            "normalization_F.json",
            "truth.tsv",
            "candidates.csv",
            "pairs.tsv",
            "preferences.jsonl",
            "preference_pairs.csv",
            "evaluation.json",
            "run.json",
            "__pycache__",
        }
        for path in AGENT_ROOT.rglob("*"):
            self.assertNotIn(path.name, forbidden, path)
            self.assertNotEqual(path.suffix, ".pyc", path)

    def test_output_directories_are_empty(self):
        for name in SCENARIOS:
            output = AGENT_ROOT / name / "workspace" / "output"
            self.assertTrue((output / "analysis").is_dir(), name)
            self.assertEqual([path for path in output.rglob("*") if path.is_file()], [])

    def test_neutral_inspectors_run_without_candidate_data(self):
        for name in SCENARIOS:
            tools = AGENT_ROOT / name / "workspace" / "tools"
            inspectors = sorted(tools.glob("inspect_*.py"))
            self.assertEqual(len(inspectors), 1, name)
            completed = subprocess.run(
                [sys.executable, str(inspectors[0])],
                cwd=inspectors[0].parent.parent,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, (name, completed.stderr))
            self.assertTrue(completed.stdout.strip(), name)


if __name__ == "__main__":
    unittest.main()
