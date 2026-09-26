# Preference-Informed Bayesian Optimization: Six-Benchmark Release

This package contains three independent but compatible components:

1. six executable black-box benchmarks;
2. six lightweight, no-Skill Codex Agent workspace templates for eliciting pairwise preferences before objective evaluation; and
3. the Sparse Preference BO implementation used to turn hard Agent preferences into a probabilistic prior and update it with direct objective observations.

No generated preference dataset, optimization trajectory, evaluation report, figure, log, or historical experiment output is included. The COF arrays are not included because their upstream redistribution terms were not explicit; the COF interface remains available for user-supplied authorized data.

## Included benchmarks

| Identifier | Domain | Dimension | Search space | Objective (minimize) |
|---|---|---:|---|---|
| `cof14d` | COF Xe/Kr separation | 14 | 608-point finite-pool | negative high-fidelity GCMC selectivity |
| `net3_profile_f` | Net3 pump control | 24 | continuous `[0.7,1.2]^24` | Profile F: 0.75 energy + 0.05 pressure tail + 0.20 speed variation |
| `evcharging_jpl12d` | JPL EV charging | 12 | bounded continuous | normalized carbon minus completion |
| `evcharging_caltech12d` | Caltech EV charging | 12 | bounded continuous | normalized carbon minus completion |
| `tsc_cologne3_11d` | traffic signal control | 11 | 5,000-point finite pool | total vehicle waiting time |
| `tsc_cologne8_25d` | traffic signal control | 25 | 5,000-point finite pool | total vehicle waiting time |

All six interfaces return a scalar objective to minimize. Cologne3 and Cologne8 are immediately usable exact lookup benchmarks. Net3 Profile F, JPL, and Caltech execute their simulator-backed objectives. COF becomes an exact lookup benchmark after the user supplies its two authorized arrays as described in `benchmarks/cof14d/data/README.md`.

## Project structure

```text
preference_bo_release/
├── benchmarks/                # executable objective functions and evaluator-private data
├── agent_workspaces/          # pre-evaluation information exposed to the Agent
├── sparse_preference_bo/      # sparse preference posterior and BO routines
├── examples/                  # small code-only examples
├── tests/                     # integrity and numerical unit tests
└── evaluate.py                # unified benchmark dispatcher
```

## Information boundary

The split between `benchmarks/` and `agent_workspaces/` is intentional.

- `benchmarks/` may contain objective tables, fixed normalization parameters, and executable simulator code needed to obtain ground truth.
- `agent_workspaces/<scenario>/workspace/` contains only information available before expensive evaluation: the problem specification, decision-variable mapping, constraints, static scenario inputs, empty candidate/pair interfaces, and neutral parsers.
- An Agent workspace never contains objective values, true pair labels, evaluation histories, generated preferences, or the executable black-box evaluator. Net3 additionally withholds the Profile F normalization bounds from the Agent.
- Run Codex with exactly one scenario's `workspace/` as `-C`; do not use the release root as the Agent working directory, because the release root contains evaluator-private benchmark data.

## Installation

Python 3.10 is the reference environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

NumPy and SciPy are sufficient for preference modeling and the lookup benchmarks. Net3 requires WNTR. JPL and Caltech require SustainGym and ACNPortal. The charging benchmarks disable in-environment action projection and therefore do not require a MOSEK license.

## Evaluate a benchmark

```bash
python evaluate.py cof14d --x-file benchmarks/cof14d/examples/candidate_0.json
python evaluate.py net3_profile_f --x-file benchmarks/net3_profile_f/examples/all_low.json
python evaluate.py evcharging_jpl12d --x-file benchmarks/evcharging_jpl12d/examples/candidate_0.json
python evaluate.py evcharging_caltech12d --x-file benchmarks/evcharging_caltech12d/examples/candidate_0.json
python evaluate.py tsc_cologne3_11d --x-file benchmarks/tsc_cologne3_11d/examples/candidate_0.json
python evaluate.py tsc_cologne8_25d --x-file benchmarks/tsc_cologne8_25d/examples/candidate_0.json
```

The first command requires user-supplied COF arrays. Those arrays are not part
of this public repository; see `benchmarks/cof14d/data/README.md`.

Python API:

```python
from evaluate import evaluate

y = evaluate("net3_profile_f", [0.7] * 24)
```

Simulator-backed modules also expose `evaluate_detailed(x)` for raw diagnostic components.

## Use an Agent workspace

Each scenario follows the same interface:

```text
agent_workspaces/<scenario>/workspace/
├── AGENTS.md
├── input/
│   ├── problem_context.json
│   ├── candidates.template.csv
│   ├── pairs.template.tsv
│   └── simulation/
├── tools/
└── output/analysis/
```

To run a new elicitation:

1. Copy one complete `workspace/` directory to a separate run directory.
2. Rename `candidates.template.csv` to `candidates.csv` and fill one unique `candidate_index` plus the scenario-specific decision variables per row.
3. Rename `pairs.template.tsv` to `pairs.tsv` and fill `pair_id`, `a_candidate_index`, and `b_candidate_index`.
4. Keep the provided `AGENTS.md`, context, static inputs, and neutral tools unchanged unless the benchmark definition itself changes.
5. Invoke Codex from the copied workspace only:

```bash
codex exec --ephemeral --sandbox workspace-write --skip-git-repo-check --disable plugins \
  -C /absolute/path/to/copied/workspace \
  "Please follow AGENTS.md, compare every pair in input/pairs.tsv, and write output/preferences.jsonl."
```

The Agent uses the categorical values `A`, `B`, `tie`, and `insufficient_information`. For Sparse Preference BO, `A` becomes `+1`, `B` becomes `-1`, and the two abstention classes are reported but excluded from the binary probit fit.

## Sparse Preference BO

The Sparse Preference Posterior implements the released method directly:

- feature standardization and an RBF kernel;
- Nyström inducing points;
- a probit pairwise likelihood over all hard A/B labels;
- stochastic MAP fitting; and
- diagonal Laplace posterior uncertainty.

The implementation intentionally uses unit-weight hard labels. It does not add confidence weighting, reliability filtering, Feature-CV, or label-flip noise. After direct objective evaluations become available, `conditioned_gp.py` conditions the preference prior on standardized utility `-y`. `finite_pool_bo.py` supports finite candidate libraries, while `continuous_bo.py` proposes bounded continuous points. Both EI and the experiment-compatible objective-LCB/utility-UCB acquisition are provided.

```python
from sparse_preference_bo import (
    SparsePreferencePosterior,
    load_candidate_matrix,
    load_hard_preferences,
)

candidate_ids, x, columns = load_candidate_matrix("candidates.csv")
preferences = load_hard_preferences("preferences.jsonl", candidate_ids)
prior = SparsePreferencePosterior.fit(x, preferences.pairs, preferences.labels)
```

See `examples/` for executable synthetic demonstrations that do not ship experimental outputs.

## Verification

```bash
python -m unittest discover -s tests -v
```

Third-party origins and redistribution notes are recorded in `THIRD_PARTY_NOTICES.md`. The upstream COF arrays are intentionally excluded from this public release.
