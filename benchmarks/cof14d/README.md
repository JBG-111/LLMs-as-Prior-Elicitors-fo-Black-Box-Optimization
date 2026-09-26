# COF 14D

## Background

This benchmark screens covalent organic frameworks for Xe/Kr separation.

## Decision vector and bounds

The decision vector contains 14 structural and compositional descriptors. It
must match one of the 608 published candidates; arbitrary off-pool vectors are
not valid materials in this benchmark.

## Scalar minimization objective

The objective is the negative high-fidelity GCMC Xe/Kr selectivity. Minimize
the returned value; a more negative value represents greater selectivity.

## Evaluation and required data

`benchmark.py` performs an exact-tolerance lookup in user-supplied
`data/candidates.npy` and returns the aligned entry in
`data/objective_values.npy`. These arrays are not distributed in this public
release. See `data/README.md` for filenames, expected hashes, and provenance.

## Data/framework source

The source is the public Gantzler et al. COF Xe/Kr repository cited in the
suite's third-party notices. Users must obtain the arrays from an authorized
source and comply with its terms.

## Python example

```python
from benchmarks.cof14d.benchmark import evaluate
print(evaluate([3.84928, 0.3102, 4190.07, 1049.36734440269,
                1/7, 1/7, 3/7, 2/7, 0, 0, 0, 0, 0, 0]))
```

## CLI example

```bash
python evaluate.py cof14d --x-file benchmarks/cof14d/examples/candidate_0.json
```

## Limitations

The package queries archived GCMC results and does not run a new molecular
simulation. Numerical descriptor bounds do not define a continuous domain.
