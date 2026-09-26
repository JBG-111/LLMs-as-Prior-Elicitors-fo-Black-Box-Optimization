# JPL EV Charging 12D

## Background

This benchmark schedules charging in the 52-EVSE JPL ACN parking facility to
balance charge completion and marginal carbon emissions.

## Decision vector and bounds

Four daily time blocks each use `(b, k_u, k_c)`. Their repeated bounds are
`[-4,2]`, `[0,15]`, and `[0,30]`, giving 12 continuous variables.

## Scalar minimization objective

The objective is `normalized_carbon_kg - normalized_completion_rate`, using
the fixed, unclipped site calibration in `definition.json`. Minimize this value.

## Evaluation and required data

Each evaluation simulates the fixed seed-46 day (2021-08-20) with GMM vehicle
sessions and SCE MOER data stored under `data/`.

## Data/framework source

The simulator is SustainGym/ACNPortal; sessions derive from ACN-Data and MOER
files from the SGIP/WattTime signal. Full citations are in third-party notices.

## Python example

```python
from benchmarks.evcharging_jpl12d.benchmark import evaluate
print(evaluate([-1, 7.5, 15] * 4))
```

## CLI example

```bash
python evaluate.py evcharging_jpl12d --x-file benchmarks/evcharging_jpl12d/examples/candidate_0.json
```

## Limitations

This is one deterministic day. Actions remain in `[0,1]`, but charging-network
projection and network-violation constraints are disabled, so MOSEK is not
required.
