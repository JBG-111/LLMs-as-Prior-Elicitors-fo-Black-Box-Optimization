# Net3 Profile F

## Background

This benchmark optimizes one day of pump operation in the EPANET Example
Network 3 water-distribution system.

## Decision vector and bounds

The continuous 24D vector contains twelve two-hour speed multipliers for pump
10 followed by twelve for pump 335. Every value is in `[0.7, 1.2]`.

## Scalar minimization objective

The objective is `0.75 * normalized_energy + 0.05 * normalized_tail_loss +
0.20 * cyclic_speed_variation`. The normalization constants are frozen in
`data/normalization_F.json`. Minimize this scalar objective.

## Evaluation and required data

Each call runs a one-day WNTR/EPANET simulation using `data/Net3.inp` and the
settings in `definition.json`.

## Data/framework source

Net3 is an EPANET example network distributed with WNTR. Source and license
details appear in the suite's third-party notices.

## Python example

```python
from benchmarks.net3_profile_f.benchmark import evaluate
print(evaluate([0.7] * 24))
```

## CLI example

```bash
python evaluate.py net3_profile_f --x-file benchmarks/net3_profile_f/examples/all_low.json
```

## Limitations

Minimum pressure and terminal tank levels are diagnostics, not explicit
constraints. Simulation failure receives the benchmark's established finite
penalty.
