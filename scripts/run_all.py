#!/usr/bin/env python
"""
run_all.py

Convenience script: generates both synthetic datasets, then runs both
benchmarks, end to end. Reads `data_generation.n_samples` /
`data_generation.n_sweep_points` from the task configs so those YAML
values are the single source of truth (previously they were
documentation only -- this script silently used the generator's
hardcoded default instead). Equivalent to running:

    python scripts/generate_dataset.py --task all --n-samples <from antenna1 config> --n-sweep-points <from antenna2 config>
    python scripts/run_benchmark.py --config configs/antenna1_benchmark.yaml
    python scripts/run_benchmark.py --config configs/antenna2_benchmark.yaml
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ANTENNA1_CONFIG = REPO_ROOT / "configs" / "antenna1_benchmark.yaml"
ANTENNA2_CONFIG = REPO_ROOT / "configs" / "antenna2_benchmark.yaml"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    python = sys.executable

    with open(ANTENNA1_CONFIG) as f:
        antenna1_cfg = yaml.safe_load(f)
    with open(ANTENNA2_CONFIG) as f:
        antenna2_cfg = yaml.safe_load(f)

    n_samples = antenna1_cfg["data_generation"]["n_samples"]
    n_sweep_points = antenna2_cfg["data_generation"]["n_sweep_points"]

    run([
        python, "scripts/generate_dataset.py", "--task", "all",
        "--n-samples", str(n_samples),
        "--n-sweep-points", str(n_sweep_points),
    ])
    run([python, "scripts/run_benchmark.py", "--config", str(ANTENNA1_CONFIG)])
    run([python, "scripts/run_benchmark.py", "--config", str(ANTENNA2_CONFIG)])


if __name__ == "__main__":
    main()
