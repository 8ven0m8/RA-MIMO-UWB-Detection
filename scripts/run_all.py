#!/usr/bin/env python
"""
run_all.py

Convenience script: generates both synthetic datasets, then runs both
benchmarks, end to end. Reads `data_generation.n_samples` /
`data_generation.n_sweep_points` and `data_dir` from the task configs so
those YAML values are the single source of truth (previously they were
documentation only -- this script silently used the generator's
hardcoded default instead). Equivalent to running:

    python scripts/generate_dataset.py --task antenna1 --n-samples <from config> --out-dir <parent of data_dir>
    python scripts/generate_dataset.py --task antenna2 --n-sweep-points <from config> --out-dir <parent of data_dir>
    python scripts/run_benchmark.py --config configs/antenna1_benchmark[_<variant>].yaml
    python scripts/run_benchmark.py --config configs/antenna2_benchmark[_<variant>].yaml

Usage:
    python scripts/run_all.py                                # baseline configs
    python scripts/run_all.py --variant large --require-gpu  # *_large.yaml configs, abort if no CUDA
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def config_path(task: str, variant: str | None) -> Path:
    suffix = f"_{variant}" if variant else ""
    path = CONFIG_DIR / f"{task}_benchmark{suffix}.yaml"
    if not path.exists():
        sys.exit(f"Config not found: {path}")
    return path


def check_gpu(require: bool) -> None:
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})", flush=True)
        return
    msg = f"No CUDA GPU visible to torch {torch.__version__} (built for CUDA {torch.version.cuda}); deep models will train on CPU."
    if require:
        sys.exit(f"ERROR: {msg} Check `nvidia-smi`.")
    print(f"WARNING: {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=None, help="Config suffix, e.g. 'large' -> configs/*_benchmark_large.yaml")
    parser.add_argument("--require-gpu", action="store_true", help="Abort before training if CUDA is unavailable")
    args = parser.parse_args()

    check_gpu(args.require_gpu)

    python = sys.executable
    antenna1_config = config_path("antenna1", args.variant)
    antenna2_config = config_path("antenna2", args.variant)

    with open(antenna1_config) as f:
        antenna1_cfg = yaml.safe_load(f)
    with open(antenna2_config) as f:
        antenna2_cfg = yaml.safe_load(f)

    # generate_dataset.py writes to <out-dir>/<task>, so out-dir is the
    # parent of each config's data_dir.
    run([
        python, "scripts/generate_dataset.py", "--task", "antenna1",
        "--n-samples", str(antenna1_cfg["data_generation"]["n_samples"]),
        "--out-dir", str((REPO_ROOT / antenna1_cfg["data_dir"]).parent),
    ])
    run([
        python, "scripts/generate_dataset.py", "--task", "antenna2",
        "--n-sweep-points", str(antenna2_cfg["data_generation"]["n_sweep_points"]),
        "--out-dir", str((REPO_ROOT / antenna2_cfg["data_dir"]).parent),
    ])
    run([python, "scripts/run_benchmark.py", "--config", str(antenna1_config)])
    run([python, "scripts/run_benchmark.py", "--config", str(antenna2_config)])


if __name__ == "__main__":
    main()
