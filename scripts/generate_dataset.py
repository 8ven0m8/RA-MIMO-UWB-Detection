#!/usr/bin/env python
"""
generate_dataset.py

Generates the synthetic Antenna I and/or Antenna II datasets from the
paper's equivalent-circuit values.

Usage:
    python scripts/generate_dataset.py --task all
    python scripts/generate_dataset.py --task antenna1 --n-samples 10
    python scripts/generate_dataset.py --task antenna2 --n-sweep-points 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rabench.data_generation import antenna1_generator, antenna2_generator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["antenna1", "antenna2", "all"], default="all")
    parser.add_argument("--n-samples", type=int, default=8, help="Antenna I: number of LOOCV-fold samples")
    parser.add_argument("--n-sweep-points", type=int, default=6, help="Antenna II: sweep points per geometric param")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "data" / "raw"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.task in ("antenna1", "all"):
        meta = antenna1_generator.generate_dataset(
            out_dir / "antenna1", n_samples=args.n_samples, seed=args.seed
        )
        print(f"[antenna1] generated {len(meta)} samples -> {out_dir / 'antenna1'}")
        print(meta)

    if args.task in ("antenna2", "all"):
        files = antenna2_generator.generate_dataset(
            out_dir / "antenna2", n_sweep_points=args.n_sweep_points, seed=args.seed
        )
        print(f"[antenna2] generated {len(files)} sweep files -> {out_dir / 'antenna2'}")
        for f in files:
            print(" -", f.name)


if __name__ == "__main__":
    main()
