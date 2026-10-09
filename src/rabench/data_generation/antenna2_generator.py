"""
antenna2_generator.py

Generates the synthetic Antenna II dataset: the antenna whose S11 is
modelled/predicted by a deterministic deep ensemble of MLPs in the paper
(Section 5). Unlike Antenna I, this is a plain point-wise regression task:
(frequency, geometric params) -> single S11 value. No sequence/channel
structure is involved.

DATA SHAPE (mirrors Section 5 text)
------------------------------------
"The dataset is compiled from multiple simulated results stored in
individual Excel files, each corresponding to a unique configuration of
one or more geometric parameters. These files are merged into a single
structured dataset comprising the operating frequency, feedline width,
ground slot width, ground patch radius, and the measured S11 values."

So here we generate several Excel files, each being a full frequency sweep
at one specific geometric configuration (matching Fig. 9's parametric
sweep style: wf, wst, lg individually varied), and the downstream loader
(`rabench.data.datasets.Antenna2Dataset`) concatenates them into one flat
table, exactly like the paper describes, then adds the f^2 / interaction
feature engineering the paper also describes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rabench.data_generation.circuit_models import build_antenna2_baseline_circuit

GEOM_PARAM_NAMES: List[str] = ["feedline_width", "gnd_slot_width", "gnd_patch_radius"]

BASELINE_MM: Dict[str, float] = {
    "feedline_width": 2.0,     # wf, Fig. 9 best value
    "gnd_slot_width": 3.0,     # wst, Fig. 9 best value
    "gnd_patch_radius": 1.5,   # not explicitly swept for Antenna II in the
                               # paper text; reasonable synthetic range used
}

RANGE_MM: Dict[str, tuple[float, float]] = {
    "feedline_width": (1.0, 2.5),
    "gnd_slot_width": (1.0, 4.0),
    "gnd_patch_radius": (0.5, 3.0),
}

PARAM_TO_COMPONENT: Dict[str, Dict] = {
    "feedline_width": {"block": "p1", "field": "L", "effect": "inverse", "strength": 0.5},
    "gnd_slot_width": {"block": "p2", "field": "C", "effect": "direct", "strength": 0.5},
    "gnd_patch_radius": {"block": "p4", "field": "L", "effect": "direct", "strength": 0.4},
}

FREQ_GHZ_RANGE = (2.0, 10.0)
N_FREQ_POINTS = 161


def _scale_factor(param_name: str, value_mm: float) -> float:
    lo, hi = RANGE_MM[param_name]
    baseline = BASELINE_MM[param_name]
    span = max(hi - lo, 1e-9)
    norm_delta = (value_mm - baseline) / span

    cfg = PARAM_TO_COMPONENT[param_name]
    factor = 1.0 + cfg["strength"] * norm_delta
    if cfg["effect"] == "inverse":
        factor = 1.0 / factor
    return float(np.clip(factor, 0.4, 2.5))


def _s11_curve(param_values_mm: Dict[str, float], freq_hz: np.ndarray) -> np.ndarray:
    circuit = build_antenna2_baseline_circuit()
    for pname, value in param_values_mm.items():
        factor = _scale_factor(pname, value)
        cfg = PARAM_TO_COMPONENT[pname]
        circuit.scale_component(cfg["block"], cfg["field"], factor)
    return circuit.s11_db(freq_hz)


def generate_sweep_file(
    output_dir: Path,
    varying_param: str,
    fixed_values_mm: Dict[str, float],
    sweep_values_mm: List[float],
    rng: np.random.Generator,
) -> Path:
    """One Excel file = one parameter swept across several values, all
    other geometric params fixed at baseline. Mirrors Fig. 9's structure."""
    freq_ghz = np.linspace(*FREQ_GHZ_RANGE, N_FREQ_POINTS)
    freq_hz = freq_ghz * 1e9

    rows = []
    for sweep_val in sweep_values_mm:
        params = dict(fixed_values_mm)
        params[varying_param] = sweep_val
        s11 = _s11_curve(params, freq_hz)
        noise = rng.normal(0.0, 0.2, size=s11.shape)
        s11_noisy = s11 + noise
        for f, s in zip(freq_ghz, s11_noisy):
            row = {"frequency_ghz": f, **params, "s11_db": s}
            rows.append(row)

    df = pd.DataFrame(rows)
    file_path = output_dir / f"antenna2_sweep_{varying_param}.xlsx"
    df.to_excel(file_path, index=False)
    return file_path


def generate_dataset(output_dir: str | Path, n_sweep_points: int = 6, seed: int = 7) -> List[Path]:
    """Generate one sweep-file per geometric parameter (mirrors the
    'multiple Excel files, each corresponding to a unique configuration of
    one or more geometric parameters' description in Section 5)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    files = []
    for pname in GEOM_PARAM_NAMES:
        lo, hi = RANGE_MM[pname]
        sweep_values = list(np.linspace(lo, hi, n_sweep_points))
        fixed = {k: v for k, v in BASELINE_MM.items() if k != pname}
        f = generate_sweep_file(output_dir, pname, fixed, sweep_values, rng)
        files.append(f)

    manifest = {
        "task": "antenna2",
        "geom_param_names": GEOM_PARAM_NAMES,
        "freq_ghz_range": FREQ_GHZ_RANGE,
        "n_freq_points": N_FREQ_POINTS,
        "n_sweep_points_per_param": n_sweep_points,
        "seed": seed,
        "note": (
            "Synthetic surrogate dataset generated from the paper's "
            "equivalent-circuit values (Section 4 / Fig. 7b). NOT measured "
            "or EM-simulated data. See circuit_models.py docstring."
        ),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return files


if __name__ == "__main__":
    files = generate_dataset("data/raw/antenna2", n_sweep_points=6, seed=7)
    print([str(f) for f in files])
