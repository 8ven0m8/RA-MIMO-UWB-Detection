"""
antenna1_generator.py

Generates the synthetic Antenna I dataset: the UWB microstrip patch antenna
whose S11 is modelled/predicted by a 1D UNet CNN in the paper (Section 3).

DATA SHAPE (mirrors the paper's description in Section 3 + Table 1 + Table 2)
-------------------------------------------------------------------------------
The paper says: "input data consists of multiple Excel files, each
representing a distinct antenna sample characterized by frequency values
and six S11-related design parameters" and the model predicts "6-channel
output corresponding to the predicted S11 parameters", with each channel
tied to one named geometric parameter (Table 2: gndslot_width,
feedline_width, patch_strip_length, patch_strip_width, ellipse_cut_radius,
radius_of_semicircle_cut).

We reproduce that structure like this:
  * One "sample" = one Excel file = one LOOCV fold, matching Table 1
    (a full 6-parameter configuration vector, e.g.
    [1.0, 3.0, 1.0, 0.4, 1.0, 9.0]).
  * Within a sample's file, there are 6 output columns ("channels"). Channel
    i is the S11(f) curve you get when parameter i is set to this sample's
    assigned value for that parameter, while the OTHER 5 parameters are held
    at the global baseline. This matches Table 2 / Fig. 4-5, where each
    channel is captioned by a single parameter name and shows one
    true-vs-predicted curve.
  * Frequency sweep: 2-15 GHz (matches the axis range used throughout the
    paper's Antenna I figures).

This file-per-sample layout is deliberately the same shape LOOCV expects:
`rabench.data.datasets.Antenna1Dataset` treats each generated file as one
fold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rabench.data_generation.circuit_models import build_antenna1_baseline_circuit

# --------------------------------------------------------------------------
# Parameter definitions
# --------------------------------------------------------------------------

# Names match Table 2 of the paper exactly (used as column / channel names
# everywhere downstream).
PARAM_NAMES: List[str] = [
    "gndslot_width",
    "feedline_width",
    "patch_strip_length",
    "patch_strip_width",
    "ellipse_cut_radius",
    "radius_of_semicircle_cut",
]

# Baseline (mm) and sweep ranges (mm). wf, wst, rc, lstrip ranges are taken
# from the paper's Fig. 3 parametric sweeps; patch_strip_width and
# radius_of_semicircle_cut are not swept explicitly in the paper, so
# reasonable synthetic ranges are assigned (documented here, not hidden).
PARAM_BASELINE_MM: Dict[str, float] = {
    "gndslot_width": 3.0,           # wst, Fig. 3(b): 1-6 mm
    "feedline_width": 3.0,          # wf,  Fig. 3(a): 1-3.5 mm (best at 3mm)
    "patch_strip_length": 2.0,      # lstrip, Fig. 3(d): 1-3 mm
    "patch_strip_width": 1.0,       # not swept in paper -- assumed range
    "ellipse_cut_radius": 9.0,      # rc, Fig. 3(c): 7-9.5 mm
    "radius_of_semicircle_cut": 1.0,  # not swept in paper -- assumed range
}

PARAM_RANGE_MM: Dict[str, tuple[float, float]] = {
    "gndslot_width": (1.0, 6.0),
    "feedline_width": (1.0, 3.5),
    "patch_strip_length": (1.0, 3.0),
    "patch_strip_width": (0.3, 2.5),
    "ellipse_cut_radius": (7.0, 9.5),
    "radius_of_semicircle_cut": (0.3, 2.5),
}

# Which circuit component each geometric parameter perturbs, and how
# strongly (heuristic, see circuit_models.py module docstring for caveats).
# effect: "direct" -> component scales up as parameter increases
#         "inverse" -> component scales down as parameter increases
PARAM_TO_COMPONENT: Dict[str, Dict] = {
    "feedline_width": {"block": "p1", "field": "L", "effect": "inverse", "strength": 0.5},
    "gndslot_width": {"block": "p2", "field": "C", "effect": "direct", "strength": 0.6},
    "patch_strip_length": {"block": "p2", "field": "L", "effect": "direct", "strength": 0.4},
    "patch_strip_width": {"block": "p1", "field": "R", "effect": "direct", "strength": 0.3},
    "ellipse_cut_radius": {"block": "p1", "field": "C", "effect": "direct", "strength": 0.3},
    "radius_of_semicircle_cut": {"block": "s2", "field": "L", "effect": "direct", "strength": 0.35},
}

FREQ_GHZ_RANGE = (2.0, 15.0)
N_FREQ_POINTS = 251


def _scale_factor(param_name: str, value_mm: float) -> float:
    """Map a geometric parameter's value to a multiplicative scale factor
    for its associated circuit component, normalised by the parameter's
    full sweep range so factors stay in a sane (~0.4x-2.5x) band."""
    lo, hi = PARAM_RANGE_MM[param_name]
    baseline = PARAM_BASELINE_MM[param_name]
    span = max(hi - lo, 1e-9)
    norm_delta = (value_mm - baseline) / span  # roughly in [-1, 1]

    cfg = PARAM_TO_COMPONENT[param_name]
    strength = cfg["strength"]
    factor = 1.0 + strength * norm_delta
    if cfg["effect"] == "inverse":
        factor = 1.0 / factor
    return float(np.clip(factor, 0.4, 2.5))


def _s11_for_single_param_override(param_name: str, value_mm: float, freq_hz: np.ndarray) -> np.ndarray:
    """S11(f) with ONE parameter set away from baseline, all others at
    baseline. This produces one "channel" for one sample."""
    circuit = build_antenna1_baseline_circuit()
    factor = _scale_factor(param_name, value_mm)
    cfg = PARAM_TO_COMPONENT[param_name]
    circuit.scale_component(cfg["block"], cfg["field"], factor)
    return circuit.s11_db(freq_hz)


def generate_sample(param_vector_mm: Dict[str, float], rng: np.random.Generator) -> pd.DataFrame:
    """Generate one LOOCV-fold sample: a DataFrame with a `frequency_ghz`
    column plus one S11 column per geometric parameter (the "channels").
    Small measurement-like noise is added so the dataset isn't perfectly
    deterministic/noise-free (more realistic for ML benchmarking)."""
    freq_ghz = np.linspace(*FREQ_GHZ_RANGE, N_FREQ_POINTS)
    freq_hz = freq_ghz * 1e9

    data = {"frequency_ghz": freq_ghz}
    for pname in PARAM_NAMES:
        value = param_vector_mm[pname]
        s11 = _s11_for_single_param_override(pname, value, freq_hz)
        noise = rng.normal(0.0, 0.15, size=s11.shape)  # ~0.15 dB measurement noise
        data[f"s11_{pname}"] = s11 + noise

    return pd.DataFrame(data)


def generate_dataset(
    output_dir: str | Path,
    n_samples: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate `n_samples` Excel files (one per LOOCV fold) plus a
    `params.csv` metadata file listing each sample's 6-parameter vector
    (mirrors the paper's Table 1). Returns the metadata DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records = []
    for sample_idx in range(n_samples):
        param_vector = {}
        for pname in PARAM_NAMES:
            lo, hi = PARAM_RANGE_MM[pname]
            param_vector[pname] = float(rng.uniform(lo, hi))

        df = generate_sample(param_vector, rng)
        file_path = output_dir / f"antenna1_sample_{sample_idx:02d}.xlsx"
        df.to_excel(file_path, index=False)

        record = {"sample_idx": sample_idx, "file": file_path.name}
        record.update(param_vector)
        records.append(record)

    meta_df = pd.DataFrame(records)
    meta_df.to_csv(output_dir / "params.csv", index=False)

    manifest = {
        "task": "antenna1",
        "param_names": PARAM_NAMES,
        "n_samples": n_samples,
        "freq_ghz_range": FREQ_GHZ_RANGE,
        "n_freq_points": N_FREQ_POINTS,
        "seed": seed,
        "note": (
            "Synthetic surrogate dataset generated from the paper's "
            "equivalent-circuit values (Section 2 / Fig. 2e). NOT measured "
            "or EM-simulated data. See circuit_models.py docstring."
        ),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return meta_df


if __name__ == "__main__":
    meta = generate_dataset("data/raw/antenna1", n_samples=8, seed=42)
    print(meta)
