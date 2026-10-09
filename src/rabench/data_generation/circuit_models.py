"""
circuit_models.py

Physics core for the synthetic ("Option A") dataset generator.

WHY THIS EXISTS
----------------
Das et al. 2026 ("Deep learning enabled heterogeneous MIMO antenna sensor
for detection of rheumatoid arthritis") fit an equivalent lumped-element
circuit to each fabricated antenna's measured S11(f) curve (their Fig. 2e
for Antenna I, Fig. 7b for Antenna II). Those component values ARE given
explicitly in the paper text. We do not have the fabricated antennas, the
full-wave EM simulator, or the measured phantom data -- but we DO have a
physically grounded circuit that reproduces the same qualitative multi-
resonance S11 behaviour.

This module turns that circuit into a callable Zin(f) -> S11(f) function,
and lets geometric design parameters (feedline width, slot width, etc.)
perturb specific L/C/R components. This is how a real change in antenna
geometry is generated synthetically here: instead of re-running an EM
solver, we scale the lumped element that the paper's text says that
geometric feature predominantly influences.

IMPORTANT / HONESTY NOTE
-------------------------
This is a SURROGATE, not a physical reconstruction of the paper's exact
circuit topology (the paper does not give the exact interconnection
topology in closed form, only a schematic figure and component values).
The topology chosen here (series cascade of resonant blocks) is a common,
reasonable way to combine multiple RLC resonators to reproduce a
multi-dip UWB S11 response, and the component *values* at baseline are
taken directly from the paper. Anyone using this dataset for anything
beyond benchmarking model architectures / ML pipelines should treat the
resulting S11 curves as "plausible synthetic UWB antenna data", not
ground truth for antenna design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

import numpy as np

# --------------------------------------------------------------------------
# Generic RLC building blocks
# --------------------------------------------------------------------------

BlockType = Literal["parallel", "series"]


@dataclass
class RLCBlock:
    """A single resonant block: either a parallel or series RLC combination.

    R is optional. If omitted, a very large (parallel) / very small (series)
    resistance is substituted so the block behaves as lossless, which is
    used for the Antenna II blocks where the paper only specifies R for two
    of the five parallel resonators.
    """

    name: str
    kind: BlockType
    L: float  # Henries
    C: float  # Farads
    R: float | None = None  # Ohms

    def impedance(self, freq_hz: np.ndarray) -> np.ndarray:
        w = 2 * np.pi * freq_hz
        jwL = 1j * w * self.L
        jwC = 1j * w * self.C
        if self.kind == "series":
            r = self.R if self.R is not None else 1e-6
            return r + jwL + 1.0 / jwC
        else:  # parallel
            r = self.R if self.R is not None else 1e6
            y = (1.0 / r) + (1.0 / jwL) + jwC
            return 1.0 / y


@dataclass
class CascadedCircuit:
    """A series cascade of RLC blocks representing one antenna's
    equivalent circuit. Zin(f) = sum of each block's impedance.
    """

    blocks: Dict[str, RLCBlock] = field(default_factory=dict)
    feed_resistance: float = 50.0  # source/feed resistance R in schematics

    def input_impedance(self, freq_hz: np.ndarray) -> np.ndarray:
        z = np.full_like(freq_hz, self.feed_resistance, dtype=complex)
        for block in self.blocks.values():
            z = z + block.impedance(freq_hz)
        return z

    def s11_db(self, freq_hz: np.ndarray, z0: float = 50.0) -> np.ndarray:
        zin = self.input_impedance(freq_hz)
        gamma = (zin - z0) / (zin + z0)
        mag = np.clip(np.abs(gamma), 1e-6, None)
        return 20.0 * np.log10(mag)

    def scale_component(self, block_name: str, param: str, factor: float) -> None:
        """Multiply a given block's L, C, or R by `factor` in place."""
        block = self.blocks[block_name]
        current = getattr(block, param)
        if current is None:
            return
        setattr(block, param, current * factor)


# --------------------------------------------------------------------------
# Baseline circuits, values taken directly from the paper (Section 2 / Fig 2e
# for Antenna I; Section 4 / Fig 7b for Antenna II).
# --------------------------------------------------------------------------


def build_antenna1_baseline_circuit() -> CascadedCircuit:
    """Antenna I: two cascaded parallel RLC resonators + two series
    RLC resonators (paper Fig. 2e / Section 2 text)."""
    blocks = {
        "p1": RLCBlock("p1", "parallel", L=0.7e-9, C=0.36e-12, R=900.0),
        "p2": RLCBlock("p2", "parallel", L=5.61e-9, C=0.31e-12, R=1000.0),
        "s1": RLCBlock("s1", "series", L=1.5e-9, C=0.076e-12, R=None),
        "s2": RLCBlock("s2", "series", L=9.21e-9, C=0.61e-12, R=1.0),
    }
    return CascadedCircuit(blocks=blocks, feed_resistance=50.0)


def build_antenna2_baseline_circuit() -> CascadedCircuit:
    """Antenna II: five cascaded parallel RLC resonators (paper Fig. 7b /
    Section 5 text). Only R1, R2 are given explicitly; R3-R5 are treated
    as lossless (very high resistance) per the module-level docstring."""
    blocks = {
        "p1": RLCBlock("p1", "parallel", L=1.85e-9, C=0.34e-12, R=390.0),
        "p2": RLCBlock("p2", "parallel", L=3.66e-9, C=0.5e-12, R=390.0),
        "p3": RLCBlock("p3", "parallel", L=0.5e-9, C=16e-12, R=None),
        "p4": RLCBlock("p4", "parallel", L=20.64e-9, C=0.5e-12, R=None),
        "p5": RLCBlock("p5", "parallel", L=0.4e-9, C=0.5e-12, R=None),
    }
    return CascadedCircuit(blocks=blocks, feed_resistance=58.0)
