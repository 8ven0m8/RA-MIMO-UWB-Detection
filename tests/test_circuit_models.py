import numpy as np

from rabench.data_generation.circuit_models import (
    build_antenna1_baseline_circuit,
    build_antenna2_baseline_circuit,
)


def test_antenna1_s11_shape_and_range():
    circuit = build_antenna1_baseline_circuit()
    freq_hz = np.linspace(2e9, 15e9, 100)
    s11 = circuit.s11_db(freq_hz)
    assert s11.shape == (100,)
    assert np.all(s11 <= 0.5)  # S11 in dB should be <= ~0
    assert np.isfinite(s11).all()


def test_antenna2_s11_shape_and_range():
    circuit = build_antenna2_baseline_circuit()
    freq_hz = np.linspace(2e9, 10e9, 80)
    s11 = circuit.s11_db(freq_hz)
    assert s11.shape == (80,)
    assert np.isfinite(s11).all()


def test_scale_component_changes_curve():
    circuit_a = build_antenna1_baseline_circuit()
    circuit_b = build_antenna1_baseline_circuit()
    circuit_b.scale_component("p1", "L", 2.0)

    freq_hz = np.linspace(2e9, 15e9, 50)
    s11_a = circuit_a.s11_db(freq_hz)
    s11_b = circuit_b.s11_db(freq_hz)
    assert not np.allclose(s11_a, s11_b)
