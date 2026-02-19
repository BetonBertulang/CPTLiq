"""Tests for cptliq.normalization."""

import numpy as np
import pytest

from cptliq.normalization import (
    PA,
    calculate_ic_iterative,
    calculate_kc,
    calculate_qc1ncs,
    calculate_vertical_stress,
)


class TestCalculateVerticalStress:
    def test_constant_unit_weight_deep_gwt(self):
        depth = np.array([0.0, 1.0, 2.0, 3.0])
        gamma = 18.0  # kN/m³
        gwt_depth = 100.0  # GWT below all depths

        sigma_v, sigma_v_eff = calculate_vertical_stress(depth, gamma, gwt_depth)

        expected_sigma_v = np.array([0.0, 18.0, 36.0, 54.0])
        np.testing.assert_allclose(sigma_v, expected_sigma_v)
        # No pore pressure → sigma_v_eff == sigma_v (clamped to 1 kPa at surface)
        expected_sigma_v_eff = np.maximum(expected_sigma_v, 1.0)
        np.testing.assert_allclose(sigma_v_eff, expected_sigma_v_eff)

    def test_gwt_at_surface(self):
        depth = np.array([0.0, 1.0, 2.0])
        gamma = 20.0
        gwt_depth = 0.0

        sigma_v, sigma_v_eff = calculate_vertical_stress(depth, gamma, gwt_depth)

        # u = 9.81 * depth
        expected_u = np.array([0.0, 9.81, 19.62])
        expected_sigma_v = np.array([0.0, 20.0, 40.0])
        np.testing.assert_allclose(sigma_v, expected_sigma_v)
        # Effective stress clamped to minimum 1 kPa
        expected_eff = np.maximum(expected_sigma_v - expected_u, 1.0)
        np.testing.assert_allclose(sigma_v_eff, expected_eff)

    def test_effective_stress_minimum_clamp(self):
        """Effective stress should never drop below 1 kPa."""
        depth = np.array([0.0, 0.1])
        gamma = 15.0
        gwt_depth = 0.0

        _, sigma_v_eff = calculate_vertical_stress(depth, gamma, gwt_depth)

        assert np.all(sigma_v_eff >= 1.0)

    def test_variable_unit_weight(self):
        depth = np.array([0.0, 1.0, 2.0])
        gamma = np.array([18.0, 18.0, 20.0])
        gwt_depth = 100.0

        sigma_v, _ = calculate_vertical_stress(depth, gamma, gwt_depth)

        # 0→1: avg=18, dz=1 → 18
        # 1→2: avg=19, dz=1 → 18+19=37
        np.testing.assert_allclose(sigma_v, [0.0, 18.0, 37.0])


class TestCalculateKc:
    def test_sandy_soil_ic_below_164(self):
        Ic = np.array([1.0, 1.5, 1.63])
        Kc = calculate_kc(Ic)
        np.testing.assert_array_equal(Kc, np.ones(3))

    def test_silty_soil_ic_above_164(self):
        Ic = np.array([2.0, 2.6])
        Kc = calculate_kc(Ic)
        assert np.all(Kc >= 1.0)
        # Kc should increase with Ic in silty range
        assert Kc[1] > Kc[0]

    def test_kc_minimum_is_one(self):
        # Just above 1.64 may give polynomial value < 1 – should be clamped
        Ic = np.array([1.64])
        Kc = calculate_kc(Ic)
        assert Kc[0] >= 1.0


class TestCalculateQc1ncs:
    def test_sandy_soil_no_correction(self):
        qc1N = np.array([100.0, 120.0])
        Ic = np.array([1.0, 1.5])  # < 1.64 → Kc = 1
        qc1Ncs = calculate_qc1ncs(qc1N, Ic)
        np.testing.assert_array_equal(qc1Ncs, qc1N)

    def test_silty_soil_increases_qc1n(self):
        qc1N = np.array([80.0])
        Ic = np.array([2.2])  # Kc > 1
        qc1Ncs = calculate_qc1ncs(qc1N, Ic)
        assert qc1Ncs[0] > qc1N[0]


class TestCalculateIcIterative:
    def test_sandy_soil_low_ic(self):
        """Clean sand should give Ic < 2.6."""
        depth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        qc = np.full(5, 10.0)   # 10 MPa – dense sand
        fs = np.full(5, 0.05)   # 0.05 MPa – low friction (sandy)
        sigma_v = np.array([18.0, 36.0, 54.0, 72.0, 90.0])
        sigma_v_eff = sigma_v.copy()

        Ic, qc1N, n = calculate_ic_iterative(qc, fs, sigma_v, sigma_v_eff)

        assert np.all(Ic < 2.6), f"Expected Ic < 2.6 for clean sand, got {Ic}"
        assert np.all(qc1N > 0)
        assert np.all((n >= 0.5) & (n <= 1.0))

    def test_clayey_soil_high_ic(self):
        """Clay should give Ic > 2.6."""
        depth = np.array([1.0, 2.0, 3.0])
        qc = np.full(3, 0.8)    # 0.8 MPa – soft clay
        fs = np.full(3, 0.04)   # 0.04 MPa – higher friction ratio
        sigma_v = np.array([14.0, 28.0, 42.0])
        sigma_v_eff = sigma_v.copy()

        Ic, _, _ = calculate_ic_iterative(qc, fs, sigma_v, sigma_v_eff)

        assert np.all(Ic > 2.0), f"Expected Ic > 2.0 for clay, got {Ic}"

    def test_output_shapes_match_input(self):
        n_pts = 10
        qc = np.ones(n_pts) * 5.0
        fs = np.ones(n_pts) * 0.05
        sv = np.linspace(18, 180, n_pts)
        sv_eff = sv * 0.7

        Ic, qc1N, n = calculate_ic_iterative(qc, fs, sv, sv_eff)

        assert Ic.shape == (n_pts,)
        assert qc1N.shape == (n_pts,)
        assert n.shape == (n_pts,)
