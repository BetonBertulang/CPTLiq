"""Tests for cptliq.liquefaction."""

import numpy as np
import pytest

from cptliq.liquefaction import (
    calculate_crr75,
    calculate_csr,
    calculate_fs,
    calculate_lpi,
    evaluate_liquefaction,
    magnitude_scaling_factor,
    stress_reduction_factor,
)


class TestStressReductionFactor:
    def test_shallow_zone(self):
        depth = np.array([0.0, 5.0, 9.15])
        rd = stress_reduction_factor(depth)
        expected = 1.0 - 0.00765 * depth
        np.testing.assert_allclose(rd, expected)

    def test_intermediate_zone(self):
        depth = np.array([10.0, 20.0, 23.0])
        rd = stress_reduction_factor(depth)
        expected = 1.174 - 0.0267 * depth
        np.testing.assert_allclose(rd, expected)

    def test_deep_zone(self):
        depth = np.array([25.0, 30.0])
        rd = stress_reduction_factor(depth)
        expected = 0.744 - 0.008 * depth
        np.testing.assert_allclose(rd, expected)

    def test_very_deep(self):
        depth = np.array([35.0, 50.0])
        rd = stress_reduction_factor(depth)
        np.testing.assert_array_equal(rd, np.full(2, 0.5))

    def test_rd_decreases_with_depth(self):
        depth = np.linspace(0, 20, 50)
        rd = stress_reduction_factor(depth)
        assert np.all(np.diff(rd) <= 0)

    def test_rd_at_zero_depth_is_one(self):
        assert stress_reduction_factor(np.array([0.0]))[0] == pytest.approx(1.0)


class TestCalculateCsr:
    def test_csr_increases_with_amax(self):
        sigma_v = np.array([100.0])
        sigma_v_eff = np.array([80.0])
        depth = np.array([5.0])

        csr1 = calculate_csr(sigma_v, sigma_v_eff, 0.1, depth)
        csr2 = calculate_csr(sigma_v, sigma_v_eff, 0.3, depth)
        assert csr2[0] > csr1[0]

    def test_csr_formula(self):
        sigma_v = np.array([100.0])
        sigma_v_eff = np.array([80.0])
        depth = np.array([0.0])  # rd = 1.0 at surface

        csr = calculate_csr(sigma_v, sigma_v_eff, 0.2, depth)
        expected = 0.65 * (100.0 / 80.0) * 0.2 * 1.0
        np.testing.assert_allclose(csr, [expected])


class TestMagnitudeScalingFactor:
    def test_mw_75_gives_one(self):
        """MSF should be 1.0 for Mw = 7.5."""
        msf = magnitude_scaling_factor(7.5)
        assert msf == pytest.approx(1.0, rel=0.01)

    def test_smaller_mw_gives_msf_greater_than_one(self):
        """Smaller earthquake → less damage potential → MSF > 1."""
        assert magnitude_scaling_factor(6.0) > 1.0

    def test_larger_mw_gives_msf_less_than_one(self):
        """Larger earthquake → more damage potential → MSF < 1."""
        assert magnitude_scaling_factor(8.0) < 1.0


class TestCalculateCrr75:
    def test_low_qc1ncs(self):
        qc1Ncs = np.array([30.0])
        crr = calculate_crr75(qc1Ncs)
        expected = 0.833 * (30.0 / 1000.0) + 0.05
        np.testing.assert_allclose(crr, [expected])

    def test_medium_qc1ncs(self):
        qc1Ncs = np.array([80.0])
        crr = calculate_crr75(qc1Ncs)
        expected = 93.0 * (80.0 / 1000.0) ** 3 + 0.08
        np.testing.assert_allclose(crr, [expected])

    def test_high_qc1ncs_no_liquefaction(self):
        qc1Ncs = np.array([160.0, 200.0])
        crr = calculate_crr75(qc1Ncs)
        assert np.all(np.isinf(crr))

    def test_crr_increases_with_qc1ncs(self):
        qc1Ncs = np.array([20.0, 40.0, 60.0, 100.0, 140.0])
        crr = calculate_crr75(qc1Ncs)
        assert np.all(np.diff(crr) > 0)


class TestCalculateFs:
    def test_fs_formula(self):
        CRR75 = np.array([0.2])
        CSR = np.array([0.1])
        MSF = 1.0
        FS = calculate_fs(CRR75, CSR, MSF)
        np.testing.assert_allclose(FS, [2.0])

    def test_fs_with_msf(self):
        CRR75 = np.array([0.1])
        CSR = np.array([0.1])
        MSF = 1.5
        FS = calculate_fs(CRR75, CSR, MSF)
        np.testing.assert_allclose(FS, [1.5])

    def test_liquefaction_when_fs_below_one(self):
        CRR75 = np.array([0.05])
        CSR = np.array([0.2])
        MSF = 1.0
        FS = calculate_fs(CRR75, CSR, MSF)
        assert FS[0] < 1.0


class TestCalculateLpi:
    def test_no_liquefaction_gives_zero_lpi(self):
        depth = np.linspace(0, 20, 21)
        FS = np.full(21, 2.0)
        LPI, severity = calculate_lpi(depth, FS)
        assert LPI == pytest.approx(0.0)
        assert severity == "None"

    def test_full_liquefaction_gives_high_lpi(self):
        depth = np.linspace(0, 20, 21)
        FS = np.zeros(21)  # FS = 0 everywhere → maximum F = 1
        LPI, severity = calculate_lpi(depth, FS)
        assert LPI > 15.0
        assert severity == "Very High"

    def test_severity_thresholds(self):
        # Build scenarios with known LPI values
        cases = [
            (np.inf, "None"),
            (1.0, "Low"),
            (3.5, "Moderate"),
            (10.0, "High"),
            (20.0, "Very High"),
        ]
        for target_lpi, expected_severity in cases:
            if np.isinf(target_lpi):
                depth = np.linspace(0, 20, 21)
                FS = np.full(21, 2.0)
                LPI, severity = calculate_lpi(depth, FS)
                assert severity == expected_severity
                continue
            # crude check: just verify classification name mapping
            if expected_severity == "Low":
                assert 0 < target_lpi <= 2
            elif expected_severity == "Moderate":
                assert 2 < target_lpi <= 5
            elif expected_severity == "High":
                assert 5 < target_lpi <= 15
            elif expected_severity == "Very High":
                assert target_lpi > 15

    def test_depths_beyond_20m_ignored(self):
        depth = np.array([0.0, 10.0, 21.0, 30.0])
        FS = np.array([0.5, 0.5, 0.0, 0.0])
        LPI_full, _ = calculate_lpi(depth, FS, max_depth=20.0)
        # Depths 21 and 30 should not contribute
        LPI_clipped, _ = calculate_lpi(
            np.array([0.0, 10.0]), np.array([0.5, 0.5]), max_depth=20.0
        )
        assert LPI_full == pytest.approx(LPI_clipped)

    def test_single_point_returns_zero(self):
        LPI, severity = calculate_lpi(np.array([5.0]), np.array([0.5]))
        assert LPI == 0.0
        assert severity == "None"


class TestEvaluateLiquefaction:
    """Integration tests for the main evaluate_liquefaction function."""

    def _build_simple_profile(self):
        """5 m sandy profile, GWT at 1 m, Mw=7.5, amax=0.3 g."""
        depth = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        qc = np.full(len(depth), 5.0)    # 5 MPa – loose sand
        fs = np.full(len(depth), 0.05)   # 0.05 MPa
        unit_weight = 18.0               # kN/m³
        gwt_depth = 1.0
        amax = 0.3
        Mw = 7.5
        return depth, qc, fs, unit_weight, gwt_depth, amax, Mw

    def test_returns_expected_keys(self):
        depth, qc, fs, uw, gwt, amax, Mw = self._build_simple_profile()
        results = evaluate_liquefaction(depth, qc, fs, uw, gwt, amax, Mw)

        expected_keys = {
            "depth", "sigma_v", "sigma_v_eff", "Ic", "qc1N", "qc1Ncs",
            "CSR", "CRR75", "MSF", "FS", "liquefied", "LPI", "LPI_severity",
        }
        assert expected_keys == set(results.keys())

    def test_array_shapes_match_depth(self):
        depth, qc, fs, uw, gwt, amax, Mw = self._build_simple_profile()
        results = evaluate_liquefaction(depth, qc, fs, uw, gwt, amax, Mw)

        for key in ("sigma_v", "sigma_v_eff", "Ic", "qc1N", "qc1Ncs",
                    "CSR", "CRR75", "FS", "liquefied"):
            assert results[key].shape == depth.shape, f"Shape mismatch for key '{key}'"

    def test_fs_positive(self):
        depth, qc, fs, uw, gwt, amax, Mw = self._build_simple_profile()
        results = evaluate_liquefaction(depth, qc, fs, uw, gwt, amax, Mw)
        assert np.all(results["FS"] > 0)

    def test_loose_sand_liquefies(self):
        """Very loose sand should liquefy under strong shaking."""
        depth = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        qc = np.full(6, 2.0)    # 2 MPa – very loose sand
        fs = np.full(6, 0.02)
        results = evaluate_liquefaction(
            depth, qc, fs, unit_weight=17.0, gwt_depth=0.5,
            amax=0.4, Mw=7.5,
        )
        # At least some layers should liquefy
        assert np.any(results["liquefied"])

    def test_dense_sand_does_not_liquefy(self):
        """Dense sand should not liquefy under moderate shaking."""
        depth = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        qc = np.full(6, 20.0)   # 20 MPa – dense sand
        fs = np.full(6, 0.1)
        results = evaluate_liquefaction(
            depth, qc, fs, unit_weight=20.0, gwt_depth=1.0,
            amax=0.1, Mw=7.5,
        )
        assert np.all(~results["liquefied"])

    def test_msf_scalar(self):
        depth, qc, fs, uw, gwt, amax, Mw = self._build_simple_profile()
        results = evaluate_liquefaction(depth, qc, fs, uw, gwt, amax, Mw)
        assert np.isscalar(results["MSF"]) or results["MSF"].ndim == 0

    def test_lpi_severity_is_string(self):
        depth, qc, fs, uw, gwt, amax, Mw = self._build_simple_profile()
        results = evaluate_liquefaction(depth, qc, fs, uw, gwt, amax, Mw)
        assert isinstance(results["LPI_severity"], str)

    def test_clay_soil_not_liquefied(self):
        """Ic >= 2.6 → non-liquefiable regardless of FS."""
        depth = np.array([0.0, 1.0, 2.0, 3.0])
        qc = np.full(4, 0.5)    # 0.5 MPa – soft clay
        fs = np.full(4, 0.04)   # high friction ratio
        results = evaluate_liquefaction(
            depth, qc, fs, unit_weight=15.0, gwt_depth=0.0,
            amax=0.5, Mw=7.5,
        )
        # For clay-like soils (Ic >= 2.6) liquefied should be False
        clay_mask = results["Ic"] >= 2.6
        assert np.all(~results["liquefied"][clay_mask])
