"""
CPT-based liquefaction evaluation following Robertson & Wride (1998)
and Youd et al. (2001).
"""

import numpy as np

from .normalization import (
    PA,
    calculate_vertical_stress,
    calculate_ic_iterative,
    calculate_qc1ncs,
)


def stress_reduction_factor(depth):
    """
    Depth-dependent stress reduction factor rd (Youd et al. 2001).

    Parameters
    ----------
    depth : array-like
        Depth (m).

    Returns
    -------
    rd : np.ndarray
        Stress reduction factor (dimensionless).
    """
    depth = np.asarray(depth, dtype=float)
    rd = np.where(
        depth <= 9.15,
        1.0 - 0.00765 * depth,
        np.where(
            depth <= 23.0,
            1.174 - 0.0267 * depth,
            np.where(
                depth <= 30.0,
                0.744 - 0.008 * depth,
                0.5,
            ),
        ),
    )
    return rd


def calculate_csr(sigma_v, sigma_v_eff, amax, depth):
    """
    Calculate Cyclic Stress Ratio (CSR).

    CSR = 0.65 * (sigma_v / sigma_v_eff) * amax * rd

    Parameters
    ----------
    sigma_v : array-like
        Total vertical stress (kPa).
    sigma_v_eff : array-like
        Effective vertical stress (kPa).
    amax : float
        Peak ground surface acceleration (fraction of g, e.g. 0.3 for 0.3 g).
    depth : array-like
        Depth (m).

    Returns
    -------
    CSR : np.ndarray
        Cyclic stress ratio.
    """
    rd = stress_reduction_factor(depth)
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_v_eff = np.asarray(sigma_v_eff, dtype=float)
    return 0.65 * (sigma_v / sigma_v_eff) * amax * rd


def magnitude_scaling_factor(Mw):
    """
    Magnitude Scaling Factor MSF (Youd et al. 2001).

    MSF = 10^2.24 / Mw^2.56

    Parameters
    ----------
    Mw : float
        Earthquake moment magnitude.

    Returns
    -------
    MSF : float
        Magnitude scaling factor.
    """
    return (10.0 ** 2.24) / (float(Mw) ** 2.56)


def calculate_crr75(qc1Ncs):
    """
    Calculate CRR for Mw = 7.5 using Robertson & Wride (1998).

    - qc1Ncs < 50:          CRR7.5 = 0.833 * (qc1Ncs / 1000) + 0.05
    - 50 <= qc1Ncs < 160:   CRR7.5 = 93 * (qc1Ncs / 1000)^3 + 0.08
    - qc1Ncs >= 160:        no liquefaction (CRR7.5 = inf)

    Parameters
    ----------
    qc1Ncs : array-like
        Clean-sand equivalent normalized cone resistance.

    Returns
    -------
    CRR75 : np.ndarray
        Cyclic resistance ratio for Mw = 7.5.
    """
    qc1Ncs = np.asarray(qc1Ncs, dtype=float)
    CRR75 = np.where(
        qc1Ncs < 50.0,
        0.833 * (qc1Ncs / 1000.0) + 0.05,
        np.where(
            qc1Ncs < 160.0,
            93.0 * (qc1Ncs / 1000.0) ** 3 + 0.08,
            np.inf,
        ),
    )
    return CRR75


def calculate_fs(CRR75, CSR, MSF):
    """
    Calculate Factor of Safety against liquefaction.

    FS = (CRR7.5 * MSF) / CSR

    Parameters
    ----------
    CRR75 : array-like
        Cyclic resistance ratio for Mw = 7.5.
    CSR : array-like
        Cyclic stress ratio.
    MSF : float
        Magnitude scaling factor.

    Returns
    -------
    FS : np.ndarray
        Factor of safety.
    """
    CRR75 = np.asarray(CRR75, dtype=float)
    CSR = np.asarray(CSR, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (CRR75 * float(MSF)) / CSR
    # Where CSR is zero (surface), FS is effectively infinite
    result = np.where(CSR == 0.0, np.inf, result)
    return result


def calculate_lpi(depth, FS, max_depth=20.0):
    """
    Calculate Liquefaction Potential Index (LPI) (Iwasaki et al. 1978).

    LPI = integral[0 to 20 m] of F(FS) * w(z) dz

    where:
        F(FS) = max(1 - FS, 0)
        w(z)  = 10 - 0.5 * z

    Severity classification (Iwasaki et al. 1978):
        LPI = 0          → "None"
        0  < LPI <= 2    → "Low"
        2  < LPI <= 5    → "Moderate"
        5  < LPI <= 15   → "High"
        LPI > 15         → "Very High"

    Parameters
    ----------
    depth : array-like
        Depth array (m).
    FS : array-like
        Factor of safety at each depth.
    max_depth : float
        Maximum depth for LPI integration (m), default 20.0.

    Returns
    -------
    LPI : float
        Liquefaction Potential Index.
    severity : str
        LPI severity classification.
    """
    depth = np.asarray(depth, dtype=float)
    FS = np.asarray(FS, dtype=float)

    mask = depth <= max_depth
    z = depth[mask]
    fs_vals = FS[mask]

    if len(z) < 2:
        return 0.0, "None"

    F = np.maximum(1.0 - fs_vals, 0.0)
    w = 10.0 - 0.5 * z
    LPI = float(np.trapezoid(F * w, z))

    if LPI == 0.0:
        severity = "None"
    elif LPI <= 2.0:
        severity = "Low"
    elif LPI <= 5.0:
        severity = "Moderate"
    elif LPI <= 15.0:
        severity = "High"
    else:
        severity = "Very High"

    return LPI, severity


def evaluate_liquefaction(
    depth,
    qc,
    fs,
    unit_weight,
    gwt_depth,
    amax,
    Mw,
    gamma_water=9.81,
    max_iter=20,
    tol=0.001,
):
    """
    Full CPT-based liquefaction evaluation (Robertson & Wride 1998 /
    Youd et al. 2001).

    Parameters
    ----------
    depth : array-like
        Depth array (m).
    qc : array-like
        Cone resistance (MPa).
    fs : array-like
        Sleeve friction (MPa).
    unit_weight : array-like or float
        Soil unit weight (kN/m³).
    gwt_depth : float
        Depth to groundwater table (m).
    amax : float
        Peak ground surface acceleration (fraction of g).
    Mw : float
        Earthquake moment magnitude.
    gamma_water : float
        Unit weight of water (kN/m³), default 9.81.
    max_iter : int
        Maximum iterations for Ic calculation, default 20.
    tol : float
        Convergence tolerance for Ic, default 0.001.

    Returns
    -------
    results : dict
        Dictionary with the following keys:

        - ``depth``        – depth array (m)
        - ``sigma_v``      – total vertical stress (kPa)
        - ``sigma_v_eff``  – effective vertical stress (kPa)
        - ``Ic``           – soil behaviour type index
        - ``qc1N``         – normalized cone resistance
        - ``qc1Ncs``       – clean-sand equivalent normalized cone resistance
        - ``CSR``          – cyclic stress ratio
        - ``CRR75``        – cyclic resistance ratio for Mw = 7.5
        - ``MSF``          – magnitude scaling factor (scalar)
        - ``FS``           – factor of safety
        - ``liquefied``    – boolean array (True where liquefaction occurs)
        - ``LPI``          – Liquefaction Potential Index
        - ``LPI_severity`` – LPI severity label

    Notes
    -----
    Soils with Ic >= 2.6 are classified as clay-like and considered
    non-liquefiable regardless of FS.
    """
    depth = np.asarray(depth, dtype=float)
    qc = np.asarray(qc, dtype=float)
    fs = np.asarray(fs, dtype=float)

    sigma_v, sigma_v_eff = calculate_vertical_stress(
        depth, unit_weight, gwt_depth, gamma_water
    )

    Ic, qc1N, _n = calculate_ic_iterative(
        qc, fs, sigma_v, sigma_v_eff, max_iter=max_iter, tol=tol
    )

    qc1Ncs = calculate_qc1ncs(qc1N, Ic)

    CSR = calculate_csr(sigma_v, sigma_v_eff, amax, depth)
    MSF = magnitude_scaling_factor(Mw)
    CRR75 = calculate_crr75(qc1Ncs)
    FS = calculate_fs(CRR75, CSR, MSF)

    liquefiable_soil = Ic < 2.6
    liquefied = (FS < 1.0) & liquefiable_soil

    # Non-liquefiable soils get FS = inf for LPI calculation
    FS_for_lpi = np.where(liquefiable_soil, FS, np.inf)
    LPI, LPI_severity = calculate_lpi(depth, FS_for_lpi)

    return {
        "depth": depth,
        "sigma_v": sigma_v,
        "sigma_v_eff": sigma_v_eff,
        "Ic": Ic,
        "qc1N": qc1N,
        "qc1Ncs": qc1Ncs,
        "CSR": CSR,
        "CRR75": CRR75,
        "MSF": MSF,
        "FS": FS,
        "liquefied": liquefied,
        "LPI": LPI,
        "LPI_severity": LPI_severity,
    }
