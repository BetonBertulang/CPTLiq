"""
CPT data normalization following Robertson & Wride (1998) and Youd et al. (2001).
"""

import numpy as np

# Atmospheric pressure in kPa
PA = 101.325


def calculate_vertical_stress(depth, unit_weight, gwt_depth, gamma_water=9.81):
    """
    Calculate total and effective vertical stress at each depth.

    Parameters
    ----------
    depth : array-like
        Depth array (m).
    unit_weight : array-like or float
        Soil unit weight (kN/m³). If scalar, assumed constant with depth.
    gwt_depth : float
        Depth to groundwater table (m).
    gamma_water : float
        Unit weight of water (kN/m³), default 9.81.

    Returns
    -------
    sigma_v : np.ndarray
        Total vertical stress (kPa).
    sigma_v_eff : np.ndarray
        Effective vertical stress (kPa).
    """
    depth = np.asarray(depth, dtype=float)
    if np.isscalar(unit_weight):
        unit_weight = np.full_like(depth, unit_weight)
    else:
        unit_weight = np.asarray(unit_weight, dtype=float)

    sigma_v = np.zeros_like(depth)
    for i in range(1, len(depth)):
        dz = depth[i] - depth[i - 1]
        gamma_avg = 0.5 * (unit_weight[i] + unit_weight[i - 1])
        sigma_v[i] = sigma_v[i - 1] + gamma_avg * dz

    # Hydrostatic pore water pressure
    u = np.where(depth > gwt_depth, gamma_water * (depth - gwt_depth), 0.0)

    sigma_v_eff = sigma_v - u
    sigma_v_eff = np.maximum(sigma_v_eff, 1.0)  # minimum 1 kPa to avoid division by zero

    return sigma_v, sigma_v_eff


def calculate_ic_iterative(qc, fs, sigma_v, sigma_v_eff, max_iter=20, tol=0.001):
    """
    Iteratively calculate soil behaviour type index Ic and normalized cone
    resistance qc1N (Robertson & Wride 1998).

    The stress exponent n is updated each iteration as:
        n = 0.381 * Ic + 0.05 * (sigma_v_eff / Pa) - 0.15
    bounded to [0.5, 1.0].

    Parameters
    ----------
    qc : array-like
        Cone resistance (MPa).
    fs : array-like
        Sleeve friction (MPa).
    sigma_v : array-like
        Total vertical stress (kPa).
    sigma_v_eff : array-like
        Effective vertical stress (kPa).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on Ic.

    Returns
    -------
    Ic : np.ndarray
        Soil behaviour type index.
    qc1N : np.ndarray
        Normalized cone resistance (dimensionless).
    n : np.ndarray
        Final stress exponent.
    """
    qc = np.asarray(qc, dtype=float)
    fs = np.asarray(fs, dtype=float)
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_v_eff = np.asarray(sigma_v_eff, dtype=float)

    qc_kpa = qc * 1000.0
    fs_kpa = fs * 1000.0

    n = np.ones_like(qc)
    Ic_old = np.zeros_like(qc)

    for _ in range(max_iter):
        CN = np.clip((PA / sigma_v_eff) ** n, 0.1, 2.0)

        net_qc = np.maximum(qc_kpa - sigma_v, 1.0)
        Q = np.maximum((net_qc / PA) * CN, 0.001)
        F = np.clip((fs_kpa / net_qc) * 100.0, 0.001, 10.0)

        Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)

        n_new = np.clip(0.381 * Ic + 0.05 * (sigma_v_eff / PA) - 0.15, 0.5, 1.0)

        if np.max(np.abs(Ic - Ic_old)) < tol:
            break

        n = n_new
        Ic_old = Ic.copy()

    CN = np.clip((PA / sigma_v_eff) ** n, 0.1, 2.0)
    qc1N = CN * (qc_kpa / PA)

    return Ic, qc1N, n


def calculate_kc(Ic):
    """
    Fines-content correction factor Kc (Robertson & Wride 1998).

    Parameters
    ----------
    Ic : array-like
        Soil behaviour type index.

    Returns
    -------
    Kc : np.ndarray
        Correction factor (>= 1.0).
    """
    Ic = np.asarray(Ic, dtype=float)
    Kc = np.where(
        Ic < 1.64,
        1.0,
        -0.403 * Ic ** 4 + 5.581 * Ic ** 3 - 21.63 * Ic ** 2 + 33.75 * Ic - 17.88,
    )
    return np.maximum(Kc, 1.0)


def calculate_qc1ncs(qc1N, Ic):
    """
    Calculate clean-sand equivalent normalized cone resistance qc1Ncs.

    Parameters
    ----------
    qc1N : array-like
        Normalized cone resistance.
    Ic : array-like
        Soil behaviour type index.

    Returns
    -------
    qc1Ncs : np.ndarray
        Clean-sand equivalent normalized cone resistance.
    """
    Kc = calculate_kc(Ic)
    return Kc * np.asarray(qc1N, dtype=float)
