"""
CPTLiq – CPT-based liquefaction evaluation module.

Main entry point:

    from cptliq import evaluate_liquefaction

    results = evaluate_liquefaction(
        depth, qc, fs, unit_weight, gwt_depth, amax, Mw
    )
"""

from .liquefaction import (
    calculate_crr75,
    calculate_csr,
    calculate_fs,
    calculate_lpi,
    evaluate_liquefaction,
    magnitude_scaling_factor,
    stress_reduction_factor,
)
from .normalization import (
    PA,
    calculate_ic_iterative,
    calculate_kc,
    calculate_qc1ncs,
    calculate_vertical_stress,
)

__all__ = [
    "PA",
    "calculate_crr75",
    "calculate_csr",
    "calculate_fs",
    "calculate_ic_iterative",
    "calculate_kc",
    "calculate_lpi",
    "calculate_qc1ncs",
    "calculate_vertical_stress",
    "evaluate_liquefaction",
    "magnitude_scaling_factor",
    "stress_reduction_factor",
]
