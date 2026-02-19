# CPTLiq

A Python module for evaluating **liquefaction potential** from Cone Penetration Test (CPT) data, implementing the Robertson & Wride (1998) / Youd et al. (2001) method.

## Features

- Iterative CPT data normalization with stress-dependent exponent *n*
- Soil behaviour type index *I*c classification
- Clean-sand equivalent normalized cone resistance *q*c1Ncs
- Cyclic Stress Ratio (CSR) with depth-dependent *rd* reduction factor
- Cyclic Resistance Ratio (CRR) for Mw = 7.5
- Magnitude Scaling Factor (MSF)
- Factor of Safety (FS) against liquefaction
- Liquefaction Potential Index (LPI) with severity classification (Iwasaki et al. 1978)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import numpy as np
from cptliq import evaluate_liquefaction

# Example CPT profile
depth        = np.arange(0, 10.5, 0.5)      # m
qc           = np.full(len(depth), 5.0)      # MPa – loose sand
fs           = np.full(len(depth), 0.05)     # MPa
unit_weight  = 18.0                          # kN/m³ (constant)
gwt_depth    = 2.0                           # m
amax         = 0.3                           # g (peak ground acceleration)
Mw           = 7.5                           # moment magnitude

results = evaluate_liquefaction(
    depth, qc, fs, unit_weight, gwt_depth, amax, Mw
)

print(f"MSF:          {results['MSF']:.3f}")
print(f"LPI:          {results['LPI']:.2f}  ({results['LPI_severity']})")
print(f"Liquefied at: {depth[results['liquefied']]} m")
```

## Method

The implementation follows the procedure described in:

- **Robertson, P.K. & Wride, C.E. (1998).** Evaluating cyclic liquefaction potential using the cone penetration test. *Canadian Geotechnical Journal*, 35(3), 442–459.
- **Youd, T.L. et al. (2001).** Liquefaction Resistance of Soils: Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops. *Journal of Geotechnical and Geoenvironmental Engineering*, 127(10), 817–833.
- **Iwasaki, T. et al. (1978).** A practical method for assessing soil liquefaction potential based on case studies at various sites in Japan. *Proc. 2nd Int. Conf. on Microzonation*, San Francisco.

### Steps

1. **Vertical stresses** – total (*σv*) and effective (*σ'v*) stress profiles.
2. **Iterative normalization** – *Q*, *F*, *I*c and stress exponent *n* (converges in < 20 iterations).
3. **Fines correction** – *Kc* and *q*c1Ncs.
4. **CSR** – Cyclic Stress Ratio with Youd et al. (2001) *rd*.
5. **MSF** – Magnitude Scaling Factor = 10²·²⁴ / Mw²·⁵⁶.
6. **CRR₇.₅** – Cyclic Resistance Ratio for Mw = 7.5.
7. **FS** = (CRR₇.₅ × MSF) / CSR.
8. **Liquefaction** – FS < 1.0 **and** *I*c < 2.6 (clay-like soils excluded).
9. **LPI** – Liquefaction Potential Index integrated from 0–20 m.

## API Reference

### `evaluate_liquefaction`

```
evaluate_liquefaction(depth, qc, fs, unit_weight, gwt_depth, amax, Mw,
                      gamma_water=9.81, max_iter=20, tol=0.001) → dict
```

| Parameter     | Type                  | Description                                    |
|---------------|-----------------------|------------------------------------------------|
| `depth`       | array-like (m)        | Depth array                                    |
| `qc`          | array-like (MPa)      | Cone resistance                                |
| `fs`          | array-like (MPa)      | Sleeve friction                                |
| `unit_weight` | float or array (kN/m³)| Soil unit weight                               |
| `gwt_depth`   | float (m)             | Depth to groundwater table                     |
| `amax`        | float (g)             | Peak ground acceleration                       |
| `Mw`          | float                 | Earthquake moment magnitude                    |

**Returns** a `dict` with keys: `depth`, `sigma_v`, `sigma_v_eff`, `Ic`, `qc1N`, `qc1Ncs`, `CSR`, `CRR75`, `MSF`, `FS`, `liquefied`, `LPI`, `LPI_severity`.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
