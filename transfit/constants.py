# transfit/constants.py
# -*- coding: utf-8 -*-
"""
Project-wide numeric constants (CGS).
Keep these as plain floats (no astropy units) so they are Numba-friendly.
"""

import numpy as np

# ---- math
PI: float = float(np.pi)

# ---- cgs physical constants
C_LIGHT: float = 2.99792458e10          # cm/s
H_PLANCK: float = 6.62607015e-27        # erg*s
K_BOLTZ: float = 1.380649e-16           # erg/K
SIGMA_SB: float = 5.670316237574177e-5  # erg/s/cm^2/K^4
SIGMA_T: float = 6.652e-25              # cm^2
M_P: float = 1.673e-24                  # g
M_E: float = 9.109e-28                  # g
Q_E: float = 4.803e-10                  # statcoulomb

# ---- astronomy (cgs)
M_SUN: float = 1.9889999e33             # g
R_SUN: float = 6.96e10                  # cm
L_SUN: float = 3.828e33                 # erg/s
PC: float = 3.086e18                    # cm
MPC: float = 1.0e6 * PC                 # cm

# ---- time / frequency / flux density helpers
KM: float = 1.0e5                       # cm/s
DAY: float = 86400.0                    # s
YR: float = 365.0 * DAY                 # s (project convention)
GHZ: float = 1.0e9                      # Hz
UJY: float = 1.0e-29                    # erg/s/cm^2/Hz
MJY: float = 1.0e-26                    # erg/s/cm^2/Hz
JY: float = 1.0e-23                     # erg/s/cm^2/Hz

# ---- (optional) cosmology defaults (only if you truly want "project default")
H0: float = 71.0 * KM / MPC
OMEGA_M: float = 0.27
OMEGA_L: float = 0.73

# ---- Ni/Co heating (project model convention)
EPSILON_NI: float = 3.9e10              # erg/s/g
EPSILON_CO: float = 6.8e9               # erg/s/g
TAU_NI: float = 8.8 * DAY               # s
TAU_CO: float = 111.3 * DAY             # s
EPSILON_RATIO: float = EPSILON_CO / (EPSILON_NI - EPSILON_CO)

__all__ = [
    "PI",
    "C_LIGHT", "H_PLANCK", "K_BOLTZ", "SIGMA_SB", "SIGMA_T", "M_P", "M_E", "Q_E",
    "M_SUN", "R_SUN", "L_SUN", "PC", "MPC",
    "KM", "DAY", "YR", "GHZ", "UJY", "MJY", "JY",
    "H0", "OMEGA_M", "OMEGA_L",
    "EPSILON_NI", "EPSILON_CO", "TAU_NI", "TAU_CO", "EPSILON_RATIO",
]
