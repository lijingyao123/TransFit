# transfit/priors/magnetar.py
from __future__ import annotations
import numpy as np

# Parameter order for pure MagnetarModel:
# (M_ej, v_ej, P_ms, B14, kappa, kappa_gamma, T_floor)
#
# Notes:
# - E_Th_in is fixed to 0 in the pure magnetar model.
# - R_0 is fixed to 1 R_sun in the pure magnetar model.
# - P_ms is the initial spin period in ms.
# - B14 is the magnetic-field strength in units of 1e14 G.

MAGNETAR_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "P_ms",         # ms
    "B14",          # 1e14 G
    "kappa",        # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

# Reasonably broad default bounds.
MAGNETAR_DEFAULT_BOUNDS = np.array([
    [0.3,   30.0],      # M_ej (Msun)
    [0.1,    5.0],      # v_ej (1e9 cm/s)
    [0.7,   20.0],      # P_ms (ms)
    [0.1,   30.0],      # B14 (1e14 G)
    [0.01,   0.5],      # kappa
    [0.001,  1.0],      # kappa_gamma
    [1000.0, 20000.0],  # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(MAGNETAR_PARAM_NAMES)
    bounds = np.array(MAGNETAR_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds

