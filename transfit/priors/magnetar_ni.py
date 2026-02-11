# transfit/priors/magnetar_ni.py
from __future__ import annotations
import numpy as np

# Parameter order for MagNiModel:
# (M_ej, v_ej, P_ms, B14, M_Ni, kappa0, kappa_gamma, T_floor)
#
# Notes:
# - E_Th_in is fixed to 0.
# - R_max_in is fixed to 1 R_sun.

MAGNI_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "P_ms",         # ms
    "B14",          # 1e14 G
    "M_Ni",         # Msun
    "kappa0",       # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

MAGNI_DEFAULT_BOUNDS = np.array([
    [0.3,   30.0],      # M_ej
    [0.1,    5.0],      # v_ej
    [0.7,   20.0],      # P_ms
    [0.1,   30.0],      # B14
    [0.0,    1.5],      # M_Ni
    [0.01,   0.5],      # kappa0
    [0.001,  1.0],      # kappa_gamma
    [1000.0, 20000.0],  # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift_days"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(MAGNI_PARAM_NAMES)
    bounds = np.array(MAGNI_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
