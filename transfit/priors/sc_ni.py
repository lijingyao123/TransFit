# transfit/priors/sc_ni.py
from __future__ import annotations
import numpy as np

# Parameter order for SCNiModel:
# (M_ej, v_ej, E_Th_in, M_Ni, R_0, x_Ni, kappa, kappa_gamma, T_floor)

SCNI_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "E_Th_in",      # 1e49 erg (model uses E_Th_in * 1e49)
    "M_Ni",         # Msun
    "R_0",          # R_sun (model uses R_0 * R_sun)
    "x_Ni",         # [0,1]
    "kappa",        # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

# Reasonably broad default bounds.
SCNI_DEFAULT_BOUNDS = np.array([
    [0.1,    10.0],      # M_ej
    [0.1,     3.0],      # v_ej
    [0.01,   10.0],      # E_Th_in (unit: 1e49 erg)
    [0.001,   1.0],      # M_Ni
    [0.1,   500.0],      # R_0 (unit: R_sun)
    [0.0,     1.0],      # x_Ni
    [0.01,    0.5],      # kappa
    [0.001,   0.5],      # kappa_gamma
    [1000.0, 20000.0],   # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(SCNI_PARAM_NAMES)
    bounds = np.array(SCNI_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
