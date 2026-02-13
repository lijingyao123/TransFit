# transfit/priors/nickel.py
from __future__ import annotations
import numpy as np

# Parameter order for NickelModel:
# (M_ej, v_ej, M_Ni, x_Ni, kappa, kappa_gamma, T_floor)
# Fixed inside the model: E_Th_in=0, R_0=10 R_sun.

NICKEL_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "M_Ni",         # Msun
    "x_Ni",         # [0,1]
    "kappa",        # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

# Reasonably broad default bounds.
NICKEL_DEFAULT_BOUNDS = np.array([
    [0.1,   10],     # M_ej
    [0.1,    3],     # v_ej
    [0.001,  1.0],     # M_Ni
    [0.0,    1.0],     # x_Ni
    [0.01,   0.5],     # kappa
    [0.001, 0.5],     # kappa_gamma
    [1000.0, 20000.0], # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(NICKEL_PARAM_NAMES)
    bounds = np.array(NICKEL_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
