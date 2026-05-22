from __future__ import annotations

import numpy as np

# Canonical parameter order for MagnetarModel:
# (M_ej, v_ej, E_Th_in, P_ms, B14, R_0, kappa, kappa_gamma, T_floor)
#
# Backward compatibility:
# - old pure-magnetar calls that omit E_Th_in and R_0 are still accepted in
#   forward-model helpers and mapped to E_Th_in=0, R_0=1 R_sun.

MAGNETAR_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "E_Th_in",      # 1e49 erg
    "P_ms",         # ms
    "B14",          # 1e14 G
    "R_0",          # R_sun
    "kappa",        # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

MAGNETAR_DEFAULT_BOUNDS = np.array([
    [0.3,   30.0],      # M_ej
    [0.1,    5.0],      # v_ej
    [0.0,   50.0],      # E_Th_in
    [0.7,   20.0],      # P_ms
    [0.1,   30.0],      # B14
    [0.1, 1000.0],      # R_0
    [0.01,   0.5],      # kappa
    [0.001,  1.0],      # kappa_gamma
    [1000.0, 20000.0],  # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift"
T_SHIFT_BOUNDS = (0.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(MAGNETAR_PARAM_NAMES)
    bounds = np.array(MAGNETAR_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
