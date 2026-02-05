# transfit/priors/nickel.py
from __future__ import annotations
import numpy as np

# NickelModel 参数顺序：
# (M_ej, v_ej, E_Th_in, M_Ni, R_max_in, x_s, kappa0, kappa_gamma, T_floor)

NICKEL_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "M_Ni",         # Msun
    "x_s",          # [0,1]
    "kappa0",       # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

# 给一个“宽松但别离谱”的默认范围，后面你可以再收紧
NICKEL_DEFAULT_BOUNDS = np.array([
    [0.1,   10],     # M_ej
    [0.1,    3],     # v_ej
    [0.001,  1.0],     # M_Ni
    [0.0,    1.0],     # x_s
    [0.01,   0.5],     # kappa0
    [0.001, 0.5],     # kappa_gamma
    [1000.0, 20000.0], # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift_days"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(NICKEL_PARAM_NAMES)
    bounds = np.array(NICKEL_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
