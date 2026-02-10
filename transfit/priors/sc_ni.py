# transfit/priors/sc_ni.py
from __future__ import annotations
import numpy as np

# SCNi / SC_Nickel 核心模型 参数顺序：
# (M_ej, v_ej, E_Th_in, M_Ni, R_max_in, x_s, kappa0, kappa_gamma, T_floor)

SCNI_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "E_Th_in",      # 1e49 erg  (注意：你代码里 E_Th_in * 1e49)
    "M_Ni",         # Msun
    "R_max_in",     # R_sun     (注意：你代码里 R_max_in * R_sun)
    "x_s",          # [0,1]
    "kappa0",       # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

# 默认范围：宽松但别离谱（你后续可根据实际模型再收紧）
SCNI_DEFAULT_BOUNDS = np.array([
    [0.1,    10.0],      # M_ej
    [0.1,     3.0],      # v_ej
    [0.01,   10.0],      # E_Th_in   (单位 1e49 erg)
    [0.001,   1.0],      # M_Ni
    [0.1,   500.0],      # R_max_in  (单位 R_sun) 先给宽一点
    [0.0,     1.0],      # x_s
    [0.01,    0.5],      # kappa0
    [0.001,   0.5],      # kappa_gamma
    [1000.0, 20000.0],   # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift_days"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(SCNI_PARAM_NAMES)
    bounds = np.array(SCNI_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
