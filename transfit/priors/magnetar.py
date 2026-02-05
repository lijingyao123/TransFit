# transfit/priors/magnetar.py
from __future__ import annotations
import numpy as np

# MagnetarModel 参数顺序（与 magnetar_model.calculate_light_curve() 一致）：
# (M_ej, v_ej, E_Th_in, P_ms, B14, R_max_in, kappa0, kappa_gamma, T_floor)
#
# 说明：
# - E_Th_in : 初始热能（单位：1e49 erg；模型内部会 *1e49）
# - P_ms    : 初始自转周期（ms）
# - B14     : 磁场强度（1e14 G）
# - R_max_in: 初始外半径（单位：R_sun；模型内部会 *R_sun）
# - kappa_gamma: 仍用于沉积/漏逸 dep(t) 里的 t_gamma 标度

MAGNETAR_PARAM_NAMES = [
    "M_ej",         # Msun
    "v_ej",         # 1e9 cm/s
    "E_Th_in",      # 1e49 erg
    "P_ms",         # ms
    "B14",          # 1e14 G
    "R_max_in",     # R_sun
    "kappa0",       # cm^2/g
    "kappa_gamma",  # cm^2/g
    "T_floor",      # K
]

# 默认范围：尽量“宽松但别离谱”
MAGNETAR_DEFAULT_BOUNDS = np.array([
    [0.3,   30.0],      # M_ej (Msun)
    [0.1,    5.0],      # v_ej (1e9 cm/s)
    [0.0,  50.0],       # E_Th_in (1e49 erg)  => 0 ~ 5e50 erg
    [0.7,   20.0],      # P_ms (ms)
    [0.1,   30.0],      # B14 (1e14 G)）
    [0.1, 1000.0],      # R_max_in (R_sun)
    [0.01,   0.5],      # kappa0
    [0.001,  1.0],      # kappa_gamma
    [1000.0, 20000.0],  # T_floor
], dtype=float)

T_SHIFT_NAME = "t_shift_days"
T_SHIFT_BOUNDS = (-10.0, 20.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(MAGNETAR_PARAM_NAMES)
    bounds = np.array(MAGNETAR_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds

