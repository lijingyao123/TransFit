# transfit/priors/iib.py
from __future__ import annotations
import numpy as np

# 你之后把 IIb 的参数名和默认范围在这里补上
IIB_PARAM_NAMES = [
    # "param1", "param2", ...
]

IIB_DEFAULT_BOUNDS = np.array([
    # [lo, hi],
], dtype=float)

T_SHIFT_NAME = "t_shift_days"
T_SHIFT_BOUNDS = (-10.0, 10.0)


def default_names_and_bounds(include_t_shift: bool = True):
    names = list(IIB_PARAM_NAMES)
    bounds = np.array(IIB_DEFAULT_BOUNDS, float)

    if include_t_shift:
        names.append(T_SHIFT_NAME)
        bounds = np.vstack([bounds, np.array([T_SHIFT_BOUNDS[0], T_SHIFT_BOUNDS[1]], float)])
    return names, bounds
