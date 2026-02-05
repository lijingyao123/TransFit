# transfit/samplers/likelihood.py
from __future__ import annotations
import numpy as np


def gaussian_lnlike(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    y_obs = np.asarray(y_obs, float)
    y_model = np.asarray(y_model, float)
    y_err = np.asarray(y_err, float)

    good = np.isfinite(y_obs) & np.isfinite(y_model) & np.isfinite(y_err) & (y_err > 0)
    if not np.any(good):
        return -np.inf

    r = (y_obs[good] - y_model[good]) / y_err[good]
    return -0.5 * float(np.sum(r * r))
