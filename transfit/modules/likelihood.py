from __future__ import annotations

import numpy as np

_MAG_TO_FRAC_FLUX = 0.4 * np.log(10.0)


def gaussian_lnlike(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    y_obs = np.asarray(y_obs, float)
    y_model = np.asarray(y_model, float)
    y_err = np.asarray(y_err, float)

    good = np.isfinite(y_obs) & np.isfinite(y_model) & np.isfinite(y_err) & (y_err > 0)
    if not np.any(good):
        return -np.inf

    r = (y_obs[good] - y_model[good]) / y_err[good]
    return -0.5 * float(np.sum(r * r))


def gaussian_lnlike_flux(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    """
    Gaussian likelihood in flux-density space.
    """
    return gaussian_lnlike(y_obs, y_model, y_err)


def gaussian_lnlike_mag(y_obs: np.ndarray, y_model: np.ndarray, y_err: np.ndarray) -> float:
    """
    Gaussian likelihood in magnitude space.
    """
    return gaussian_lnlike(y_obs, y_model, y_err)


def gaussian_lnlike_for_observation(
    *,
    y_kind: str,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    y_err: np.ndarray,
) -> float:
    kind = str(y_kind).strip().lower()
    if kind == "flux":
        return gaussian_lnlike_flux(y_obs, y_model, y_err)
    if kind == "mag":
        return gaussian_lnlike_mag(y_obs, y_model, y_err)
    raise ValueError("y_kind must be 'mag' or 'flux'.")


def gaussian_lnlike_with_nuisance(
    *,
    y_kind: str,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    y_err: np.ndarray,
    nuisance_params: dict[str, float] | None = None,
) -> float:
    """
    Gaussian likelihood with optional likelihood-only nuisance parameters.

    When no nuisance parameters are active, this preserves the historical
    TransFit likelihood exactly. When ``sigma_int`` is provided, the Gaussian
    normalization term is included because the scatter can be sampled.
    """
    sigma_int = dict(nuisance_params or {}).get("sigma_int")
    if sigma_int is None:
        return gaussian_lnlike_for_observation(
            y_kind=y_kind,
            y_obs=y_obs,
            y_model=y_model,
            y_err=y_err,
        )

    sigma_int = float(sigma_int)
    if not np.isfinite(sigma_int) or sigma_int < 0.0:
        return -np.inf

    y_obs = np.asarray(y_obs, float)
    y_model = np.asarray(y_model, float)
    y_err = np.asarray(y_err, float)
    kind = str(y_kind).strip().lower()

    good = np.isfinite(y_obs) & np.isfinite(y_model) & np.isfinite(y_err) & (y_err > 0)
    if not np.any(good):
        return -np.inf

    if kind == "mag":
        extra = np.full(np.sum(good), sigma_int, dtype=float)
    elif kind == "flux":
        extra = _MAG_TO_FRAC_FLUX * sigma_int * np.abs(y_obs[good])
    else:
        raise ValueError("y_kind must be 'mag' or 'flux'.")

    var = y_err[good] * y_err[good] + extra * extra
    if np.any(~np.isfinite(var)) or np.any(var <= 0.0):
        return -np.inf

    resid = y_obs[good] - y_model[good]
    return -0.5 * float(np.sum((resid * resid) / var + np.log(2.0 * np.pi * var)))
