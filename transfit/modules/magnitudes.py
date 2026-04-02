from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from transfit.constants import JY

from .filters import FilterProfile


AB_ZERO_POINT_JY = 3631.0


def fnu_to_abmag(fnu: np.ndarray) -> np.ndarray:
    flux = np.asarray(fnu, float)
    out = np.full_like(flux, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0.0)
    out[good] = -2.5 * np.log10(flux[good] / (AB_ZERO_POINT_JY * JY))
    return out


def fnu_to_vega_mag(fnu: np.ndarray, *, zero_point_jy: float) -> np.ndarray:
    flux = np.asarray(fnu, float)
    out = np.full_like(flux, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0.0)
    out[good] = -2.5 * np.log10(flux[good] / (float(zero_point_jy) * JY))
    return out


def fnu_grid_to_abmag_grid(fnu_grid: np.ndarray) -> np.ndarray:
    return fnu_to_abmag(np.asarray(fnu_grid, float))


def fnu_grid_to_vega_mag_grid(
    fnu_grid: np.ndarray,
    *,
    filter_map: Dict[str, FilterProfile],
    bands: Sequence[str],
) -> np.ndarray:
    flux = np.asarray(fnu_grid, float)
    out = np.empty_like(flux, dtype=float)
    for i, band in enumerate(bands):
        zp_jy = filter_map[str(band)].zero_points_jy["vega"]
        out[i] = fnu_to_vega_mag(flux[i], zero_point_jy=zp_jy)
    return out
