from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .extinction import ExtinctionSpec, apply_extinction_to_fnu_grid
from .filters import FilterProfile
from .fnu import (
    evaluate_mono_filter_fnu_grid,
    evaluate_multiband_model_fnu as _evaluate_multiband_model_fnu,
)
from .magnitudes import (
    fnu_grid_to_abmag_grid,
    fnu_grid_to_vega_mag_grid,
)


def validate_observation_mode(y_kind: str, mag_system: str) -> tuple[str, str]:
    y_kind_n = str(y_kind).strip().lower()
    mag_system_n = str(mag_system).strip().lower()
    if y_kind_n not in ("mag", "flux"):
        raise ValueError("y_kind must be 'mag' or 'flux'.")
    if mag_system_n not in ("ab", "vega"):
        raise ValueError("mag_system must be 'ab' or 'vega'.")
    return y_kind_n, mag_system_n


def evaluate_multiband_model_fnu(
    *,
    sed,
    filter_map: Dict[str, FilterProfile],
    bands: Sequence[str],
    Teff_K: np.ndarray,
    R_cm: np.ndarray,
    DL_cm: float,
    z: float,
) -> np.ndarray:
    """
    Canonical internal multi-band model quantity: observer-frame F_nu.
    """
    return _evaluate_multiband_model_fnu(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=Teff_K,
        R_cm=R_cm,
        DL_cm=DL_cm,
        z=z,
    )


def evaluate_multiband_extinguished_fnu(
    *,
    sed,
    filter_map: Dict[str, FilterProfile],
    bands: Sequence[str],
    Teff_K: np.ndarray,
    R_cm: np.ndarray,
    DL_cm: float,
    z: float,
    extinction: ExtinctionSpec | None,
) -> np.ndarray:
    fnu_grid = evaluate_multiband_model_fnu(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=Teff_K,
        R_cm=R_cm,
        DL_cm=DL_cm,
        z=z,
    )
    return apply_extinction_to_fnu_grid(
        fnu_grid,
        filter_map=filter_map,
        bands=bands,
        extinction=extinction,
        z=z,
    )


def fnu_grid_to_observation_output(
    fnu_grid: np.ndarray,
    *,
    filter_map: Dict[str, FilterProfile],
    bands: Sequence[str],
    y_kind: str,
    mag_system: str,
) -> np.ndarray:
    y_kind_n, mag_system_n = validate_observation_mode(y_kind, mag_system)
    if y_kind_n == "flux":
        return np.asarray(fnu_grid, float)
    if mag_system_n == "ab":
        return fnu_grid_to_abmag_grid(fnu_grid)
    return fnu_grid_to_vega_mag_grid(
        fnu_grid,
        filter_map=filter_map,
        bands=bands,
    )


def evaluate_multiband_observer_output(
    *,
    sed,
    filter_map: Dict[str, FilterProfile],
    bands: Sequence[str],
    Teff_K: np.ndarray,
    R_cm: np.ndarray,
    DL_cm: float,
    z: float,
    y_kind: str,
    mag_system: str,
    extinction: ExtinctionSpec | None,
) -> np.ndarray:
    """
    High-level multi-band pipeline:
      state -> model F_nu -> extincted model F_nu -> user observation space
    """
    fnu_ext = evaluate_multiband_extinguished_fnu(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=Teff_K,
        R_cm=R_cm,
        DL_cm=DL_cm,
        z=z,
        extinction=extinction,
    )
    return fnu_grid_to_observation_output(
        fnu_ext,
        filter_map=filter_map,
        bands=bands,
        y_kind=y_kind,
        mag_system=mag_system,
    )


# Backward-compatible aliases for the previous internal naming.
apply_extinction_to_flux_grid = apply_extinction_to_fnu_grid
evaluate_mono_filter_flux_grid = evaluate_mono_filter_fnu_grid
flux_grid_to_output = fnu_grid_to_observation_output
