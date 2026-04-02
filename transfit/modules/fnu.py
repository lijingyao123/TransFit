from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .filters import FilterProfile, mono_effective_frequency


def evaluate_mono_filter_fnu_grid(
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
    Evaluate observer-frame model F_nu on the requested mono-frequency filters.

    This is the canonical internal photometry quantity used by the multi-band
    model pipeline. Any extinction or magnitude conversion happens later.
    """
    nu_obs_hz = np.array([mono_effective_frequency(filter_map[str(b)]) for b in bands], float)
    return np.asarray(
        sed.fnu(
            nu_obs_hz,
            np.asarray(Teff_K, float),
            np.asarray(R_cm, float),
            DL_cm=DL_cm,
            z=z,
        ),
        float,
    )


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
    return evaluate_mono_filter_fnu_grid(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=Teff_K,
        R_cm=R_cm,
        DL_cm=DL_cm,
        z=z,
    )
