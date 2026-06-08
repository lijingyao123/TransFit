from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from ..filters import FilterProfile, mono_effective_wavelength_A
from ..labels import normalize_band_label
from .core import ExtinctionSpec
from .laws import component_extinction_mag


def resolve_extinction_values_mag(
    extinction: ExtinctionSpec | None,
    *,
    filter_map: Mapping[str, FilterProfile],
    used_bands: Iterable[str],
    z: float = 0.0,
) -> Dict[str, float]:
    bands = [normalize_band_label(b) for b in used_bands]
    out = {band: 0.0 for band in bands}
    if extinction is None:
        return out

    if extinction.band_map is not None:
        for band, value in extinction.band_map.values_mag.items():
            if band in out:
                out[band] += float(value)

    if not extinction.components:
        return out

    z_n = float(z)
    if not np.isfinite(z_n):
        raise ValueError("z must be finite when resolving extinction components.")
    if 1.0 + z_n <= 0.0:
        raise ValueError("z must satisfy 1 + z > 0 when resolving extinction components.")

    lambda_obs_A = {
        band: float(mono_effective_wavelength_A(filter_map[band]))
        for band in bands
    }

    for comp in extinction.components:
        for band in bands:
            lam_A = lambda_obs_A[band]
            if comp.frame == "rest":
                lam_A = lam_A / (1.0 + z_n)
            out[band] += float(component_extinction_mag(comp, lambda_um=lam_A * 1.0e-4))
    return out
