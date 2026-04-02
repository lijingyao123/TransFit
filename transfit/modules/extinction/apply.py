from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..filters import FilterProfile
from .core import ExtinctionSpec
from .resolve import resolve_extinction_values_mag


def apply_extinction_to_fnu_grid(
    fnu_grid: np.ndarray,
    *,
    filter_map: Dict[str, FilterProfile],
    bands: Sequence[str],
    extinction: ExtinctionSpec | None,
    z: float,
) -> np.ndarray:
    out = np.asarray(fnu_grid, float).copy()
    if extinction is None:
        return out
    values_mag = resolve_extinction_values_mag(
        extinction,
        filter_map=filter_map,
        used_bands=bands,
        z=z,
    )
    for i, band in enumerate(bands):
        a_band = float(values_mag[str(band)])
        out[i] *= 10.0 ** (-0.4 * a_band)
    return out
