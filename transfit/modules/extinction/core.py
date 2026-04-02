from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class BandExtinction:
    values_mag: Dict[str, float]
    frame: Literal["observer"] = "observer"

    def __post_init__(self):
        values = {
            str(k).strip(): float(v)
            for k, v in dict(self.values_mag or {}).items()
        }
        if not values:
            raise ValueError("BandExtinction requires at least one band.")
        for band, value in values.items():
            if not band:
                raise ValueError("BandExtinction band labels must be non-empty.")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Extinction for band {band!r} must be finite and >= 0.")
        frame = str(self.frame).strip().lower()
        if frame != "observer":
            raise ValueError("BandExtinction currently supports observer-frame values only.")
        object.__setattr__(self, "values_mag", values)
        object.__setattr__(self, "frame", frame)


@dataclass(frozen=True)
class DustComponent:
    name: str
    ebv: float
    rv: float = 3.1
    law: Literal["ccm89", "odonnell94"] = "ccm89"
    frame: Literal["observer", "rest"] = "observer"

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("DustComponent name must be a non-empty string.")
        ebv = float(self.ebv)
        rv = float(self.rv)
        law = str(self.law).strip().lower()
        frame = str(self.frame).strip().lower()
        if not np.isfinite(ebv) or ebv < 0.0:
            raise ValueError("DustComponent ebv must be finite and >= 0.")
        if not np.isfinite(rv) or rv <= 0.0:
            raise ValueError("DustComponent rv must be finite and > 0.")
        if law not in ("ccm89", "odonnell94"):
            raise ValueError(
                f"Unsupported extinction law {self.law!r}. "
                "Supported laws: 'ccm89', 'odonnell94'."
            )
        if frame not in ("observer", "rest"):
            raise ValueError("DustComponent frame must be 'observer' or 'rest'.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "ebv", ebv)
        object.__setattr__(self, "rv", rv)
        object.__setattr__(self, "law", law)
        object.__setattr__(self, "frame", frame)


@dataclass(frozen=True)
class ExtinctionSpec:
    band_map: BandExtinction | None = None
    components: Tuple[DustComponent, ...] = field(default_factory=tuple)

    def __post_init__(self):
        comps = tuple(self.components or ())
        for comp in comps:
            if not isinstance(comp, DustComponent):
                raise TypeError("ExtinctionSpec components must contain DustComponent objects.")
        if self.band_map is None and len(comps) == 0:
            raise ValueError("ExtinctionSpec requires either band_map or components.")
        object.__setattr__(self, "components", comps)
