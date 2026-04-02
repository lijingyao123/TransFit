from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class FilterProfile:
    label: str
    filter_id: str
    kind: Literal["mono", "bandpass"]
    source: Literal["builtin", "user", "legacy"]
    detector: Literal["energy", "photon"] = "energy"
    nu_eff_hz: Optional[float] = None
    wavelength_A: Optional[np.ndarray] = None
    throughput: Optional[np.ndarray] = None
    zero_points_jy: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        label = str(self.label).strip()
        if not label:
            raise ValueError("Filter label must be a non-empty string.")
        filter_id = str(self.filter_id).strip()
        if not filter_id:
            raise ValueError("filter_id must be a non-empty string.")

        kind = str(self.kind).strip().lower()
        if kind not in ("mono", "bandpass"):
            raise ValueError(f"Unknown filter kind {self.kind!r}.")

        source = str(self.source).strip().lower()
        if source not in ("builtin", "user", "legacy"):
            raise ValueError(f"Unknown filter source {self.source!r}.")

        detector = str(self.detector).strip().lower()
        if detector not in ("energy", "photon"):
            raise ValueError(f"Unknown detector type {self.detector!r}.")

        nu_eff_hz = None if self.nu_eff_hz is None else float(self.nu_eff_hz)
        if kind == "mono":
            if nu_eff_hz is None or not np.isfinite(nu_eff_hz) or nu_eff_hz <= 0.0:
                raise ValueError("Mono filters require a positive finite nu_eff_hz.")

        wavelength_A = None
        throughput = None
        if self.wavelength_A is not None or self.throughput is not None:
            wavelength_A = np.asarray(self.wavelength_A, float).reshape(-1)
            throughput = np.asarray(self.throughput, float).reshape(-1)
            if wavelength_A.size == 0 or throughput.size == 0:
                raise ValueError("Bandpass arrays must be non-empty.")
            if wavelength_A.shape != throughput.shape:
                raise ValueError("wavelength_A and throughput must have the same shape.")
            if np.any(~np.isfinite(wavelength_A)) or np.any(~np.isfinite(throughput)):
                raise ValueError("Bandpass arrays must be finite.")
            if np.any(np.diff(wavelength_A) <= 0.0):
                raise ValueError("wavelength_A must be strictly increasing.")
            if np.any(throughput < 0.0) or not np.any(throughput > 0.0):
                raise ValueError("throughput must be non-negative and contain at least one positive value.")

        zero_points_jy = {
            str(k).strip().lower(): float(v)
            for k, v in dict(self.zero_points_jy or {}).items()
        }
        for key, value in zero_points_jy.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Zero point {key!r} must be positive and finite.")

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "filter_id", filter_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "nu_eff_hz", nu_eff_hz)
        object.__setattr__(self, "wavelength_A", wavelength_A)
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "zero_points_jy", zero_points_jy)
        object.__setattr__(self, "meta", dict(self.meta or {}))
