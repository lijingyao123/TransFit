from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Iterable, Optional

import numpy as np

from transfit.constants import C_LIGHT

from ..labels import normalize_band_label
from .core import FilterProfile
from .registry import get_builtin_filter


def _coerce_zero_points(spec: Mapping[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "zero_points_jy" in spec:
        raw = spec["zero_points_jy"]
        if not isinstance(raw, Mapping):
            raise TypeError("zero_points_jy must be a mapping of system -> Jy.")
        for key, value in raw.items():
            out[str(key).strip().lower()] = float(value)
    if "vega_zero_point_jy" in spec:
        out["vega"] = float(spec["vega_zero_point_jy"])
    return out


def _normalize_one(label: str, spec: object) -> FilterProfile:
    if isinstance(spec, FilterProfile):
        if spec.label != label:
            return FilterProfile(
                label=label,
                filter_id=spec.filter_id,
                kind=spec.kind,
                source=spec.source,
                detector=spec.detector,
                nu_eff_hz=spec.nu_eff_hz,
                wavelength_A=spec.wavelength_A,
                throughput=spec.throughput,
                zero_points_jy=spec.zero_points_jy,
                meta=spec.meta,
            )
        return spec

    if np.isscalar(spec) and not isinstance(spec, (str, bytes)):
        return FilterProfile(
            label=label,
            filter_id=f"legacy:{label}",
            kind="mono",
            source="legacy",
            nu_eff_hz=float(spec),
        )

    if isinstance(spec, str):
        return get_builtin_filter(label=label, filter_id=spec)

    if not isinstance(spec, Mapping):
        raise TypeError(
            f"Unsupported filter spec for band {label!r}: {type(spec).__name__}. "
            "Use a float, preset id string, or mapping."
        )

    if any(k in spec for k in ("filter_id", "id", "preset")):
        filter_id = spec.get("filter_id", spec.get("id", spec.get("preset")))
        prof = get_builtin_filter(label=label, filter_id=str(filter_id))
        zero_points_jy = dict(prof.zero_points_jy)
        zero_points_jy.update(_coerce_zero_points(spec))
        meta = dict(prof.meta)
        meta.update(dict(spec.get("meta", {}) or {}))
        return FilterProfile(
            label=label,
            filter_id=prof.filter_id,
            kind=prof.kind,
            source=prof.source,
            detector=prof.detector,
            nu_eff_hz=prof.nu_eff_hz,
            wavelength_A=prof.wavelength_A,
            throughput=prof.throughput,
            zero_points_jy=zero_points_jy,
            meta=meta,
        )

    if "nu_eff_hz" in spec or "nu_eff" in spec:
        return FilterProfile(
            label=label,
            filter_id=str(spec.get("filter_id", f"user:{label}")),
            kind="mono",
            source="user",
            detector=str(spec.get("detector", "energy")),
            nu_eff_hz=float(spec.get("nu_eff_hz", spec.get("nu_eff"))),
            zero_points_jy=_coerce_zero_points(spec),
            meta=dict(spec.get("meta", {}) or {}),
        )

    if "wavelength_A" in spec or "throughput" in spec:
        raise NotImplementedError(
            "Bandpass filter curves are not implemented yet. "
            "Phase 1 supports mono-frequency filters only."
        )

    raise ValueError(
        f"Could not interpret filter spec for band {label!r}. "
        "Use a float, preset id string, or mapping with nu_eff_hz."
    )


def _require_used_bands(filters: Mapping[str, FilterProfile], used_bands: Iterable[str]) -> None:
    needed = [normalize_band_label(b) for b in used_bands]
    missing = [b for b in needed if b not in filters]
    if missing:
        raise KeyError(
            f"These bands are missing in filters (case-sensitive): {missing}. "
            f"Available: {sorted(filters.keys())}"
        )


def _require_mag_system(filters: Mapping[str, FilterProfile], used_bands: Iterable[str], mag_system: str) -> None:
    system = str(mag_system).strip().lower()
    if system not in ("ab", "vega"):
        raise ValueError("mag_system must be 'ab' or 'vega'.")
    if system != "vega":
        return
    missing = [
        b for b in [normalize_band_label(x) for x in used_bands]
        if "vega" not in filters[b].zero_points_jy
    ]
    if missing:
        raise ValueError(
            "Vega magnitudes require a Vega zero point for every used band. "
            f"Missing Vega zero points for: {missing}"
        )


def normalize_filters(
    filters: Mapping[str, object],
    *,
    used_bands: Optional[Iterable[str]] = None,
    mag_system: str = "ab",
) -> Dict[str, FilterProfile]:
    if not isinstance(filters, Mapping) or len(filters) == 0:
        raise ValueError("filters must be a non-empty mapping.")

    out: Dict[str, FilterProfile] = {}
    for raw_label, spec in filters.items():
        label = normalize_band_label(raw_label)
        out[label] = _normalize_one(label, spec)

    bands_for_check = list(out.keys()) if used_bands is None else [normalize_band_label(b) for b in used_bands]
    _require_used_bands(out, bands_for_check)
    _require_mag_system(out, bands_for_check, mag_system)
    return out


def validate_filter_map(
    filters: Mapping[str, object],
    *,
    used_bands: Iterable[str],
    mag_system: str = "ab",
) -> Dict[str, FilterProfile]:
    return normalize_filters(filters, used_bands=used_bands, mag_system=mag_system)


def mono_effective_frequency(profile: FilterProfile) -> float:
    if profile.kind != "mono" or profile.nu_eff_hz is None:
        raise NotImplementedError(
            f"Filter {profile.label!r} is not a mono-frequency filter. "
            "Bandpass photometry is not implemented yet."
        )
    return float(profile.nu_eff_hz)


def mono_effective_wavelength_A(profile: FilterProfile) -> float:
    nu_eff_hz = mono_effective_frequency(profile)
    return (C_LIGHT / nu_eff_hz) * 1.0e8
