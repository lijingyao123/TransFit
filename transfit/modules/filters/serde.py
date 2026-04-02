from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

from .core import FilterProfile
from .normalize import normalize_filters


def filter_profile_to_dict(profile: FilterProfile) -> Dict[str, object]:
    out: Dict[str, object] = {
        "label": profile.label,
        "filter_id": profile.filter_id,
        "kind": profile.kind,
        "source": profile.source,
        "detector": profile.detector,
        "zero_points_jy": dict(profile.zero_points_jy),
        "meta": dict(profile.meta),
    }
    if profile.nu_eff_hz is not None:
        out["nu_eff_hz"] = float(profile.nu_eff_hz)
    if profile.wavelength_A is not None:
        out["wavelength_A"] = profile.wavelength_A.tolist()
    if profile.throughput is not None:
        out["throughput"] = profile.throughput.tolist()
    return out


def filters_to_dict(filters: Mapping[str, FilterProfile]) -> Dict[str, Dict[str, object]]:
    return {str(label): filter_profile_to_dict(profile) for label, profile in dict(filters or {}).items()}


def filter_profile_from_dict(label: str, payload: Mapping[str, object]) -> FilterProfile:
    return FilterProfile(
        label=str(payload.get("label", label)),
        filter_id=str(payload.get("filter_id", payload.get("id", f"user:{label}"))),
        kind=str(payload.get("kind", "mono")),
        source=str(payload.get("source", "user")),
        detector=str(payload.get("detector", "energy")),
        nu_eff_hz=payload.get("nu_eff_hz"),
        wavelength_A=payload.get("wavelength_A"),
        throughput=payload.get("throughput"),
        zero_points_jy=dict(payload.get("zero_points_jy", {}) or {}),
        meta=dict(payload.get("meta", {}) or {}),
    )


def filters_from_dict(payload: Mapping[str, object]) -> Dict[str, FilterProfile]:
    if not payload:
        return {}
    if all(not isinstance(v, Mapping) for v in payload.values()):
        return normalize_filters(payload)
    return {
        str(label): filter_profile_from_dict(str(label), dict(spec))
        for label, spec in payload.items()
    }
