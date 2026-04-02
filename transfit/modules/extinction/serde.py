from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional

from .core import ExtinctionSpec
from .normalize import normalize_extinction


def extinction_to_dict(extinction: Optional[ExtinctionSpec]) -> Optional[Dict[str, object]]:
    if extinction is None:
        return None

    payload: Dict[str, object] = {
        "kind": "extinction_spec",
        "band_map": None,
        "components": [],
    }

    if extinction.band_map is not None:
        payload["band_map"] = {
            "frame": str(extinction.band_map.frame),
            "values_mag": {
                str(k): float(v)
                for k, v in dict(extinction.band_map.values_mag).items()
            },
        }

    payload["components"] = [
        {
            "name": str(comp.name),
            "ebv": float(comp.ebv),
            "rv": float(comp.rv),
            "law": str(comp.law),
            "frame": str(comp.frame),
        }
        for comp in extinction.components
    ]
    return payload


def extinction_from_dict(payload: object) -> Optional[ExtinctionSpec]:
    if payload in (None, {}):
        return None
    if isinstance(payload, ExtinctionSpec):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("Serialized extinction payload must be a mapping or None.")

    kind = str(payload.get("kind", "band_map")).strip().lower()
    if kind == "band_map":
        values_mag = {
            str(k): float(v)
            for k, v in dict(payload.get("values_mag", {}) or {}).items()
        }
        return normalize_extinction(values_mag)

    if kind != "extinction_spec":
        raise NotImplementedError(
            f"Serialized extinction kind {kind!r} is not supported."
        )

    band_map_payload = payload.get("band_map", None)
    components_payload = payload.get("components", [])

    normalized: Dict[str, object] = {}
    if band_map_payload not in (None, {}):
        if not isinstance(band_map_payload, Mapping):
            raise TypeError("Serialized extinction band_map must be a mapping.")
        normalized["band_map"] = dict(band_map_payload.get("values_mag", {}) or {})
    if components_payload not in (None, []):
        if not isinstance(components_payload, list):
            raise TypeError("Serialized extinction components must be a list.")
        for comp in components_payload:
            if not isinstance(comp, Mapping):
                raise TypeError("Serialized extinction component entries must be mappings.")
            name = str(comp.get("name", "dust")).strip() or "dust"
            normalized[name] = {
                "ebv": float(comp.get("ebv")),
                "rv": float(comp.get("rv", 3.1)),
                "law": str(comp.get("law", "ccm89")),
                "frame": str(comp.get("frame", "observer")),
            }
    return normalize_extinction(normalized)
