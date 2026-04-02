from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Optional

from ..filters import FilterProfile
from .core import BandExtinction, DustComponent, ExtinctionSpec
from .resolve import resolve_extinction_values_mag


_EBV_KEYS = {"ebv", "e_bv", "e(b-v)"}
_AV_KEYS = {"av", "a_v"}
_RV_KEYS = {"rv", "r_v"}
_LAW_KEYS = {"law"}
_FRAME_KEYS = {"frame"}
_NAME_KEYS = {"name"}
_COMPONENT_KEYS = _EBV_KEYS | _AV_KEYS | _RV_KEYS | _LAW_KEYS | _FRAME_KEYS | _NAME_KEYS
_BAND_MAP_KEYS = {"band_map", "values_mag"}


def _norm_band(label: object) -> str:
    out = str(label).strip()
    if not out:
        raise ValueError("Band labels must be non-empty strings.")
    return out


def _norm_key(key: object) -> str:
    return str(key).strip().lower()


def _band_map_from_mapping(payload: Mapping[str, object]) -> BandExtinction:
    return BandExtinction(
        values_mag={
            _norm_band(k): float(v)
            for k, v in dict(payload).items()
        },
        frame="observer",
    )


def _default_frame_for_name(name: str) -> str:
    key = str(name).strip().lower()
    if key in ("host", "rest", "restframe", "rest_frame"):
        return "rest"
    return "observer"


def _dust_component_from_mapping(
    name: str,
    payload: Mapping[str, object],
    *,
    default_frame: str,
) -> DustComponent:
    norm = {_norm_key(k): v for k, v in dict(payload).items()}
    ebv_keys = [k for k in _EBV_KEYS if k in norm]
    av_keys = [k for k in _AV_KEYS if k in norm]
    if bool(ebv_keys) == bool(av_keys):
        raise ValueError(
            f"Extinction component {name!r} must provide exactly one of "
            "`ebv` or `av`."
        )

    rv = float(norm.get("rv", norm.get("r_v", 3.1)))
    law = str(norm.get("law", "ccm89")).strip().lower()
    frame = str(norm.get("frame", default_frame)).strip().lower()
    comp_name = str(norm.get("name", name)).strip() or str(name).strip()

    if ebv_keys:
        ebv = float(norm[ebv_keys[0]])
    else:
        ebv = float(norm[av_keys[0]]) / rv

    return DustComponent(
        name=comp_name,
        ebv=ebv,
        rv=rv,
        law=law,
        frame=frame,
    )


def _is_single_component_mapping(payload: Mapping[str, object]) -> bool:
    keys = {_norm_key(k) for k in payload.keys()}
    return any(k in _COMPONENT_KEYS for k in keys)


def normalize_extinction(
    extinction: object,
    *,
    used_bands: Optional[Iterable[str]] = None,
    filter_map: Optional[Mapping[str, FilterProfile]] = None,
    z: float = 0.0,
) -> Optional[ExtinctionSpec]:
    if extinction is None:
        return None
    if isinstance(extinction, ExtinctionSpec):
        spec = extinction
    else:
        if not isinstance(extinction, Mapping) or len(extinction) == 0:
            raise ValueError("extinction must be None or a non-empty mapping.")

        payload = dict(extinction)
        if _is_single_component_mapping(payload):
            spec = ExtinctionSpec(
                components=(
                    _dust_component_from_mapping(
                        "dust",
                        payload,
                        default_frame="observer",
                    ),
                ),
            )
        elif all(not isinstance(v, Mapping) for v in payload.values()):
            spec = ExtinctionSpec(band_map=_band_map_from_mapping(payload))
        else:
            band_map = None
            components = []
            for raw_key, value in payload.items():
                key = _norm_key(raw_key)
                if key in _BAND_MAP_KEYS:
                    if not isinstance(value, Mapping) or len(value) == 0:
                        raise ValueError(
                            f"extinction[{raw_key!r}] must be a non-empty mapping of band -> A_band."
                        )
                    band_map = _band_map_from_mapping(value)
                    continue
                if not isinstance(value, Mapping):
                    raise ValueError(
                        "Mixed extinction inputs must use a nested `band_map={...}` for direct A_band values. "
                        f"Top-level key {raw_key!r} is scalar and ambiguous."
                    )
                components.append(
                    _dust_component_from_mapping(
                        str(raw_key),
                        value,
                        default_frame=_default_frame_for_name(str(raw_key)),
                    )
                )
            spec = ExtinctionSpec(
                band_map=band_map,
                components=tuple(components),
            )

    if used_bands is not None:
        needed = [_norm_band(b) for b in used_bands]
        missing_from_map = []
        if spec.band_map is not None:
            missing_from_map = [b for b in needed if b not in spec.band_map.values_mag]

        if not spec.components:
            if missing_from_map:
                raise KeyError(
                    "Per-band extinction must include every used band. "
                    f"Missing bands: {missing_from_map}"
                )
        else:
            if filter_map is None:
                raise ValueError(
                    "Structured extinction components require filter_map during validation."
                )
            missing_filters = [b for b in needed if b not in filter_map]
            if missing_filters:
                raise KeyError(
                    "Structured extinction components need filters for every used band. "
                    f"Missing filters: {missing_filters}"
                )
            resolve_extinction_values_mag(
                spec,
                filter_map=filter_map,
                used_bands=needed,
                z=z,
            )
    return spec


def validate_extinction_spec(
    extinction: object,
    *,
    used_bands: Iterable[str],
    filter_map: Optional[Mapping[str, FilterProfile]] = None,
    z: float = 0.0,
) -> Optional[ExtinctionSpec]:
    return normalize_extinction(
        extinction,
        used_bands=used_bands,
        filter_map=filter_map,
        z=z,
    )
