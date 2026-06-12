from __future__ import annotations

from typing import Any, Dict

from .blackbody import BlackbodySED
from .cutoff_blackbody import CutoffBlackbodySED


def sed_to_dict(sed) -> Dict[str, Any]:
    """
    JSON-serializable SED configuration for built-in TransFit SEDs.

    Unknown custom SED objects are recorded by name only and cannot be
    reconstructed automatically from saved fit metadata.
    """
    if type(sed) is CutoffBlackbodySED:
        return {
            "name": "CutoffBlackbodySED",
            "builtin": True,
            "params": {
                "cutoff_wavelength_A": float(sed.cutoff_wavelength_A),
                "uv_slope": float(sed.uv_slope),
                "min_factor": float(sed.min_factor),
                "Tmin": float(sed.Tmin),
                "Rmin": float(sed.Rmin),
            },
        }
    if type(sed) is BlackbodySED:
        return {
            "name": "BlackbodySED",
            "builtin": True,
            "params": {
                "Tmin": float(sed.Tmin),
                "Rmin": float(sed.Rmin),
            },
        }
    return {
        "name": sed.__class__.__name__,
        "builtin": False,
        "params": None,
    }


def sed_from_dict(config: Dict[str, Any] | None):
    """
    Rebuild a built-in SED object from ``sed_to_dict`` output.
    """
    if config is None:
        return BlackbodySED()
    cfg = dict(config or {})
    name = str(cfg.get("name", "BlackbodySED"))
    params = dict(cfg.get("params") or {})

    if name in {"BlackbodySED", "Blackbody"}:
        return BlackbodySED(**params)
    if name in {"CutoffBlackbodySED", "CutoffBlackbody"}:
        return CutoffBlackbodySED(**params)

    raise ValueError(
        f"Cannot reconstruct SED '{name}' from saved metadata. "
        "Pass the original SED object with sed=... when plotting."
    )
