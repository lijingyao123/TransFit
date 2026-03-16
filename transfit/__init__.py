from importlib import import_module

from .api import (
    Distance, Context,
    BolometricLC, MultiBandLC,
    BolometricData, MultiBandData,
    lightcurve_bol, lightcurve_multiband,
    predict_bol, predict_multiband,
    fit_bol, fit_multiband,
)
from .modules.io import save, load, default_outpath


def __getattr__(name):
    if name == "plot":
        # Plotting stays optional until the user actually asks for it.
        return import_module(".modules.plot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BolometricData", "MultiBandData",
    "fit_bol", "fit_multiband",
    "save", "load", "default_outpath",
    "plot",
]
