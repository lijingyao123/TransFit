# transfit/modules/__init__.py
from importlib import import_module

from .io import save, load, default_outpath


def __getattr__(name):
    if name == "plot":
        # Keep plotting optional at import time.
        return import_module(".plot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["save", "load", "default_outpath", "plot"]
