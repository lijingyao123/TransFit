# transfit/samplers/__init__.py
from .result import FitResult


def run_emcee(*args, **kwargs):
    # Defer backend import so `import transfit` does not require optional deps.
    from .emcee import run_emcee as _run_emcee

    return _run_emcee(*args, **kwargs)


def run_zeus(*args, **kwargs):
    # Defer backend import so `import transfit` does not require optional deps.
    from .zeus import run_zeus as _run_zeus

    return _run_zeus(*args, **kwargs)


def run_dynesty(*args, **kwargs):
    # Defer backend import so `import transfit` does not require optional deps.
    from .dynesty import run_dynesty as _run_dynesty

    return _run_dynesty(*args, **kwargs)


__all__ = [
    "FitResult",
    "run_emcee",
    "run_zeus",
    "run_dynesty",
]
