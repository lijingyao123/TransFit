# transfit/samplers/__init__.py
from .result import FitResult


def run_emcee(*args, **kwargs):
    # Defer backend import so `import transfit` does not require optional deps.
    try:
        from .emcee import run_emcee as _run_emcee
    except ModuleNotFoundError as exc:
        if exc.name == "emcee":
            raise ImportError(
                "emcee is required for sampler='emcee'. Install the default "
                "TransFit package with dependencies, or install emcee manually."
            ) from exc
        raise

    return _run_emcee(*args, **kwargs)


def run_zeus(*args, **kwargs):
    # Defer backend import so `import transfit` does not require optional deps.
    try:
        from .zeus import run_zeus as _run_zeus
    except ModuleNotFoundError as exc:
        if exc.name == "zeus":
            raise ImportError(
                "zeus-mcmc is required for sampler='zeus'. Install with "
                "`pip install transfit[all-samplers]` or `pip install zeus-mcmc`."
            ) from exc
        raise

    return _run_zeus(*args, **kwargs)


def run_dynesty(*args, **kwargs):
    # Defer backend import so `import transfit` does not require optional deps.
    try:
        from .dynesty import run_dynesty as _run_dynesty
    except ModuleNotFoundError as exc:
        if exc.name == "dynesty":
            raise ImportError(
                "dynesty is required for sampler='dynesty'. Install with "
                "`pip install transfit[all-samplers]` or `pip install dynesty`."
            ) from exc
        raise

    return _run_dynesty(*args, **kwargs)


__all__ = [
    "FitResult",
    "run_emcee",
    "run_zeus",
    "run_dynesty",
]
