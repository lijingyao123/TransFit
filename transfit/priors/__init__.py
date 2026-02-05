# transfit/priors/__init__.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

from .common import UniformBoundsPrior, apply_user_bounds


def build_bounds(
    model: str,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    include_t_shift: bool = True,
):
    m = model.lower().strip()

    if m in ["ni", "nickel"]:
        from .nickel import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    elif m in ["iib", "ii_b", "ii-b"]:
        from .iib import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    # ✅ 新增：SCNi
    elif m in ["scni", "sc_ni", "sc-nickel", "shockcooling+ni"]:
        from .scni import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)
     # ✅ 新增：Magnetar
    elif m in ["magnetar", "mag", "mg"]:
        from .magnetar import default_names_and_bounds
        names, bounds = default_names_and_bounds(include_t_shift=include_t_shift)

    else:
        raise ValueError(f"Unknown model='{model}' for priors.build_bounds")

    bounds = apply_user_bounds(names, np.asarray(bounds, float), priors=priors)
    return names, np.asarray(bounds, float)


__all__ = ["UniformBoundsPrior", "build_bounds"]
