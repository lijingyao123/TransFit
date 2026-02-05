# transfit/priors/common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class UniformBoundsPrior:
    bounds: np.ndarray
    param_names: Sequence[str]

    def __post_init__(self):
        b = np.asarray(self.bounds, float)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError("bounds must be shape (ndim, 2)")
        if len(self.param_names) != b.shape[0]:
            raise ValueError("param_names length must match bounds ndim")

    def lnprior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, float)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        ok = np.all((theta > lo) & (theta < hi))
        return 0.0 if ok else -np.inf

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return rng.uniform(lo, hi, size=(int(n), self.bounds.shape[0]))


def apply_user_bounds(names, bounds: np.ndarray, priors: Optional[Dict[str, Tuple[float, float]]] = None):
    if not priors:
        return np.asarray(bounds, float)

    b = np.asarray(bounds, float).copy()
    idx = {n: i for i, n in enumerate(names)}
    for k, (lo, hi) in priors.items():
        if k not in idx:
            raise KeyError(f"Unknown prior key '{k}'. Allowed: {list(names)}")
        lo, hi = float(lo), float(hi)
        if not (lo < hi):
            raise ValueError(f"Invalid bounds for '{k}': ({lo},{hi})")
        b[idx[k], 0] = lo
        b[idx[k], 1] = hi
    return b
