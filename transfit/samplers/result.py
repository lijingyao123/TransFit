# transfit/samplers/result.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass(frozen=True)
class FitResult:
    """
    Minimal, stable fit result container for v0.x.

    Notes:
    - ctx 用 Any 避免循环导入（api.Context -> samplers -> api）
    """
    model: str
    ctx: Any
    sampler: str

    # sampling vector (free params only)
    param_names: List[str]
    fixed: Dict[str, float]

    # full parameter order (includes fixed + optional t_shift_days)
    all_param_names: List[str]

    # flattened samples
    samples: np.ndarray        # (Ns, ndim)
    log_prob: np.ndarray       # (Ns,)

    meta: Dict[str, Any]

    def median(self) -> Dict[str, float]:
        """Median of posterior (free params), merged with fixed."""
        med = np.median(self.samples, axis=0)
        out = dict(self.fixed)
        out.update({k: float(v) for k, v in zip(self.param_names, med)})
        return out

    def best(self) -> Dict[str, float]:
        """MAP-like: pick sample with max log_prob."""
        i = int(np.argmax(self.log_prob))
        best = self.samples[i]
        out = dict(self.fixed)
        out.update({k: float(v) for k, v in zip(self.param_names, best)})
        return out
