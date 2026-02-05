# modules/interp.py 里追加这个函数即可

from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Union

FillMode = Literal["edge", "nan", "raise"]
YScale = Literal["linear", "log10"]


def interp_fit(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    xq: np.ndarray,
    *,
    yscale: YScale = "linear",
    fill: FillMode = "edge",
    floor: float = 1e-300,
) -> np.ndarray:
    """
    Fitting-friendly 1D interpolator.

    Parameters
    ----------
    x_grid : (Nt,) strictly increasing
        Grid x values.
    y_grid : (Nt,) or (Nb, Nt)
        Values on grid. If 2D, last axis must match x_grid.
    xq : (N,) or scalar
        Query x values.
    yscale : {"linear","log10"}
        Interpolate in linear space or log10(y) space (requires y>0).
        Recommended:
          - Lbol, Fnu: yscale="log10"
          - Teff, Rph, mag: yscale="linear"
    fill : {"edge","nan","raise"}
        Out-of-range handling:
          - "edge": clamp to edges
          - "nan": return NaN out of range
          - "raise": raise ValueError out of range
    floor : float
        Minimum positive floor for log10 interpolation.

    Returns
    -------
    yq : (N,) or (Nb, N)
        Interpolated values aligned with xq. Keeps leading dimensions of y_grid.
    """
    x = np.asarray(x_grid, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x_grid must be 1D, got shape={x.shape}")
    if np.any(~np.isfinite(x)):
        raise ValueError("x_grid contains non-finite values.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x_grid must be strictly increasing.")

    xq = np.asarray(xq, dtype=float)

    y = np.asarray(y_grid, dtype=float)
    if y.ndim == 1:
        y2 = y[None, :]  # (1, Nt) unify
        squeeze_out = True
    elif y.ndim == 2:
        y2 = y
        squeeze_out = False
    else:
        raise ValueError(f"y_grid must be 1D or 2D, got shape={y.shape}")

    if y2.shape[-1] != x.shape[0]:
        raise ValueError(f"y_grid last axis must match x_grid: {y2.shape[-1]} vs {x.shape[0]}")

    # out-of-range policy
    if fill == "raise":
        if np.any((xq < x[0]) | (xq > x[-1])):
            raise ValueError("xq contains values outside interpolation range and fill='raise'.")

    # transform for log-scale if requested
    if yscale == "log10":
        y_work = np.clip(y2, floor, None)
        y_work = np.log10(y_work)
    elif yscale == "linear":
        y_work = y2
    else:
        raise ValueError(f"Unknown yscale={yscale}")

    # choose left/right fill values for np.interp
    if fill == "edge":
        left = y_work[:, 0]
        right = y_work[:, -1]
    else:  # "nan"
        left = np.full((y_work.shape[0],), np.nan, dtype=float)
        right = np.full((y_work.shape[0],), np.nan, dtype=float)

    # vectorized over leading axis by looping bands (Nb small typically)
    out = np.empty((y_work.shape[0],) + xq.shape, dtype=float)
    for i in range(y_work.shape[0]):
        out[i] = np.interp(xq, x, y_work[i], left=left[i], right=right[i])

    # inverse transform
    if yscale == "log10":
        out = np.power(10.0, out)

    if squeeze_out:
        return out[0]
    return out
