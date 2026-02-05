# transfit/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class MultiBandData:
    """
    Standard multi-band observation container.

    Parameters
    ----------
    t_days : array
        Observer-frame time in days (already relative to some t0 chosen by user).
    band : array
        Band labels (e.g. "g","r","b","v"...), same length as t_days.
    y : array
        Observed values. If ctx.y_kind == "mag", y is magnitude (AB by default).
        If ctx.y_kind == "flux", y is flux density Fnu (cgs).
    yerr : array
        1-sigma uncertainties, same length as y.
    mask : array, optional
        Boolean mask of same length; if provided, only masked-in points are used.
    """
    t_days: np.ndarray
    band: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        t = np.asarray(self.t_days, float)
        b = np.asarray(self.band)
        y = np.asarray(self.y, float)
        e = np.asarray(self.yerr, float)

        if not (t.ndim == b.ndim == y.ndim == e.ndim == 1):
            raise ValueError("t_days/band/y/yerr must be 1D arrays.")
        n = t.size
        if not (b.size == y.size == e.size == n):
            raise ValueError("t_days/band/y/yerr must have the same length.")

        if self.mask is not None:
            m = np.asarray(self.mask, bool)
            if m.shape != (n,):
                raise ValueError("mask must have shape (N,).")

        # store normalized arrays back (frozen dataclass -> use object.__setattr__)
        object.__setattr__(self, "t_days", t)
        object.__setattr__(self, "band", b)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "yerr", e)

    def filtered(self) -> "MultiBandData":
        """
        Return a new MultiBandData with invalid points removed:
        finite t/y/yerr and yerr>0, plus optional mask.
        """
        good = (
            np.isfinite(self.t_days)
            & np.isfinite(self.y)
            & np.isfinite(self.yerr)
            & (self.yerr > 0)
        )
        if self.mask is not None:
            good &= np.asarray(self.mask, bool)

        return MultiBandData(
            t_days=self.t_days[good],
            band=self.band[good],
            y=self.y[good],
            yerr=self.yerr[good],
            mask=None,
        )

    @property
    def bands(self):
        """Sorted unique band labels present in the data."""
        return sorted(set(self.band.tolist()))

@dataclass(frozen=True)
class BolometricData:
    """
    Bolometric observation container.

    t_days : observer-frame days
    y     : Lbol (cgs) or other bolometric observable
    yerr  : 1-sigma errors
    """
    t_days: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        t = np.asarray(self.t_days, float)
        y = np.asarray(self.y, float)
        e = np.asarray(self.yerr, float)

        if not (t.ndim == y.ndim == e.ndim == 1):
            raise ValueError("t_days/y/yerr must be 1D arrays.")
        if not (t.size == y.size == e.size):
            raise ValueError("t_days/y/yerr must have the same length.")

        if self.mask is not None:
            m = np.asarray(self.mask, bool)
            if m.shape != (t.size,):
                raise ValueError("mask must have shape (N,).")

        object.__setattr__(self, "t_days", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "yerr", e)

    def filtered(self) -> "BolometricData":
        good = np.isfinite(self.t_days) & np.isfinite(self.y) & np.isfinite(self.yerr) & (self.yerr > 0)
        if self.mask is not None:
            good &= np.asarray(self.mask, bool)
        return BolometricData(self.t_days[good], self.y[good], self.yerr[good], None)
