from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transfit.constants import C_LIGHT

from .blackbody import BlackbodySED


@dataclass(frozen=True, init=False)
class CutoffBlackbodySED(BlackbodySED):
    """
    Blackbody SED with a short-wavelength power-law cutoff.

    The cutoff is applied in the rest frame:

      C(lambda) = 1                         for lambda >= lambda_cut
      C(lambda) = (lambda / lambda_cut)^a   for lambda < lambda_cut

    The class keeps the same public interface as BlackbodySED, so it can be
    passed anywhere TransFit accepts ``sed=...``.
    """

    cutoff_wavelength_A: float = 3000.0
    uv_slope: float = 2.0
    min_factor: float = 0.0

    def __init__(
        self,
        cutoff_wavelength_A: float = 3000.0,
        uv_slope: float = 2.0,
        min_factor: float = 0.0,
        *,
        Tmin: float = 1.0,
        Rmin: float = 1.0,
    ) -> None:
        object.__setattr__(self, "Tmin", float(Tmin))
        object.__setattr__(self, "Rmin", float(Rmin))
        object.__setattr__(self, "cutoff_wavelength_A", float(cutoff_wavelength_A))
        object.__setattr__(self, "uv_slope", float(uv_slope))
        object.__setattr__(self, "min_factor", float(min_factor))
        self._validate()

    def _validate(self) -> None:
        if not np.isfinite(self.cutoff_wavelength_A) or self.cutoff_wavelength_A <= 0.0:
            raise ValueError("cutoff_wavelength_A must be finite and positive.")
        if not np.isfinite(self.uv_slope) or self.uv_slope < 0.0:
            raise ValueError("uv_slope must be finite and non-negative.")
        if not np.isfinite(self.min_factor) or not (0.0 <= self.min_factor <= 1.0):
            raise ValueError("min_factor must be finite and satisfy 0 <= min_factor <= 1.")

    def cutoff_factor(self, nu_rest_hz: np.ndarray) -> np.ndarray:
        """
        Multiplicative cutoff factor, shape (Nnu, 1).
        """
        nu = np.asarray(nu_rest_hz, dtype=float).reshape(-1, 1)
        good = np.isfinite(nu) & (nu > 0.0)

        lambda_A = np.full_like(nu, np.nan, dtype=float)
        lambda_A[good] = C_LIGHT / nu[good] * 1.0e8

        factor = np.ones_like(lambda_A, dtype=float)
        blue = good & (lambda_A < self.cutoff_wavelength_A)
        factor[blue] = (lambda_A[blue] / self.cutoff_wavelength_A) ** self.uv_slope
        factor = np.where(good, factor, np.nan)

        if self.min_factor > 0.0:
            factor = np.maximum(factor, self.min_factor)
        return factor

    def lnu(self, nu_rest_hz: np.ndarray, Teff_K: np.ndarray, R_cm: np.ndarray) -> np.ndarray:
        """
        Cutoff luminosity density L_nu [erg/s/Hz], shape (Nnu, Nt).
        """
        lnu_bb = super().lnu(nu_rest_hz, Teff_K, R_cm)
        return lnu_bb * self.cutoff_factor(nu_rest_hz)


CutoffBlackbody = CutoffBlackbodySED
