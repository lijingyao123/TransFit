from __future__ import annotations

LIKELIHOOD_NUISANCE_PARAM_SPECS = {
    "sigma_int": dict(
        units="mag",
        minimum=0.0,
        likelihood="gaussian_with_sigma_int_mag",
    ),
}


__all__ = ["LIKELIHOOD_NUISANCE_PARAM_SPECS"]
