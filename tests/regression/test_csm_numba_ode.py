from __future__ import annotations

import numpy as np
import pytest

from transfit.models.csm import CSMModel


CSM_THETA = (5.0, 1.0, 1.0, 3000.0, 0.2, 2.0, 0.5, 4500.0)


def _interp_log_luminosity(t_ref, t_model, l_model):
    return 10.0 ** np.interp(
        np.asarray(t_ref, dtype=float),
        np.asarray(t_model, dtype=float),
        np.log10(np.maximum(np.asarray(l_model, dtype=float), 1.0e-300)),
    )


def test_csm_numba_shock_ode_matches_scipy_reference_light_curve():
    model = CSMModel()
    kwargs = dict(Nx=80, Ny=500, t_max_days=120.0)

    t_ref, l_ref, t_eff_ref, r_ph_ref = model.calculate_light_curve(
        CSM_THETA,
        shock_ode_solver="scipy",
        **kwargs,
    )
    t_new, l_new, t_eff_new, r_ph_new = model.calculate_light_curve(
        CSM_THETA,
        shock_ode_solver="numba",
        **kwargs,
    )

    assert np.all(np.diff(t_new) > 0.0)
    assert np.all(np.isfinite(l_new))
    assert np.all(np.isfinite(t_eff_new))
    assert np.all(np.isfinite(r_ph_new))

    l_on_ref = _interp_log_luminosity(t_ref, t_new, l_new)
    valid = (l_ref > 1.0e-200) & np.isfinite(l_on_ref)
    rel_l = np.abs(l_on_ref[valid] - l_ref[valid]) / np.maximum(np.abs(l_ref[valid]), 1.0e-300)
    rms_log_l = np.sqrt(
        np.mean(
            (
                np.log10(np.maximum(l_on_ref[valid], 1.0e-300))
                - np.log10(np.maximum(l_ref[valid], 1.0e-300))
            )
            ** 2
        )
    )

    t_eff_on_ref = np.interp(t_ref, t_new, t_eff_new)
    r_ph_on_ref = np.interp(t_ref, t_new, r_ph_new)

    assert float(np.max(rel_l)) < 2.0e-4
    assert float(np.percentile(rel_l, 95.0)) < 1.0e-4
    assert float(rms_log_l) < 5.0e-5
    np.testing.assert_allclose(t_eff_on_ref, t_eff_ref, rtol=3.0e-4, atol=0.0)
    np.testing.assert_allclose(r_ph_on_ref, r_ph_ref, rtol=3.0e-4, atol=0.0)


@pytest.mark.parametrize("r_csm_out", [1000.0, 3000.0, 10000.0, 30000.0, 100000.0])
def test_csm_numba_shock_ode_matches_scipy_across_radius_range(r_csm_out):
    model = CSMModel()
    theta = (5.0, 1.0, 1.0, r_csm_out, 0.2, 2.0, 0.5, 4500.0)
    kwargs = dict(Nx=80, Ny=500, t_max_days=120.0)

    t_ref, l_ref, _, _ = model.calculate_light_curve(theta, shock_ode_solver="scipy", **kwargs)
    t_new, l_new, _, _ = model.calculate_light_curve(theta, shock_ode_solver="numba", **kwargs)

    l_on_ref = _interp_log_luminosity(t_ref, t_new, l_new)
    valid = (l_ref > 1.0e-200) & np.isfinite(l_on_ref)
    rel_l = np.abs(l_on_ref[valid] - l_ref[valid]) / np.maximum(np.abs(l_ref[valid]), 1.0e-300)

    assert np.all(np.diff(t_new) > 0.0)
    assert float(np.max(rel_l)) < 2.0e-4
    assert float(np.percentile(rel_l, 95.0)) < 1.0e-4
