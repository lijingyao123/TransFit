from __future__ import annotations

import inspect
import numpy as np
import pytest

import transfit as tf
from transfit.api import _physical_constraints_lnprior
from transfit.constants import MPC
from transfit.modules.extinction import (
    apply_extinction_to_fnu_grid,
    extinction_from_dict,
    extinction_to_dict,
    normalize_extinction,
    resolve_extinction_values_mag,
)
from transfit.modules.filters import normalize_filters
from transfit.modules.fnu import evaluate_multiband_model_fnu
from transfit.modules.io import _validate_ctx_dict
from transfit.modules.likelihood import (
    gaussian_lnlike_flux,
    gaussian_lnlike_for_observation,
    gaussian_lnlike_mag,
)
from transfit.modules.magnitudes import fnu_grid_to_vega_mag_grid
from transfit.modules.photometry import evaluate_multiband_observer_output
from transfit.modules.sed import BlackbodySED
from transfit.priors import MixedBoundsPrior
from transfit.samplers import run_emcee, run_zeus

MU_7P5_MPC = 29.37530631695874


PARAMS_NI = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 1.5,
    "M_Ni": 0.08,
    "R_0": 120.0,
    "x_Ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 3000.0,
}

LEGACY_PARAMS_NI = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "M_Ni": 0.08,
    "x_Ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 3000.0,
}


def test_public_nickel_and_magnetar_use_canonical_full_parameter_sets():
    nickel_names = tf.model_param_names("nickel")
    magnetar_names = tf.model_param_names("magnetar")

    assert nickel_names == ["M_ej", "v_ej", "E_Th_in", "M_Ni", "R_0", "x_Ni", "kappa", "kappa_gamma", "T_floor"]
    assert magnetar_names == ["M_ej", "v_ej", "E_Th_in", "P_ms", "B14", "R_0", "kappa", "kappa_gamma", "T_floor"]


def test_removed_sc_alias_is_rejected():
    with pytest.raises(ValueError):
        tf.model_param_names("sc_ni")

    with pytest.raises(ValueError):
        tf.model_param_names("sc_magnetar")


def test_negative_redshift_is_rejected_by_public_forward_apis():
    with pytest.raises(ValueError, match="z must be non-negative"):
        tf.lightcurve_bol(
            model="nickel",
            params=LEGACY_PARAMS_NI,
            z=-0.1,
            Nx=20,
            Ny=50,
            t_max_days=5.0,
        )

    with pytest.raises(ValueError, match="z must be non-negative"):
        tf.predict_multiband(
            model="nickel",
            params=PARAMS_NI,
            z=-0.1,
            distance_modulus=MU_7P5_MPC,
            filters={"B": "johnson_cousins.B"},
            t_days=np.array([1.0], float),
            band=np.array(["B"], dtype=object),
            y_kind="flux",
            Nx=20,
            Ny=50,
            t_max_days=5.0,
        )


def test_physical_constraints_reject_ni_mass_larger_than_ejecta():
    assert _physical_constraints_lnprior({"M_ej": 1.0, "M_Ni": 1.1}) == -np.inf
    assert _physical_constraints_lnprior({"M_ej": 1.0, "M_Ni": 1.0}) == pytest.approx(0.0)
    assert _physical_constraints_lnprior({"M_ej": 1.0, "M_Ni": 0.1}) == pytest.approx(0.0)


def test_public_fit_rejects_fixed_ni_mass_larger_than_ejecta():
    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.0e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    with pytest.raises(ValueError, match="M_Ni must be <= M_ej"):
        tf.fit_bol(
            data=data,
            model="nickel",
            fixed={"M_ej": 0.5, "M_Ni": 0.6},
        )


def test_mcmc_backends_treat_nsteps_as_production_length():
    pytest.importorskip("emcee")
    pytest.importorskip("zeus")

    prior = MixedBoundsPrior(
        bounds=np.array([[-1.0, 1.0]], float),
        param_names=["x"],
    )

    def lnprob(x):
        x = np.asarray(x, float)
        return -0.5 * float(np.sum(x * x))

    common = dict(
        lnprob=lnprob,
        prior=prior,
        nwalkers=8,
        nsteps=6,
        burnin=3,
        thin=1,
        seed=2,
        init="prior",
        robust_init=False,
        progress=False,
    )
    samples_emcee, _, _ = run_emcee(**common)
    samples_zeus, _, _ = run_zeus(**common)

    assert samples_emcee.shape == (8 * 6, 1)
    assert samples_zeus.shape == (8 * 6, 1)


def test_legacy_short_nickel_param_dict_is_still_accepted_for_forward_calls():
    out = tf.predict_multiband(
        model="nickel",
        params=LEGACY_PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=np.array([1.0, 2.0], float),
        band=np.array(["B", "B"], dtype=object),
        y_kind="flux",
        mag_system="ab",
        Nx=20,
        Ny=50,
        t_max_days=5.0,
    )

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_bolometric_forward_tmax_is_public_observer_frame():
    lc = tf.lightcurve_bol(
        model="nickel",
        params=LEGACY_PARAMS_NI,
        z=1.0,
        Nx=20,
        Ny=50,
        t_max_days=10.0,
    )
    pred = tf.predict_bol(
        model="nickel",
        params=LEGACY_PARAMS_NI,
        z=1.0,
        t_days=np.array([9.0, 11.0], float),
        Nx=20,
        Ny=50,
        t_max_days=10.0,
        interp_fill="nan",
    )

    assert float(np.nanmax(lc.t_days)) == pytest.approx(10.0)
    assert np.isfinite(pred[0])
    assert np.isnan(pred[1])


def test_multiband_forward_tmax_is_public_observer_frame():
    lc = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NI,
        z=1.0,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        bands=["B"],
        y_kind="flux",
        Nx=20,
        Ny=50,
        t_max_days=10.0,
    )
    pred = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        z=1.0,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=np.array([9.0, 11.0], float),
        band=np.array(["B", "B"], dtype=object),
        y_kind="flux",
        Nx=20,
        Ny=50,
        t_max_days=10.0,
        interp_fill="nan",
    )

    assert float(np.nanmax(lc.t_days)) == pytest.approx(10.0)
    assert np.isfinite(pred[0])
    assert np.isnan(pred[1])


def test_flux_output_is_independent_of_mag_system():
    t_days = np.array([1.0, 2.0, 3.0], float)
    band = np.array(["B", "B", "B"], dtype=object)

    flux_ab = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=t_days,
        band=band,
        y_kind="flux",
        mag_system="ab",
        Nx=20,
        Ny=50,
        t_max_days=5.0,
    )
    flux_vega = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B"},
        t_days=t_days,
        band=band,
        y_kind="flux",
        mag_system="vega",
        Nx=20,
        Ny=50,
        t_max_days=5.0,
    )

    assert np.allclose(flux_ab, flux_vega, equal_nan=True)


def test_public_multiband_signatures_hide_dl_cm():
    for func in (tf.lightcurve_multiband, tf.predict_multiband, tf.fit_multiband):
        params = inspect.signature(func).parameters
        assert "DL_cm" not in params
        assert "DL_Mpc" not in params
        assert "distance_modulus" in params


def test_photometry_pipeline_is_fnu_then_extinction_then_mag():
    filters = normalize_filters({"V": "johnson_cousins.V"})
    extinction = normalize_extinction({"mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"}})
    sed = BlackbodySED()
    bands = ["V"]
    teff = np.array([7000.0, 6500.0], float)
    rph = np.array([4.0e14, 4.3e14], float)

    fnu_model = evaluate_multiband_model_fnu(
        sed=sed,
        filter_map=filters,
        bands=bands,
        Teff_K=teff,
        R_cm=rph,
        DL_cm=7.5 * MPC,
        z=0.01,
    )
    fnu_ext = apply_extinction_to_fnu_grid(
        fnu_model,
        filter_map=filters,
        bands=bands,
        extinction=extinction,
        z=0.01,
    )
    expected = fnu_grid_to_vega_mag_grid(
        fnu_ext,
        filter_map=filters,
        bands=bands,
    )

    out = evaluate_multiband_observer_output(
        sed=sed,
        filter_map=filters,
        bands=bands,
        Teff_K=teff,
        R_cm=rph,
        DL_cm=7.5 * MPC,
        z=0.01,
        y_kind="mag",
        mag_system="vega",
        extinction=extinction,
    )

    assert np.allclose(out, expected, equal_nan=True)


def test_observation_likelihood_dispatch_is_explicit():
    y_obs = np.array([1.0, 2.0, 3.0], float)
    y_model = np.array([1.1, 1.9, 3.2], float)
    y_err = np.array([0.1, 0.2, 0.2], float)

    assert gaussian_lnlike_for_observation(
        y_kind="flux",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
    ) == pytest.approx(gaussian_lnlike_flux(y_obs, y_model, y_err))
    assert gaussian_lnlike_for_observation(
        y_kind="mag",
        y_obs=y_obs,
        y_model=y_model,
        y_err=y_err,
    ) == pytest.approx(gaussian_lnlike_mag(y_obs, y_model, y_err))


def test_structured_extinction_component_resolves_standard_av():
    filters = normalize_filters({"V": "johnson_cousins.V"})
    extinction = normalize_extinction({"mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"}})
    values = resolve_extinction_values_mag(
        extinction,
        filter_map=filters,
        used_bands=["V"],
        z=0.0,
    )

    assert values["V"] == pytest.approx(0.124, rel=0.08)


def test_host_extinction_uses_rest_frame_wavelength():
    filters = normalize_filters({"g": "sdss.g"})
    obs = normalize_extinction({"mw": {"ebv": 0.04, "rv": 3.1, "law": "odonnell94"}})
    host = normalize_extinction({"host": {"ebv": 0.04, "rv": 3.1, "law": "odonnell94"}})

    obs_vals = resolve_extinction_values_mag(obs, filter_map=filters, used_bands=["g"], z=0.5)
    host_vals = resolve_extinction_values_mag(host, filter_map=filters, used_bands=["g"], z=0.5)

    assert host_vals["g"] > obs_vals["g"]


def test_structured_extinction_roundtrip_and_band_map_addition():
    filters = normalize_filters({"B": "johnson_cousins.B", "V": "johnson_cousins.V"})
    extinction = normalize_extinction(
        {
            "band_map": {"B": 0.01},
            "mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"},
        }
    )
    payload = extinction_to_dict(extinction)
    restored = extinction_from_dict(payload)
    values = resolve_extinction_values_mag(
        restored,
        filter_map=filters,
        used_bands=["B", "V"],
        z=0.0,
    )

    assert values["B"] > values["V"]
    assert values["B"] - values["V"] == pytest.approx(0.05, rel=0.4)


def test_predict_multiband_accepts_structured_extinction():
    t_days = np.array([2.0, 4.0, 6.0], float)
    band = np.array(["V", "V", "V"], dtype=object)

    baseline = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"V": "johnson_cousins.V"},
        t_days=t_days,
        band=band,
        y_kind="mag",
        mag_system="vega",
        Nx=20,
        Ny=50,
        t_max_days=8.0,
    )
    reddened = tf.predict_multiband(
        model="nickel",
        params=PARAMS_NI,
        distance_modulus=MU_7P5_MPC,
        filters={"V": "johnson_cousins.V"},
        t_days=t_days,
        band=band,
        y_kind="mag",
        mag_system="vega",
        extinction={"mw": {"ebv": 0.04, "rv": 3.1, "law": "ccm89"}},
        Nx=20,
        Ny=50,
        t_max_days=8.0,
    )

    assert np.all(reddened > baseline)


def test_vega_requires_zero_points_for_all_used_bands():
    with pytest.raises(ValueError):
        tf.lightcurve_multiband(
            model="nickel",
            params=PARAMS_NI,
            z=0.001728,
            filters={"B": 6.8e14},
            bands=["B"],
            y_kind="mag",
            mag_system="vega",
            Nx=20,
            Ny=50,
            t_max_days=5.0,
        )


def test_explicit_distance_and_extinction_roundtrip(tmp_path):
    mb = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NI,
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
        bands=["B", "V"],
        y_kind="mag",
        mag_system="vega",
        extinction={"B": 0.1, "V": 0.05},
        Nx=20,
        Ny=60,
        t_max_days=8.0,
    )
    idx = np.linspace(0, len(mb.t_days) - 1, 5, dtype=int)
    data = tf.MultiBandData(
        t_days=np.repeat(mb.t_days[idx], 2),
        band=np.array(["B", "V"] * len(idx), dtype=object),
        y=np.column_stack([mb.y["B"][idx], mb.y["V"][idx]]).reshape(-1),
        yerr=np.full(10, 0.1),
    )

    res = tf.fit_multiband(
        data=data,
        model="nickel",
        z=0.001728,
        distance_modulus=MU_7P5_MPC,
        filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
        y_kind="mag",
        mag_system="vega",
        extinction={"B": 0.1, "V": 0.05},
        priors={
            "M_ej": (1.0, 5.0),
            "v_ej": (0.5, 2.0),
            "E_Th_in": (0.05, 8.0),
            "M_Ni": (0.01, 0.2),
            "R_0": (10.0, 400.0),
            "T_floor": (2000.0, 6000.0),
        },
        fixed={"x_Ni": 0.2, "kappa": 0.12, "kappa_gamma": 0.03, "t_shift": 0.0},
        sampler="emcee",
        sampler_kwargs={"nwalkers": 16, "nsteps": 10, "burnin": 2, "thin": 1, "seed": 1, "progress": False},
        model_kwargs={"Nx": 20, "Ny": 60, "t_max_days": 8.0},
    )

    path = tmp_path / "fit_multiband_redesign.npz"
    saved = tf.save(res, path=path)
    loaded = tf.load(saved)

    assert loaded["ctx"]["photometry"]["mag_system"] == "vega"
    assert loaded["ctx"]["distance"]["DL_cm"] is not None
    assert "B" in loaded["ctx"]["filters"]
    assert loaded["ctx"]["extinction"]["band_map"]["values_mag"]["B"] == pytest.approx(0.1)


def test_legacy_context_is_upgraded_to_schema_one():
    ctx = {
        "distance": {"z": 0.001728},
        "filters": {"B": 6.8e14},
        "y_kind": "mag",
    }
    out = _validate_ctx_dict(ctx)

    assert out["schema_version"] == 1
    assert out["photometry"]["mag_system"] == "ab"
    assert out["filters"]["B"]["filter_id"] == "legacy:B"
