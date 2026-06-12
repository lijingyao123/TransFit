from __future__ import annotations

import numpy as np
import pytest

import transfit as tf
import transfit.api as api
from transfit.constants import C_LIGHT, MPC
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED


PARAMS_NICKEL = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "E_Th_in": 1.5,
    "M_Ni": 0.08,
    "R_0": 120.0,
    "x_Ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 4500.0,
}


def test_model_parameter_helpers_are_small_and_public():
    assert tf.model_param_names("nickel") == [
        "M_ej",
        "v_ej",
        "E_Th_in",
        "M_Ni",
        "R_0",
        "x_Ni",
        "kappa",
        "kappa_gamma",
        "T_floor",
    ]
    assert tf.model_param_names("csm") == [
        "M_ej",
        "E_sn",
        "M_csm",
        "R_csm_out",
        "kappa",
        "s",
        "eps_sh",
        "T_floor",
    ]
    assert "t_shift" in tf.param_template("nickel", include_t_shift=True)


def test_forward_bolometric_light_curve_is_finite():
    lc = tf.lightcurve_bol(
        model="nickel",
        params=PARAMS_NICKEL,
        z=0.001728,
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 80},
    )

    assert lc.t_days.ndim == 1
    assert np.all(np.diff(lc.t_days) > 0.0)
    assert np.all(np.isfinite(lc.Lbol))
    assert np.all(lc.Lbol > 0.0)
    assert np.all(np.isfinite(lc.Teff))
    assert np.all(np.isfinite(lc.Rph))


def test_forward_multiband_light_curve_is_finite():
    lc = tf.lightcurve_multiband(
        model="nickel",
        params=PARAMS_NICKEL,
        z=0.001728,
        distance_modulus=29.84,
        filters={"B": "johnson_cousins.B", "V": "johnson_cousins.V"},
        bands=["B", "V"],
        y_kind="mag",
        mag_system="vega",
        t_max_days=20.0,
        solver_kwargs={"Nx": 20, "Ny": 80},
    )

    assert lc.bands == ["B", "V"]
    assert set(lc.y) == {"B", "V"}
    assert np.all(np.isfinite(lc.y["B"]))
    assert np.all(np.isfinite(lc.y["V"]))


def test_fit_bol_and_save_load_smoke(monkeypatch, tmp_path):
    def fake_run_sampler(*, sampler, lnprob, prior, sampler_kwargs):
        assert sampler == "emcee"
        assert list(prior.param_names) == []
        sample = np.empty(0, dtype=float)
        logp = lnprob(sample)
        assert np.isfinite(logp)
        return sample.reshape(1, 0), np.array([logp], float), {"fake": True}, "fake"

    monkeypatch.setattr(api, "_run_sampler", fake_run_sampler)

    fixed = dict(PARAMS_NICKEL)
    fixed.pop("T_floor")
    fixed["t_shift"] = 0.0

    data = tf.BolometricData(
        t_days=np.array([1.0, 2.0, 3.0], float),
        y=np.array([1.0e41, 1.1e41, 1.2e41], float),
        yerr=np.array([1.0e40, 1.0e40, 1.0e40], float),
    )

    res = tf.fit_bol(
        data=data,
        model="nickel",
        z=0.001728,
        fixed=fixed,
        model_kwargs={
            "t_max_days": 10.0,
            "solver_kwargs": {"Nx": 20, "Ny": 80},
        },
    )

    assert res.sampler == "fake"
    assert res.best_params_raw["M_ej"] == pytest.approx(PARAMS_NICKEL["M_ej"])
    assert res.best_params_raw["t_shift"] == pytest.approx(0.0)

    out = tf.save(res, tmp_path / "fit_smoke.npz")
    loaded = tf.load(out)

    assert loaded["model"] == "nickel"
    assert loaded["samples"].shape == (1, 0)
    assert loaded["fixed"]["M_ej"] == pytest.approx(PARAMS_NICKEL["M_ej"])


def test_cutoff_blackbody_suppresses_blue_flux_only():
    bb = BlackbodySED()
    sed = CutoffBlackbodySED(
        cutoff_wavelength_A=3000.0,
        uv_slope=2.0,
        min_factor=0.0,
    )

    wavelengths_A = np.array([2000.0, 4000.0], float)
    nu_obs = C_LIGHT / (wavelengths_A * 1.0e-8)
    teff = np.array([10000.0, 9000.0], float)
    rph = np.array([1.0e15, 1.1e15], float)

    f_bb = bb.fnu(nu_obs, teff, rph, DL_cm=7.5 * MPC, z=0.0)
    f_cut = sed.fnu(nu_obs, teff, rph, DL_cm=7.5 * MPC, z=0.0)

    assert np.allclose(f_cut[0] / f_bb[0], (2000.0 / 3000.0) ** 2)
    assert np.allclose(f_cut[1], f_bb[1])
