# transfit/api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Literal, Any, Tuple

import numpy as np

from .modules.interp import interp_fit
from .modules.sed import BlackbodySED
from .samplers import FitResult, gaussian_lnlike, run_emcee
from .priors import UniformBoundsPrior, build_bounds
from transfit.constants import DAY


# -------------------------
# Context / Distance
# -------------------------

@dataclass(frozen=True)
class Distance:
    z: Optional[float] = None
    DL_cm: Optional[float] = None

    def get_z(self) -> float:
        return float(self.z or 0.0)

    def get_DL_cm(self) -> float:
        if self.DL_cm is not None:
            return float(self.DL_cm)
        if self.z is None:
            raise ValueError("Distance needs either DL_cm or z.")
        from astropy.cosmology import Planck15 as cosmo
        import astropy.units as u
        return cosmo.luminosity_distance(self.z).to(u.cm).value


@dataclass(frozen=True)
class Context:
    """
    Context for forward model.

    - Bolometric (predict_bol / fit_bol): only distance is required.
    - Multi-band (predict_multiband / fit_multiband): filters is required,
      y_kind defaults to "mag".
    """
    distance: Distance
    filters: Optional[Dict[str, float]] = None        # band -> nu_eff (Hz), only for multiband
    y_kind: Literal["mag", "flux"] = "mag"            # only matters for multiband



# -------------------------
# Light curve containers
# -------------------------

@dataclass(frozen=True)
class BolometricLC:
    t_days: np.ndarray
    Lbol: np.ndarray
    Teff: np.ndarray
    Rph: np.ndarray


@dataclass(frozen=True)
class MultiBandLC:
    t_days: np.ndarray
    bands: List[str]
    y: Dict[str, np.ndarray]                  # band -> mag/flux array


# -------------------------
# Data containers
# -------------------------

@dataclass(frozen=True)
class BolometricData:
    t_days: np.ndarray
    y: np.ndarray
    yerr: np.ndarray


@dataclass(frozen=True)
class MultiBandData:
    t_days: np.ndarray
    band: np.ndarray
    y: np.ndarray
    yerr: np.ndarray


# -------------------------
# Small utilities
# -------------------------

def _norm_band(x: Any) -> str:
    """
    IMPORTANT: Keep case. Only strip whitespace.
    This makes band matching case-sensitive (as requested).
    """
    return str(x).strip()


def _norm_filters(filters: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize filter keys by strip only (keep case).
    """
    return {_norm_band(k): float(v) for k, v in filters.items()}


def _require_bands_in_filters(bands: Sequence[str], filters: Dict[str, float]) -> None:
    """
    Case-sensitive check.
    """
    missing = [b for b in bands if b not in filters]
    if missing:
        raise KeyError(
            f"These bands are missing in ctx.filters (case-sensitive): {missing}. "
            f"Available: {sorted(filters.keys())}"
        )


def _as_1d_float(x, name: str) -> np.ndarray:
    arr = np.asarray(x, float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _check_same_length(**arrays: np.ndarray) -> None:
    lens = {k: int(np.asarray(v).size) for k, v in arrays.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Input arrays must have the same length, got: {lens}")


# -------------------------
# Model registry
# -------------------------

_ENGINE_CACHE = {}

def _get_engine(model: str):
    # model name itself can be case-insensitive
    m = str(model).lower().strip()
    if m in _ENGINE_CACHE:
        return _ENGINE_CACHE[m]

    if m in ["ni", "nickel"]:
        from .model.Nickel import NickelModel
        eng = NickelModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m in ["scni", "sc_ni", "sc-nickel", "shockcooling+ni"]:
        from .model.SC_Nickel import SCNiModel
        eng = SCNiModel()
        _ENGINE_CACHE[m] = eng
        return eng

    # ✅ Magnetar
    if m in ["magnetar", "mag", "mg"]:
        from .model.Magnetar import MagnetarModel
        eng = MagnetarModel()
        _ENGINE_CACHE[m] = eng
        return eng

    raise ValueError(f"Unknown model='{model}'")


def _normalize_theta(model: str, theta, *, allow_missing_tfloor: bool):
    """
    Allow omitting T_floor in forward-model calls by appending 0.0.
    This keeps backward compatibility with shorter theta in examples.
    """
    m = str(model).lower().strip()
    theta_t = tuple(theta)

    if m in ["ni", "nickel"]:
        expected = 7
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, 1000.0)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    if m in ["scni", "sc_ni", "sc-nickel", "shockcooling+ni"]:
        expected = 9
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, 1000.0)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    if m in ["magnetar", "mag", "mg"]:
        expected = 9
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, 1000.0)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    return theta_t


def _solve_state(engine, theta, *, Nx: int, Ny: int, t_max_days: float):
    return engine.calculate_light_curve(theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)


def _t_grid_days_from_ts(t_s: np.ndarray, z: float) -> np.ndarray:
    # 保持你当前工程的时间映射约定
    return (np.asarray(t_s, float) * (1.0 + z)) / DAY


# -------------------------
# Forward model
# -------------------------

def lightcurve_bol(
    *,
    model: str,
    theta,
    ctx: Context,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
) -> BolometricLC:
    engine = _get_engine(model)
    theta = _normalize_theta(model, theta, allow_missing_tfloor=True)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.get_z()
    t_days = _t_grid_days_from_ts(t_s, z=z)
    return BolometricLC(
        t_days=np.asarray(t_days, float),
        Lbol=np.asarray(Lbol, float),
        Teff=np.asarray(Teff, float),
        Rph=np.asarray(Rph, float),
    )


def predict_bol(
    *,
    model: str,
    theta,
    ctx: Context,
    t_days: np.ndarray,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
) -> np.ndarray:
    engine = _get_engine(model)
    theta = _normalize_theta(model, theta, allow_missing_tfloor=True)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.get_z()
    t_grid_days = _t_grid_days_from_ts(t_s, z=z)

    # Lbol 正值量：log10 插值更稳
    return interp_fit(
        t_grid_days,
        np.asarray(Lbol, float),
        np.asarray(t_days, float),
        yscale="log10",
        fill="edge",
    )


def lightcurve_multiband(
    *,
    model: str,
    theta,
    ctx: Context,
    bands: Sequence[str],
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    sed=None,
) -> MultiBandLC:


    sed = sed or BlackbodySED()
    if ctx.filters is None:
        raise ValueError("ctx.filters is required for multiband. For bolometric you can omit it.")
    filters = _norm_filters(ctx.filters)
    # ✅ bands 保留大小写，只 strip
    bands = [_norm_band(b) for b in list(bands)]
    _require_bands_in_filters(bands, filters)

    engine = _get_engine(model)
    theta = _normalize_theta(model, theta, allow_missing_tfloor=False)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.get_z()
    DL_cm = ctx.distance.get_DL_cm()
    t_days = _t_grid_days_from_ts(t_s, z=z)

    nu_obs = np.array([filters[b] for b in bands], float)

    if ctx.y_kind == "mag":
        y_grid = sed.abmag(nu_obs, np.asarray(Teff), np.asarray(Rph), DL_cm=DL_cm, z=z)  # (Nb,Nt)
    else:
        y_grid = sed.fnu(nu_obs, np.asarray(Teff), np.asarray(Rph), DL_cm=DL_cm, z=z)

    y = {b: np.asarray(y_grid[i], float).copy() for i, b in enumerate(bands)}
    return MultiBandLC(t_days=np.asarray(t_days, float), bands=bands, y=y)


def predict_multiband(
    *,
    model: str,
    theta,
    ctx: Context,
    t_days: np.ndarray,
    band: np.ndarray,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    sed=None,
) -> np.ndarray:
    sed = sed or BlackbodySED()

    filters = _norm_filters(ctx.filters)
    t_days = np.asarray(t_days, float).reshape(-1)

    # ✅ band 保留大小写，只 strip
    band = np.asarray([_norm_band(b) for b in np.asarray(band).reshape(-1)], dtype=object)
    _check_same_length(t_days=t_days, band=band)

    uniq = sorted(set(band.tolist()))
    _require_bands_in_filters(uniq, filters)

    nu_obs = np.array([filters[b] for b in uniq], float)

    engine = _get_engine(model)
    theta = _normalize_theta(model, theta, allow_missing_tfloor=False)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.get_z()
    DL_cm = ctx.distance.get_DL_cm()
    t_grid_days = _t_grid_days_from_ts(t_s, z=z)

    if ctx.y_kind == "mag":
        y_grid = sed.abmag(nu_obs, np.asarray(Teff), np.asarray(Rph), DL_cm=DL_cm, z=z)
        itp_yscale = "linear"
    else:
        y_grid = sed.fnu(nu_obs, np.asarray(Teff), np.asarray(Rph), DL_cm=DL_cm, z=z)
        itp_yscale = "log10"

    out = np.empty_like(t_days, float)
    for i, b in enumerate(uniq):
        idx = np.where(band == b)[0]
        if idx.size == 0:
            continue
        out[idx] = interp_fit(
            t_grid_days,
            np.asarray(y_grid[i], float),
            t_days[idx],
            yscale=itp_yscale,
            fill="edge",
        )
    return out


# -------------------------
# Fitting helpers
# -------------------------

def _split_sampling(
    names_all: List[str],
    bounds_all: np.ndarray,
    fixed: Optional[Dict[str, float]],
):
    fixed = dict(fixed or {})
    bounds_all = np.asarray(bounds_all, float)

    names_samp: List[str] = []
    bounds_samp: List[List[float]] = []

    for n, (lo, hi) in zip(names_all, bounds_all):
        lo = float(lo)
        hi = float(hi)
        if n in fixed:
            v = float(fixed[n])
            # allow boundary values for fixed params
            if not (lo <= v <= hi):
                raise ValueError(f"fixed['{n}']={v} out of bounds ({lo}, {hi})")
            continue
        names_samp.append(n)
        bounds_samp.append([lo, hi])

    return names_samp, np.asarray(bounds_samp, float), fixed


def _assemble_theta(
    sample_vec: np.ndarray,
    names_samp: List[str],
    fixed: Dict[str, float],
    names_all: List[str],
):
    vals = dict(fixed)
    vals.update({k: float(v) for k, v in zip(names_samp, np.asarray(sample_vec, float))})

    t_shift_days = float(vals.get("t_shift_days", 0.0))

    theta_model: List[float] = []
    for n in names_all:
        if n == "t_shift_days":
            continue
        if n not in vals:
            raise KeyError(
                f"Missing parameter '{n}'. "
                f"Either provide it in priors (to sample) or in fixed."
            )
        theta_model.append(float(vals[n]))

    return tuple(theta_model), t_shift_days


# -------------------------
# Public fit API
# -------------------------

def fit_multiband(
    *,
    data: MultiBandData,
    model: str,
    ctx: Context,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    include_t_shift: bool = True,
) -> FitResult:
    sampler_kwargs = dict(sampler_kwargs or {})
    model_kwargs = dict(model_kwargs or {})

    # ---- data ----
    t_obs = _as_1d_float(data.t_days, "data.t_days")
    y_obs = _as_1d_float(data.y, "data.y")
    y_err = _as_1d_float(data.yerr, "data.yerr")

    # ✅ 保留 band 大小写，只 strip
    band = np.asarray([_norm_band(b) for b in np.asarray(data.band).reshape(-1)], dtype=object)

    _check_same_length(t_days=t_obs, band=band, y=y_obs, yerr=y_err)

    if np.any(~np.isfinite(y_obs)) or np.any(~np.isfinite(y_err)) or np.any(y_err <= 0):
        raise ValueError("data.y and data.yerr must be finite and yerr > 0.")

    # ---- bounds/prior ----
    names_all, bounds_all = build_bounds(model, priors=priors, include_t_shift=include_t_shift)
    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed)
    prior = UniformBoundsPrior(bounds=bounds_samp, param_names=names_samp)

    # ---- band check (case-sensitive) ----
    if ctx.filters is None:
        raise ValueError("ctx.filters is required for multiband. For bolometric you can omit it.")
    filters = _norm_filters(ctx.filters)
    uniq_b = sorted(set(band.tolist()))
    _require_bands_in_filters(uniq_b, filters)

    # ---- lnprob ----
    def lnprob(sample_vec: np.ndarray) -> float:
        lp = prior.lnprior(sample_vec)
        if not np.isfinite(lp):
            return -np.inf

        theta_model, t_shift_days = _assemble_theta(sample_vec, names_samp, fixed, names_all)
        t_eval = t_obs + t_shift_days

        y_mod = predict_multiband(
            model=model,
            theta=theta_model,
            ctx=ctx,
            t_days=t_eval,
            band=band,
            **model_kwargs,
        )

        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        return lp + gaussian_lnlike(y_obs, y_mod, y_err)

    if sampler.lower() != "emcee":
        raise ValueError("v0.1 only supports sampler='emcee'")

    ndim = len(names_samp)
    nwalkers = int(sampler_kwargs.get("nwalkers", max(32, 4 * max(ndim, 1))))
    nsteps = int(sampler_kwargs.get("nsteps", 2000))
    burnin = int(sampler_kwargs.get("burnin", nsteps // 2))
    thin = int(sampler_kwargs.get("thin", 1))
    seed = sampler_kwargs.get("seed", None)
    init = sampler_kwargs.get("init", "prior")
    pool = sampler_kwargs.get("pool", None)
    progress = bool(sampler_kwargs.get("progress", True))

    samples, logp, meta = run_emcee(
        lnprob=lnprob,
        prior=prior,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        thin=thin,
        seed=seed,
        init=init,
        pool=pool,
        progress=progress,
    )

    meta.update(
        dict(
            model=model,
            y_kind=ctx.y_kind,
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            include_t_shift=include_t_shift,
            model_kwargs=model_kwargs,
        )
    )

    return FitResult(
        model=model,
        ctx=ctx,
        sampler="emcee",
        param_names=names_samp,
        fixed=fixed,
        all_param_names=names_all,
        samples=samples,
        log_prob=logp,
        meta=meta,
    )


def fit_bol(
    *,
    data: BolometricData,
    model: str,
    ctx: Context,
    priors: Optional[Dict[str, Tuple[float, float]]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    include_t_shift: bool = True,
) -> FitResult:
    sampler_kwargs = dict(sampler_kwargs or {})
    model_kwargs = dict(model_kwargs or {})

    t_obs = _as_1d_float(data.t_days, "data.t_days")
    y_obs = _as_1d_float(data.y, "data.y")
    y_err = _as_1d_float(data.yerr, "data.yerr")
    _check_same_length(t_days=t_obs, y=y_obs, yerr=y_err)

    if np.any(~np.isfinite(y_obs)) or np.any(~np.isfinite(y_err)) or np.any(y_err <= 0):
        raise ValueError("data.y and data.yerr must be finite and yerr > 0.")

    names_all, bounds_all = build_bounds(model, priors=priors, include_t_shift=include_t_shift)
    # 热光变拟合：T_floor 不作为参数（不参与先验/拟合）
    if "T_floor" in names_all:
        i_tf = names_all.index("T_floor")
        names_all = [n for n in names_all if n != "T_floor"]
        bounds_all = np.asarray(bounds_all, float)
        bounds_all = np.delete(bounds_all, i_tf, axis=0)

    fixed = dict(fixed or {})
    fixed.pop("T_floor", None)

    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed)
    prior = UniformBoundsPrior(bounds=bounds_samp, param_names=names_samp)

    def lnprob(sample_vec: np.ndarray) -> float:
        lp = prior.lnprior(sample_vec)
        if not np.isfinite(lp):
            return -np.inf

        theta_model, t_shift_days = _assemble_theta(sample_vec, names_samp, fixed, names_all)
        t_eval = t_obs + t_shift_days

        y_mod = predict_bol(
            model=model,
            theta=theta_model,
            ctx=ctx,
            t_days=t_eval,
            **model_kwargs,
        )

        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        return lp + gaussian_lnlike(y_obs, y_mod, y_err)

    if sampler.lower() != "emcee":
        raise ValueError("v0.1 only supports sampler='emcee'")

    ndim = len(names_samp)
    nwalkers = int(sampler_kwargs.get("nwalkers", max(32, 4 * max(ndim, 1))))
    nsteps = int(sampler_kwargs.get("nsteps", 2000))
    burnin = int(sampler_kwargs.get("burnin", nsteps // 2))
    thin = int(sampler_kwargs.get("thin", 1))
    seed = sampler_kwargs.get("seed", None)
    init = sampler_kwargs.get("init", "prior")
    pool = sampler_kwargs.get("pool", None)
    progress = bool(sampler_kwargs.get("progress", True))

    samples, logp, meta = run_emcee(
        lnprob=lnprob,
        prior=prior,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burnin=burnin,
        thin=thin,
        seed=seed,
        init=init,
        pool=pool,
        progress=progress,
    )

    meta.update(
        dict(
            model=model,
            y_kind="bol",
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            include_t_shift=include_t_shift,
            model_kwargs=model_kwargs,
        )
    )

    return FitResult(
        model=model,
        ctx=ctx,
        sampler="emcee",
        param_names=names_samp,
        fixed=fixed,
        all_param_names=names_all,
        samples=samples,
        log_prob=logp,
        meta=meta,
    )


__all__ = [
    "Distance", "Context",
    "BolometricLC", "MultiBandLC",
    "BolometricData", "MultiBandData",
    "lightcurve_bol", "predict_bol",
    "lightcurve_multiband", "predict_multiband",
    "fit_bol", "fit_multiband",
]
