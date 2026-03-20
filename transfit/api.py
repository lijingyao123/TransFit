# transfit/api.py
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Dict, List, Optional, Sequence, Literal, Any, Tuple

import numpy as np

from .data import BolometricData, MultiBandData
from .modules.interp import interp_fit
from .modules.sed import BlackbodySED
from .samplers import FitResult, gaussian_lnlike, run_emcee, run_zeus, run_dynesty
from .priors import MixedBoundsPrior, build_bounds
from transfit.constants import DAY


# -------------------------
# Internal forward metadata
# -------------------------

_BOL_INTERNAL_T_FLOOR = 1000.0

def _cosmo_luminosity_distance_cm(z: float) -> float:
    from astropy.cosmology import Planck15 as cosmo
    import astropy.units as u

    return cosmo.luminosity_distance(float(z)).to(u.cm).value

@dataclass(frozen=True)
class _Distance:
    z: Optional[float] = None
    DL_cm: Optional[float] = None

    def get_z(self) -> float:
        return float(self.z or 0.0)

    def require_z(self) -> float:
        if self.z is None:
            raise ValueError("Redshift z is required for observer-frame multiband calculations.")
        return float(self.z)

    def get_DL_cm(self) -> float:
        if self.DL_cm is not None:
            dl = float(self.DL_cm)
            if self.z is not None:
                dl_cosmo = _cosmo_luminosity_distance_cm(float(self.z))
                if np.isfinite(dl_cosmo) and dl_cosmo > 0.0:
                    frac = abs(dl - dl_cosmo) / dl_cosmo
                    if frac > 0.05:
                        warnings.warn(
                            "Using a user-supplied DL_cm that differs from the Planck15 luminosity distance "
                            "implied by z by more than 5%. z is still used for time/frequency redshift terms.",
                            stacklevel=2,
                        )
            return dl
        if self.z is None:
            raise ValueError("Distance needs either DL_cm or z.")
        return _cosmo_luminosity_distance_cm(float(self.z))


@dataclass(frozen=True)
class _Context:
    """
    Internal context for forward-model evaluation.

    - Bolometric prediction: only distance is required.
    - Multi-band prediction: filters is required,
      y_kind defaults to "mag".
    """
    distance: _Distance
    filters: Optional[Dict[str, float]] = None        # band -> nu_eff (Hz), only for multiband
    y_kind: Literal["mag", "flux"] = "mag"            # only matters for multiband


def _context_from_fit_inputs(
    *,
    z: Optional[float],
    filters: Optional[Dict[str, float]],
    y_kind: Literal["mag", "flux"],
    require_filters: bool,
) -> _Context:
    """
    Build the internal Context used by fitting.

    Public fitting APIs accept direct scalar inputs instead of exposing
    Context/Distance as required user-facing concepts.
    """
    if z is None:
        raise ValueError("Provide `z` for fitting. The public fit API standardizes distance handling on redshift.")
    if require_filters and filters is None:
        raise ValueError(
            "filters is required for multiband fitting."
        )
    return _Context(
        distance=_Distance(z=z),
        filters=filters,
        y_kind=str(y_kind),
    )


def _context_from_forward_inputs(
    *,
    z: Optional[float],
    filters: Optional[Dict[str, float]],
    y_kind: Literal["mag", "flux"],
    require_filters: bool,
    require_z: bool,
) -> _Context:
    """
    Build internal forward metadata for public prediction/lightcurve helpers.

    Public forward APIs accept direct scalar inputs instead of requiring
    Context/Distance objects from users.
    """
    if require_z and z is None:
        raise ValueError("Provide `z` for multiband forward calculations.")
    if require_filters and filters is None:
        raise ValueError("filters is required for multiband forward calculations.")
    return _Context(
        distance=_Distance(z=0.0 if z is None else z),
        filters=filters,
        y_kind=str(y_kind),
    )



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
            f"These bands are missing in filters (case-sensitive): {missing}. "
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


def _apply_data_filter(data):
    """
    Apply container-level masking/cleaning when available.

    This makes `mask` a first-class part of the public fitting API instead of
    requiring users to remember calling `.filtered()` manually.
    """
    if hasattr(data, "filtered"):
        return data.filtered()
    return data


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
        from .models.nickel import NickelModel
        eng = NickelModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m in ["scni", "sc_ni", "sc-nickel", "shockcooling+ni"]:
        from .models.sc_ni import SCNiModel
        eng = SCNiModel()
        _ENGINE_CACHE[m] = eng
        return eng

    # SC Magnetar (keeps E_Th_in and R_0 as model parameters)
    if m in ["scmagnetar", "sc_magnetar", "sc-magnetar"]:
        from .models.sc_magnetar import SCMagnetarModel
        eng = SCMagnetarModel()
        _ENGINE_CACHE[m] = eng
        return eng

    # Pure Magnetar (E_Th_in=0, R_0=1 fixed)
    if m in ["magnetar", "mag", "mg"]:
        from .models.magnetar import MagnetarModel
        eng = MagnetarModel()
        _ENGINE_CACHE[m] = eng
        return eng

    # Magnetar + Ni
    if m in ["magni", "mag_ni", "mag-ni", "mag+ni", "magnetar+ni", "magnetar_ni", "magnetar-ni"]:
        from .models.magnetar_ni import MagNiModel
        eng = MagNiModel()
        _ENGINE_CACHE[m] = eng
        return eng

    raise ValueError(f"Unknown model='{model}'")


def _normalize_theta(model: str, theta, *, allow_missing_tfloor: bool):
    """
    Allow omitting T_floor in forward-model calls by appending the internal
    numerical floor used for bolometric-only evaluation.
    This keeps backward compatibility with shorter theta in examples.
    """
    m = str(model).lower().strip()
    theta_t = tuple(theta)

    if m in ["ni", "nickel"]:
        expected = 7
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, _BOL_INTERNAL_T_FLOOR)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    if m in ["scni", "sc_ni", "sc-nickel", "shockcooling+ni"]:
        expected = 9
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, _BOL_INTERNAL_T_FLOOR)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    if m in ["scmagnetar", "sc_magnetar", "sc-magnetar"]:
        expected = 9
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, _BOL_INTERNAL_T_FLOOR)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    if m in ["magnetar", "mag", "mg"]:
        expected = 7
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, _BOL_INTERNAL_T_FLOOR)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    if m in ["magni", "mag_ni", "mag-ni", "mag+ni", "magnetar+ni", "magnetar_ni", "magnetar-ni"]:
        expected = 8
        if len(theta_t) == expected - 1:
            if not allow_missing_tfloor:
                raise ValueError(f"theta for model='{model}' must have length {expected}")
            return (*theta_t, _BOL_INTERNAL_T_FLOOR)
        if len(theta_t) != expected:
            raise ValueError(f"theta for model='{model}' must have length {expected} (or {expected-1} without T_floor)")
        return theta_t

    return theta_t


def model_param_names(model: str, *, include_t_shift: bool = False) -> List[str]:
    names, _ = build_bounds(model, include_t_shift=include_t_shift)
    return list(names)


def param_template(
    model: str,
    *,
    include_t_shift: bool = False,
    fill_value: Any = None,
) -> Dict[str, Any]:
    return {name: fill_value for name in model_param_names(model, include_t_shift=include_t_shift)}


def _theta_from_params(
    model: str,
    params: Dict[str, Any],
    *,
    allow_missing_tfloor: bool,
):
    if not isinstance(params, dict):
        raise TypeError("params must be a dict mapping parameter names to values.")

    values = dict(params)
    # Allow direct reuse of res.best_params / res.median_params in forward helpers.
    values.pop("t_shift", None)

    names = model_param_names(model, include_t_shift=False)
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise KeyError(f"Unknown parameter(s) for model='{model}': {unknown}. Allowed: {names}")

    missing = [
        n for n in names
        if n not in values and not (allow_missing_tfloor and n == "T_floor")
    ]
    if missing:
        raise KeyError(f"Missing parameter(s) for model='{model}': {missing}. Required: {names}")

    theta = []
    for n in names:
        if n == "T_floor" and n not in values and allow_missing_tfloor:
            theta.append(_BOL_INTERNAL_T_FLOOR)
        else:
            theta.append(float(values[n]))
    return tuple(theta)


def _resolve_forward_theta(
    model: str,
    *,
    params: Optional[Dict[str, Any]],
    theta,
    allow_missing_tfloor: bool,
):
    if params is not None and theta is not None:
        raise ValueError("Provide either `params` or `theta`, not both.")
    if params is not None:
        return _theta_from_params(model, params, allow_missing_tfloor=allow_missing_tfloor)
    if theta is not None:
        return _normalize_theta(model, theta, allow_missing_tfloor=allow_missing_tfloor)
    raise ValueError("Provide `params` for forward-model evaluation.")


def _solve_state(engine, theta, *, Nx: int, Ny: int, t_max_days: float):
    return engine.calculate_light_curve(theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)


def _t_grid_days_from_ts(t_s: np.ndarray, z: float) -> np.ndarray:
    # Keep the current project convention for observer-frame time mapping.
    return (np.asarray(t_s, float) * (1.0 + z)) / DAY


# -------------------------
# Forward model
# -------------------------

def lightcurve_bol(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    theta=None,
    z: Optional[float] = None,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
) -> BolometricLC:
    ctx = _context_from_forward_inputs(
        z=z,
        filters=None,
        y_kind="mag",
        require_filters=False,
        require_z=False,
    )
    engine = _get_engine(model)
    theta = _resolve_forward_theta(model, params=params, theta=theta, allow_missing_tfloor=True)
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
    params: Optional[Dict[str, Any]] = None,
    theta=None,
    z: Optional[float] = None,
    t_days: np.ndarray,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    interp_fill: Literal["edge", "nan", "raise"] = "nan",
) -> np.ndarray:
    ctx = _context_from_forward_inputs(
        z=z,
        filters=None,
        y_kind="mag",
        require_filters=False,
        require_z=False,
    )
    engine = _get_engine(model)
    theta = _resolve_forward_theta(model, params=params, theta=theta, allow_missing_tfloor=True)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.get_z()
    t_grid_days = _t_grid_days_from_ts(t_s, z=z)

    # Lbol is strictly positive; log10 interpolation is more stable.
    return interp_fit(
        t_grid_days,
        np.asarray(Lbol, float),
        np.asarray(t_days, float),
        yscale="log10",
        fill=interp_fill,
    )


def lightcurve_multiband(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    theta=None,
    z: Optional[float],
    filters: Dict[str, float],
    bands: Sequence[str],
    y_kind: Literal["mag", "flux"] = "mag",
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    sed=None,
) -> MultiBandLC:
    ctx = _context_from_forward_inputs(
        z=z,
        filters=filters,
        y_kind=y_kind,
        require_filters=True,
        require_z=True,
    )
    sed = sed or BlackbodySED()
    filters = _norm_filters(ctx.filters)
    # Keep band case; only strip whitespace.
    bands = [_norm_band(b) for b in list(bands)]
    _require_bands_in_filters(bands, filters)

    engine = _get_engine(model)
    theta = _resolve_forward_theta(model, params=params, theta=theta, allow_missing_tfloor=False)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.require_z()
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
    params: Optional[Dict[str, Any]] = None,
    theta=None,
    z: Optional[float],
    filters: Dict[str, float],
    t_days: np.ndarray,
    band: np.ndarray,
    y_kind: Literal["mag", "flux"] = "mag",
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    interp_fill: Literal["edge", "nan", "raise"] = "nan",
    sed=None,
) -> np.ndarray:
    ctx = _context_from_forward_inputs(
        z=z,
        filters=filters,
        y_kind=y_kind,
        require_filters=True,
        require_z=True,
    )
    sed = sed or BlackbodySED()
    filters = _norm_filters(ctx.filters)
    t_days = np.asarray(t_days, float).reshape(-1)

    # Keep band case; only strip whitespace.
    band = np.asarray([_norm_band(b) for b in np.asarray(band).reshape(-1)], dtype=object)
    _check_same_length(t_days=t_days, band=band)

    uniq = sorted(set(band.tolist()))
    _require_bands_in_filters(uniq, filters)

    nu_obs = np.array([filters[b] for b in uniq], float)

    engine = _get_engine(model)
    theta = _resolve_forward_theta(model, params=params, theta=theta, allow_missing_tfloor=False)
    t_s, Lbol, Teff, Rph = _solve_state(engine, theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days)

    z = ctx.distance.require_z()
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
            fill=interp_fill,
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
    unknown_fixed = sorted(set(fixed.keys()) - set(names_all))
    if unknown_fixed:
        raise KeyError(
            f"Unknown fixed parameter(s): {unknown_fixed}. Allowed: {names_all}"
        )
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

    t_shift = float(vals.get("t_shift", 0.0))

    theta_model: List[float] = []
    for n in names_all:
        if n == "t_shift":
            continue
        if n not in vals:
            raise KeyError(
                f"Missing parameter '{n}'. "
                f"Either provide it in priors (to sample) or in fixed."
            )
        theta_model.append(float(vals[n]))

    return tuple(theta_model), t_shift


def _apply_log10_priors(
    names: Sequence[str],
    bounds: np.ndarray,
    priors_log10: Optional[Dict[str, Tuple[float, float]]],
):
    """
    Apply log10 prior bounds and return updated bounds + log-prior flags.

    `priors_log10` format:
      {"param": (log10_lo, log10_hi), ...}
    """
    b = np.asarray(bounds, float).copy()
    names_l = [str(n) for n in names]
    idx = {n: i for i, n in enumerate(names_l)}
    log_set = set()

    if not priors_log10:
        return b, log_set

    for k, vv in dict(priors_log10).items():
        if k not in idx:
            raise KeyError(f"Unknown log prior key '{k}'. Allowed: {names_l}")
        if vv is None or len(vv) != 2:
            raise ValueError(f"Invalid log10 bounds for '{k}': {vv!r}")

        lo_log10 = float(vv[0])
        hi_log10 = float(vv[1])
        if not (lo_log10 < hi_log10):
            raise ValueError(
                f"Invalid log10 bounds for '{k}': ({lo_log10}, {hi_log10})"
            )

        lo = 10.0 ** lo_log10
        hi = 10.0 ** hi_log10
        if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > 0.0):
            raise ValueError(
                f"log10 bounds for '{k}' lead to invalid linear bounds: ({lo}, {hi})"
            )

        b[idx[k], 0] = lo
        b[idx[k], 1] = hi
        log_set.add(k)

    return b, log_set


def _split_prior_specs(
    priors: Optional[Dict[str, Any]],
):
    """
    Parse mixed prior specs into linear bounds and log10 bounds.

    Supported `priors` formats per parameter:
      - (lo, hi)                      -> linear uniform bounds
      - ("log10", lo_log10, hi_log10) -> log10 bounds, log-uniform prior
      - {"bounds": (lo, hi), "scale": "linear" | "log10"}
    """
    if not priors:
        return {}, {}

    pri_lin: Dict[str, Tuple[float, float]] = {}
    pri_log10: Dict[str, Tuple[float, float]] = {}

    for k, spec in dict(priors).items():
        if isinstance(spec, dict):
            if "bounds" not in spec:
                raise ValueError(
                    f"Prior spec for '{k}' must contain key 'bounds' when using dict format."
                )
            b = spec["bounds"]
            if b is None or len(b) != 2:
                raise ValueError(f"Invalid bounds for '{k}': {b!r}")
            lo = float(b[0])
            hi = float(b[1])
            if not (lo < hi):
                raise ValueError(f"Invalid bounds for '{k}': ({lo}, {hi})")

            scale = str(spec.get("scale", "linear")).strip().lower()
            if scale in ("linear", "lin"):
                pri_lin[k] = (lo, hi)
            elif scale in ("log10", "log"):
                pri_log10[k] = (lo, hi)
            else:
                raise ValueError(
                    f"Unknown prior scale '{scale}' for '{k}'. Use 'linear' or 'log10'."
                )
            continue

        if isinstance(spec, (tuple, list)):
            if len(spec) == 2 and not isinstance(spec[0], str):
                lo = float(spec[0])
                hi = float(spec[1])
                if not (lo < hi):
                    raise ValueError(f"Invalid bounds for '{k}': ({lo}, {hi})")
                pri_lin[k] = (lo, hi)
                continue

            if len(spec) == 3 and isinstance(spec[0], str):
                mode = str(spec[0]).strip().lower()
                lo = float(spec[1])
                hi = float(spec[2])
                if not (lo < hi):
                    raise ValueError(f"Invalid bounds for '{k}': ({lo}, {hi})")
                if mode in ("log10", "log"):
                    pri_log10[k] = (lo, hi)
                elif mode in ("linear", "lin"):
                    pri_lin[k] = (lo, hi)
                else:
                    raise ValueError(
                        f"Unknown prior mode '{mode}' for '{k}'. Use 'linear' or 'log10'."
                    )
                continue

        raise ValueError(
            f"Invalid prior spec for '{k}': {spec!r}. "
            "Use (lo, hi), ('log10', lo, hi), or {'bounds': (lo, hi), 'scale': 'linear|log10'}."
        )

    return pri_lin, pri_log10


def _run_sampler(
    *,
    sampler: str,
    lnprob,
    prior,
    sampler_kwargs: Dict[str, Any],
):
    """
    Dispatch sampler backend and return (samples, logp, meta, sampler_name).
    """
    key = str(sampler).lower().strip()
    kw = dict(sampler_kwargs or {})

    if key == "emcee":
        ndim = len(prior.param_names)
        nwalkers = int(kw.pop("nwalkers", max(32, 4 * max(ndim, 1))))
        nsteps = int(kw.pop("nsteps", 2000))
        burnin = int(kw.pop("burnin", nsteps // 2))
        thin = int(kw.pop("thin", 1))
        seed = kw.pop("seed", None)
        init = kw.pop("init", "prior")
        pool = kw.pop("pool", None)
        progress = bool(kw.pop("progress", True))

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
            **kw,
        )
        return samples, logp, meta, "emcee"

    if key == "zeus":
        ndim = len(prior.param_names)
        nwalkers = int(kw.pop("nwalkers", max(32, 4 * max(ndim, 1))))
        nsteps = int(kw.pop("nsteps", 2000))
        burnin = int(kw.pop("burnin", nsteps // 2))
        thin = int(kw.pop("thin", 1))
        seed = kw.pop("seed", None)
        init = kw.pop("init", "prior")
        pool = kw.pop("pool", None)
        progress = bool(kw.pop("progress", True))

        samples, logp, meta = run_zeus(
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
            **kw,
        )
        return samples, logp, meta, "zeus"

    if key == "dynesty":
        nlive = int(kw.pop("nlive", 200))
        sample = kw.pop("sample", "rwalk")
        bound = kw.pop("bound", "multi")
        dlogz = float(kw.pop("dlogz", 0.1))
        maxiter = kw.pop("maxiter", None)
        maxcall = kw.pop("maxcall", None)
        seed = kw.pop("seed", None)
        progress = bool(kw.pop("progress", True))
        nsamples = kw.pop("nsamples", None)
        add_live = bool(kw.pop("add_live", True))
        pool = kw.pop("pool", None)
        queue_size = kw.pop("queue_size", None)

        samples, logp, meta = run_dynesty(
            lnprob=lnprob,
            prior=prior,
            nlive=nlive,
            sample=sample,
            bound=bound,
            dlogz=dlogz,
            maxiter=maxiter,
            maxcall=maxcall,
            seed=seed,
            progress=progress,
            nsamples=nsamples,
            add_live=add_live,
            pool=pool,
            queue_size=queue_size,
            **kw,
        )
        return samples, logp, meta, "dynesty"

    raise ValueError(
        f"Unknown sampler='{sampler}'. Supported samplers: 'emcee', 'zeus', 'dynesty'."
    )


# -------------------------
# Fit-time interpolation policy
# -------------------------

def _split_fit_model_kwargs(model_kwargs: Optional[Dict[str, Any]]):
    """
    Extract prediction kwargs and interpolation fill policy used by fit lnprob.

    For fitting, edge-clamped extrapolation is disallowed because it evaluates
    points outside model time range using boundary values.
    """
    mk = dict(model_kwargs or {})
    fill = str(mk.pop("interp_fill", "nan")).strip().lower()

    if fill not in ("edge", "nan", "raise"):
        raise ValueError(
            "model_kwargs['interp_fill'] must be one of: 'edge', 'nan', 'raise'."
        )
    if fill == "edge":
        raise ValueError(
            "model_kwargs['interp_fill']='edge' is not allowed in fitting. "
            "Use 'nan' or 'raise' to avoid out-of-range edge extrapolation."
        )

    return mk, fill


def fit_multiband(
    *,
    data: MultiBandData,
    model: str,
    z: Optional[float] = None,
    filters: Optional[Dict[str, float]] = None,
    y_kind: Literal["mag", "flux"] = "mag",
    priors: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> FitResult:
    ctx = _context_from_fit_inputs(
        z=z,
        filters=filters,
        y_kind=y_kind,
        require_filters=True,
    )
    sampler_kwargs = dict(sampler_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    model_kwargs_pred, interp_fill_fit = _split_fit_model_kwargs(model_kwargs)
    data = _apply_data_filter(data)

    # ---- data ----
    t_obs = _as_1d_float(data.t_days, "data.t_days")
    y_obs = _as_1d_float(data.y, "data.y")
    y_err = _as_1d_float(data.yerr, "data.yerr")

    # Keep band case; only strip whitespace.
    band = np.asarray([_norm_band(b) for b in np.asarray(data.band).reshape(-1)], dtype=object)

    _check_same_length(t_days=t_obs, band=band, y=y_obs, yerr=y_err)

    if np.any(~np.isfinite(y_obs)) or np.any(~np.isfinite(y_err)) or np.any(y_err <= 0):
        raise ValueError("data.y and data.yerr must be finite and yerr > 0.")

    # ---- bounds/prior ----
    priors_lin, priors_log10 = _split_prior_specs(priors)
    names_all, bounds_all = build_bounds(model, priors=priors_lin, include_t_shift=True)
    bounds_all, log_set_all = _apply_log10_priors(names_all, bounds_all, priors_log10)
    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed)
    log_flags_samp = [n in log_set_all for n in names_samp]
    prior = MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp)

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

        theta_model, t_shift = _assemble_theta(sample_vec, names_samp, fixed, names_all)
        # Shift model time axis by t_shift and compare on observed t_obs.
        # Legacy convention:
        # y_model_shifted(t_obs) = y_model_raw(t_obs + t_shift)
        t_eval = t_obs + t_shift

        y_mod = predict_multiband(
            model=model,
            theta=theta_model,
            z=ctx.distance.get_z(),
            filters=filters,
            t_days=t_eval,
            band=band,
            y_kind=ctx.y_kind,
            interp_fill=interp_fill_fit,
            **model_kwargs_pred,
        )

        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        return lp + gaussian_lnlike(y_obs, y_mod, y_err)

    samples, logp, meta, sampler_used = _run_sampler(
        sampler=sampler,
        lnprob=lnprob,
        prior=prior,
        sampler_kwargs=sampler_kwargs,
    )

    meta.update(
        dict(
            model=model,
            y_kind=ctx.y_kind,
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            priors_input=dict(priors or {}),
            priors_linear=dict(priors_lin or {}),
            priors_log10=dict(priors_log10 or {}),
            log_prior_names=sorted([n for n in names_samp if n in log_set_all]),
            interp_fill_fit=interp_fill_fit,
            model_kwargs=model_kwargs_pred,
        )
    )

    return FitResult(
        model=model,
        ctx=ctx,
        sampler=sampler_used,
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
    z: Optional[float] = None,
    priors: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> FitResult:
    if priors and "T_floor" in priors:
        raise ValueError(
            f"`T_floor` is not a bolometric fit parameter in `fit_bol()`. "
            f"TransFit keeps an internal temperature floor of {_BOL_INTERNAL_T_FLOOR:.0f} K only for numerical stability."
        )
    if fixed and "T_floor" in fixed:
        raise ValueError(
            f"`T_floor` is not a bolometric fit parameter in `fit_bol()`. "
            f"TransFit keeps an internal temperature floor of {_BOL_INTERNAL_T_FLOOR:.0f} K only for numerical stability."
        )

    ctx = _context_from_fit_inputs(
        z=z,
        filters=None,
        y_kind="mag",
        require_filters=False,
    )
    sampler_kwargs = dict(sampler_kwargs or {})
    model_kwargs = dict(model_kwargs or {})
    model_kwargs_pred, interp_fill_fit = _split_fit_model_kwargs(model_kwargs)
    data = _apply_data_filter(data)

    t_obs = _as_1d_float(data.t_days, "data.t_days")
    y_obs = _as_1d_float(data.y, "data.y")
    y_err = _as_1d_float(data.yerr, "data.yerr")
    _check_same_length(t_days=t_obs, y=y_obs, yerr=y_err)

    if np.any(~np.isfinite(y_obs)) or np.any(~np.isfinite(y_err)) or np.any(y_err <= 0):
        raise ValueError("data.y and data.yerr must be finite and yerr > 0.")

    priors_lin, priors_log10 = _split_prior_specs(priors)
    names_all, bounds_all = build_bounds(model, priors=priors_lin, include_t_shift=True)
    bounds_all, log_set_all = _apply_log10_priors(names_all, bounds_all, priors_log10)
    # For bolometric fitting, T_floor stays internal and is not part of the fit state.
    if "T_floor" in names_all:
        i_tf = names_all.index("T_floor")
        names_all = [n for n in names_all if n != "T_floor"]
        bounds_all = np.asarray(bounds_all, float)
        bounds_all = np.delete(bounds_all, i_tf, axis=0)
        if "T_floor" in log_set_all:
            log_set_all.remove("T_floor")

    fixed = dict(fixed or {})

    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed)
    log_flags_samp = [n in log_set_all for n in names_samp]
    prior = MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp)

    def lnprob(sample_vec: np.ndarray) -> float:
        lp = prior.lnprior(sample_vec)
        if not np.isfinite(lp):
            return -np.inf

        theta_model, t_shift = _assemble_theta(sample_vec, names_samp, fixed, names_all)
        # Shift model time axis by t_shift and compare on observed t_obs.
        # Legacy convention:
        # y_model_shifted(t_obs) = y_model_raw(t_obs + t_shift)
        t_eval = t_obs + t_shift

        y_mod = predict_bol(
            model=model,
            theta=theta_model,
            z=ctx.distance.get_z(),
            t_days=t_eval,
            interp_fill=interp_fill_fit,
            **model_kwargs_pred,
        )

        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        return lp + gaussian_lnlike(y_obs, y_mod, y_err)

    samples, logp, meta, sampler_used = _run_sampler(
        sampler=sampler,
        lnprob=lnprob,
        prior=prior,
        sampler_kwargs=sampler_kwargs,
    )

    meta.update(
        dict(
            model=model,
            y_kind="bol",
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            priors_input=dict(priors or {}),
            priors_linear=dict(priors_lin or {}),
            priors_log10=dict(priors_log10 or {}),
            log_prior_names=sorted([n for n in names_samp if n in log_set_all]),
            interp_fill_fit=interp_fill_fit,
            model_kwargs=model_kwargs_pred,
            internal_t_floor=_BOL_INTERNAL_T_FLOOR,
        )
    )

    return FitResult(
        model=model,
        ctx=ctx,
        sampler=sampler_used,
        param_names=names_samp,
        fixed=fixed,
        all_param_names=names_all,
        samples=samples,
        log_prob=logp,
        meta=meta,
    )


__all__ = [
    "BolometricLC", "MultiBandLC",
    "BolometricData", "MultiBandData",
    "model_param_names", "param_template",
    "lightcurve_bol", "predict_bol",
    "lightcurve_multiband", "predict_multiband",
    "fit_bol", "fit_multiband",
]
