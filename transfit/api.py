# transfit/api.py
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Dict, List, Optional, Sequence, Literal, Any, Tuple

import numpy as np

from .data import BolometricData, MultiBandData
from .modules.extinction import ExtinctionSpec, normalize_extinction, validate_extinction_spec
from .modules.filters import FilterProfile, normalize_filters, validate_filter_map
from .modules.interp import interp_fit
from .modules.likelihood import gaussian_lnlike_flux, gaussian_lnlike_for_observation
from .modules.photometry import evaluate_multiband_observer_output
from .modules.sed import BlackbodySED
from .model_registry import canonical_model_name, forward_param_defaults
from .samplers import FitResult, run_emcee, run_zeus, run_dynesty
from .priors import MixedBoundsPrior, build_bounds
from transfit.constants import DAY, MPC, PC


# -------------------------
# Internal forward metadata
# -------------------------

_BOL_INTERNAL_T_FLOOR = 1000.0
_FIT_T_MAX_DAYS_DEFAULT = 150.0
_FIT_T_MAX_DAYS_PADDING = 20.0


class _NonPhysicalModelOutput(ValueError):
    """Raised when a solved model state is not usable as a physical prediction."""


def _cosmo_luminosity_distance_cm(z: float) -> float:
    from astropy.cosmology import Planck15 as cosmo
    import astropy.units as u

    return cosmo.luminosity_distance(float(z)).to(u.cm).value


def _distance_cm_from_modulus(distance_modulus: float) -> float:
    return (10.0 ** ((float(distance_modulus) + 5.0) / 5.0)) * PC


def _distance_modulus_from_cm(distance_cm: float) -> float:
    d_pc = float(distance_cm) / PC
    return 5.0 * np.log10(d_pc) - 5.0


@dataclass(frozen=True)
class _Distance:
    z: Optional[float] = None
    DL_cm: Optional[float] = None
    source: Optional[str] = None

    def get_z(self) -> float:
        return float(self.z or 0.0)

    def get_DL_cm(self) -> float:
        if self.DL_cm is not None:
            dl = float(self.DL_cm)
            if self.z is not None:
                dl_cosmo = _cosmo_luminosity_distance_cm(float(self.z))
                if np.isfinite(dl_cosmo) and dl_cosmo > 0.0:
                    frac = abs(dl - dl_cosmo) / dl_cosmo
                    if frac > 0.05:
                        warnings.warn(
                            "Using a user-supplied explicit distance that differs from the Planck15 luminosity distance "
                            "implied by z by more than 5%. z is still used for time/frequency redshift terms.",
                            stacklevel=2,
                        )
            return dl
        if self.z is None:
            raise ValueError("Distance needs either an explicit distance or z.")
        return _cosmo_luminosity_distance_cm(float(self.z))


def _distance_from_public_inputs(
    *,
    z: Optional[float],
    distance_modulus: Optional[float],
    require_distance: bool,
) -> _Distance:
    z_norm = None if z is None else float(z)
    if z_norm is not None and not np.isfinite(z_norm):
        raise ValueError("z must be finite when provided.")
    if z_norm is not None and z_norm < 0.0:
        raise ValueError("z must be non-negative when provided.")

    dl_norm = None
    source = None
    if distance_modulus is not None:
        dl_norm = _distance_cm_from_modulus(float(distance_modulus))
        source = "distance_modulus"

    if dl_norm is not None and (not np.isfinite(dl_norm) or dl_norm <= 0.0):
        raise ValueError("Explicit distance must be positive and finite.")
    if require_distance and z_norm is None and dl_norm is None:
        raise ValueError(
            "Provide `z` or an explicit distance via `distance_modulus`."
        )
    if source is None and z_norm is not None:
        source = "from_z"
    return _Distance(z=z_norm, DL_cm=dl_norm, source=source)


@dataclass(frozen=True)
class _Context:
    """
    Internal context for forward-model evaluation.

    - Multi-band prediction: filters is required.
    """
    distance: _Distance
    filters: Optional[Dict[str, FilterProfile]] = None
    y_kind: Literal["mag", "flux"] = "mag"
    mag_system: Literal["ab", "vega"] = "ab"
    extinction: Optional[ExtinctionSpec] = None


def _effective_mag_system(y_kind: str, mag_system: str) -> str:
    return str(mag_system).strip().lower() if str(y_kind).strip().lower() == "mag" else "ab"


def _context_from_fit_inputs(
    *,
    z: Optional[float],
    distance_modulus: Optional[float],
    filters: Optional[Dict[str, Any]],
    y_kind: Literal["mag", "flux"],
    mag_system: Literal["ab", "vega"],
    extinction: Optional[Dict[str, Any] | ExtinctionSpec],
    require_filters: bool,
) -> _Context:
    """
    Build the internal Context used by fitting.

    Public fitting APIs accept direct scalar inputs instead of exposing
    Context/Distance as required user-facing concepts.
    """
    if require_filters and filters is None:
        raise ValueError(
            "filters is required for multiband fitting."
        )
    y_kind_n = str(y_kind).strip().lower()
    mag_system_n = str(mag_system).strip().lower()
    return _Context(
        distance=_distance_from_public_inputs(
            z=z,
            distance_modulus=distance_modulus,
            require_distance=require_filters,
        ),
        filters=None if filters is None else normalize_filters(
            filters,
            mag_system=_effective_mag_system(y_kind_n, mag_system_n),
        ),
        y_kind=y_kind_n,
        mag_system=mag_system_n,
        extinction=normalize_extinction(extinction),
    )


def _context_from_forward_inputs(
    *,
    z: Optional[float],
    distance_modulus: Optional[float],
    filters: Optional[Dict[str, Any]],
    y_kind: Literal["mag", "flux"],
    mag_system: Literal["ab", "vega"],
    extinction: Optional[Dict[str, Any] | ExtinctionSpec],
    require_filters: bool,
    require_distance: bool,
) -> _Context:
    """
    Build internal forward metadata for public prediction/lightcurve helpers.

    Public forward APIs accept direct scalar inputs instead of requiring
    Context/Distance objects from users.
    """
    if require_filters and filters is None:
        raise ValueError("filters is required for multiband forward calculations.")
    y_kind_n = str(y_kind).strip().lower()
    mag_system_n = str(mag_system).strip().lower()
    return _Context(
        distance=_distance_from_public_inputs(
            z=z,
            distance_modulus=distance_modulus,
            require_distance=require_distance,
        ),
        filters=None if filters is None else normalize_filters(
            filters,
            mag_system=_effective_mag_system(y_kind_n, mag_system_n),
        ),
        y_kind=y_kind_n,
        mag_system=mag_system_n,
        extinction=normalize_extinction(extinction),
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
    m = canonical_model_name(model, warn_legacy=False)
    if m in _ENGINE_CACHE:
        return _ENGINE_CACHE[m]

    if m == "nickel":
        from .models.nickel import NickelModel
        eng = NickelModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m == "magnetar":
        from .models.magnetar import MagnetarModel
        eng = MagnetarModel()
        _ENGINE_CACHE[m] = eng
        return eng

    if m == "magnetar_ni":
        from .models.magnetar_ni import MagNiModel
        eng = MagNiModel()
        _ENGINE_CACHE[m] = eng
        return eng

    raise ValueError(f"Unknown model='{model}'")


def model_param_names(model: str, *, include_t_shift: bool = False) -> List[str]:
    model = canonical_model_name(model, warn_legacy=True)
    names, _ = build_bounds(model, include_t_shift=include_t_shift)
    return list(names)


def param_template(
    model: str,
    *,
    include_t_shift: bool = False,
    fill_value: Any = None,
) -> Dict[str, Any]:
    model = canonical_model_name(model, warn_legacy=True)
    return {name: fill_value for name in model_param_names(model, include_t_shift=include_t_shift)}


def _model_vector_from_params(
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

    model = canonical_model_name(model, warn_legacy=False)
    defaults = forward_param_defaults(model)
    names = model_param_names(model, include_t_shift=False)
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise KeyError(f"Unknown parameter(s) for model='{model}': {unknown}. Allowed: {names}")

    missing = [
        n for n in names
        if n not in values and n not in defaults and not (allow_missing_tfloor and n == "T_floor")
    ]
    if missing:
        raise KeyError(f"Missing parameter(s) for model='{model}': {missing}. Required: {names}")

    model_vector = []
    for n in names:
        if n == "T_floor" and n not in values and allow_missing_tfloor:
            model_vector.append(_BOL_INTERNAL_T_FLOOR)
        elif n not in values and n in defaults:
            model_vector.append(float(defaults[n]))
        else:
            model_vector.append(float(values[n]))
    return tuple(model_vector)


_NONNEGATIVE_PARAMS = {"E_Th_in", "M_Ni", "t_shift"}
_POSITIVE_PARAMS = {"M_ej", "v_ej", "R_0", "kappa", "kappa_gamma", "T_floor", "P_ms", "B14"}
_UNIT_INTERVAL_PARAMS = {"x_Ni"}


def _model_values_from_vector(model: str, model_vector) -> Dict[str, float]:
    names = model_param_names(model, include_t_shift=False)
    values_t = tuple(model_vector)
    if len(values_t) != len(names):
        raise ValueError(
            f"Model parameter vector for model='{model}' must have length {len(names)}."
        )
    return {str(n): float(v) for n, v in zip(names, values_t)}


def _physical_constraint_reason(vals: Dict[str, float]) -> Optional[str]:
    """
    Return a short reason when a parameter set violates model-independent
    physical constraints, otherwise None.
    """
    values = {str(k): float(v) for k, v in dict(vals or {}).items()}

    for name, value in values.items():
        if not np.isfinite(value):
            return f"{name} must be finite."

    for name in sorted(_POSITIVE_PARAMS & set(values)):
        if not (values[name] > 0.0):
            return f"{name} must be > 0."

    for name in sorted(_NONNEGATIVE_PARAMS & set(values)):
        if not (values[name] >= 0.0):
            return f"{name} must be >= 0."

    for name in sorted(_UNIT_INTERVAL_PARAMS & set(values)):
        if not (0.0 <= values[name] <= 1.0):
            return f"{name} must be in [0, 1]."

    if "M_Ni" in values and "M_ej" in values:
        if values["M_Ni"] > values["M_ej"]:
            return "M_Ni must be <= M_ej."

    return None


def _validate_physical_values(vals: Dict[str, float]) -> None:
    reason = _physical_constraint_reason(vals)
    if reason is not None:
        raise ValueError(f"Physical parameter constraints are invalid: {reason}")


def _validate_physical_model_vector(model: str, model_vector) -> None:
    _validate_physical_values(_model_values_from_vector(model, model_vector))


def _resolve_forward_params(
    model: str,
    *,
    params: Optional[Dict[str, Any]],
    allow_missing_tfloor: bool,
):
    if params is not None:
        model_vector = _model_vector_from_params(model, params, allow_missing_tfloor=allow_missing_tfloor)
        _validate_physical_model_vector(model, model_vector)
        return model_vector
    raise ValueError("Provide `params` for forward-model evaluation.")


def _observer_days_to_rest_days(t_days_obs: float, z: float) -> float:
    """
    Public APIs accept observer-frame day scales.
    Internal engines solve in rest-frame / physical time.
    """
    return float(t_days_obs) / (1.0 + float(z))


def _solve_state(engine, model_vector, *, Nx: int, Ny: int, t_max_days_obs: float, z: float):
    t_max_days_rest = _observer_days_to_rest_days(t_max_days_obs, z)
    return engine.calculate_light_curve(model_vector, Nx=Nx, Ny=Ny, t_max_days=t_max_days_rest)


def _require_positive_finite(name: str, arr: np.ndarray) -> None:
    values = np.asarray(arr, float)
    if values.size == 0:
        raise _NonPhysicalModelOutput(f"Model produced an empty {name} grid.")
    bad = ~np.isfinite(values) | (values <= 0.0)
    if np.any(bad):
        first = float(values.reshape(-1)[int(np.where(bad.reshape(-1))[0][0])])
        raise _NonPhysicalModelOutput(
            f"Model produced non-positive or non-finite {name}; first invalid value is {first!r}."
        )


def _validate_solved_state(Lbol, Teff, Rph) -> None:
    _require_positive_finite("Lbol", Lbol)
    _require_positive_finite("Teff", Teff)
    _require_positive_finite("Rph", Rph)


def _t_grid_days_from_ts(t_s: np.ndarray, z: float) -> np.ndarray:
    # Public forward outputs use observer-frame days.
    return (np.asarray(t_s, float) * (1.0 + z)) / DAY


# -------------------------
# Forward model
# -------------------------

def lightcurve_bol(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    z: Optional[float] = None,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
) -> BolometricLC:
    """
    Return a bolometric light curve on an observer-frame time grid.

    User-facing time arguments are observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
        require_filters=False,
        require_distance=False,
    )
    engine = _get_engine(model)
    model_vector = _resolve_forward_params(model, params=params, allow_missing_tfloor=True)
    z = ctx.distance.get_z()
    t_s, Lbol, Teff, Rph = _solve_state(engine, model_vector, Nx=Nx, Ny=Ny, t_max_days_obs=t_max_days, z=z)
    _validate_solved_state(Lbol, Teff, Rph)
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
    z: Optional[float] = None,
    t_days: np.ndarray,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    interp_fill: Literal["edge", "nan", "raise"] = "nan",
) -> np.ndarray:
    """
    Predict a bolometric observable at observer-frame times `t_days`.

    `t_days` and `t_max_days` are interpreted in observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
        require_filters=False,
        require_distance=False,
    )
    engine = _get_engine(model)
    model_vector = _resolve_forward_params(model, params=params, allow_missing_tfloor=True)
    z = ctx.distance.get_z()
    t_s, Lbol, Teff, Rph = _solve_state(engine, model_vector, Nx=Nx, Ny=Ny, t_max_days_obs=t_max_days, z=z)
    _validate_solved_state(Lbol, Teff, Rph)
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
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    bands: Sequence[str],
    y_kind: Literal["mag", "flux"] = "mag",
    mag_system: Literal["ab", "vega"] = "ab",
    extinction: Optional[Dict[str, Any] | ExtinctionSpec] = None,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    sed=None,
) -> MultiBandLC:
    """
    Return a multi-band light curve on an observer-frame time grid.

    User-facing time arguments are observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=distance_modulus,
        filters=filters,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
        require_filters=True,
        require_distance=True,
    )
    sed = sed or BlackbodySED()
    bands = [_norm_band(b) for b in list(bands)]
    filter_map = validate_filter_map(
        ctx.filters or {},
        used_bands=bands,
        mag_system=_effective_mag_system(ctx.y_kind, ctx.mag_system),
    )
    z = ctx.distance.get_z()
    extinction_spec = validate_extinction_spec(
        ctx.extinction,
        used_bands=bands,
        filter_map=filter_map,
        z=z,
    )

    engine = _get_engine(model)
    model_vector = _resolve_forward_params(model, params=params, allow_missing_tfloor=False)
    DL_cm = ctx.distance.get_DL_cm()
    t_s, Lbol, Teff, Rph = _solve_state(engine, model_vector, Nx=Nx, Ny=Ny, t_max_days_obs=t_max_days, z=z)
    _validate_solved_state(Lbol, Teff, Rph)
    t_days = _t_grid_days_from_ts(t_s, z=z)
    y_grid = evaluate_multiband_observer_output(
        sed=sed,
        filter_map=filter_map,
        bands=bands,
        Teff_K=np.asarray(Teff, float),
        R_cm=np.asarray(Rph, float),
        DL_cm=DL_cm,
        z=z,
        y_kind=ctx.y_kind,
        mag_system=ctx.mag_system,
        extinction=extinction_spec,
    )

    y = {b: np.asarray(y_grid[i], float).copy() for i, b in enumerate(bands)}
    return MultiBandLC(t_days=np.asarray(t_days, float), bands=bands, y=y)


def predict_multiband(
    *,
    model: str,
    params: Optional[Dict[str, Any]] = None,
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    t_days: np.ndarray,
    band: np.ndarray,
    y_kind: Literal["mag", "flux"] = "mag",
    mag_system: Literal["ab", "vega"] = "ab",
    extinction: Optional[Dict[str, Any] | ExtinctionSpec] = None,
    Nx: int = 100,
    Ny: int = 1000,
    t_max_days: float = 150.0,
    interp_fill: Literal["edge", "nan", "raise"] = "nan",
    sed=None,
) -> np.ndarray:
    """
    Predict multi-band observables at observer-frame times `t_days`.

    `t_days` and `t_max_days` are interpreted in observer-frame days.
    Internal model evolution is solved in rest-frame time.
    """
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_forward_inputs(
        z=z,
        distance_modulus=distance_modulus,
        filters=filters,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
        require_filters=True,
        require_distance=True,
    )
    sed = sed or BlackbodySED()
    t_days = np.asarray(t_days, float).reshape(-1)

    # Keep band case; only strip whitespace.
    band = np.asarray([_norm_band(b) for b in np.asarray(band).reshape(-1)], dtype=object)
    _check_same_length(t_days=t_days, band=band)

    uniq = sorted(set(band.tolist()))
    filter_map = validate_filter_map(
        ctx.filters or {},
        used_bands=uniq,
        mag_system=_effective_mag_system(ctx.y_kind, ctx.mag_system),
    )
    z = ctx.distance.get_z()
    extinction_spec = validate_extinction_spec(
        ctx.extinction,
        used_bands=uniq,
        filter_map=filter_map,
        z=z,
    )

    engine = _get_engine(model)
    model_vector = _resolve_forward_params(model, params=params, allow_missing_tfloor=False)
    DL_cm = ctx.distance.get_DL_cm()
    t_s, Lbol, Teff, Rph = _solve_state(engine, model_vector, Nx=Nx, Ny=Ny, t_max_days_obs=t_max_days, z=z)
    _validate_solved_state(Lbol, Teff, Rph)
    t_grid_days = _t_grid_days_from_ts(t_s, z=z)
    y_grid = evaluate_multiband_observer_output(
        sed=sed,
        filter_map=filter_map,
        bands=uniq,
        Teff_K=np.asarray(Teff, float),
        R_cm=np.asarray(Rph, float),
        DL_cm=DL_cm,
        z=z,
        y_kind=ctx.y_kind,
        mag_system=ctx.mag_system,
        extinction=extinction_spec,
    )
    itp_yscale = "linear" if ctx.y_kind == "mag" else "log10"

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


def _assemble_model_params(
    sample_vec: np.ndarray,
    names_samp: List[str],
    fixed: Dict[str, float],
    names_all: List[str],
):
    vals = _param_values_from_sample(sample_vec, names_samp, fixed)
    return _assemble_model_params_from_values(vals, names_all)


def _param_values_from_sample(
    sample_vec: np.ndarray,
    names_samp: List[str],
    fixed: Dict[str, float],
) -> Dict[str, float]:
    vals = {str(k): float(v) for k, v in dict(fixed or {}).items()}
    vals.update({str(k): float(v) for k, v in zip(names_samp, np.asarray(sample_vec, float))})
    return vals


def _physical_constraints_lnprior(vals: Dict[str, float]) -> float:
    """
    Model-independent physical constraints that cannot be expressed as
    independent box priors.
    """
    return -np.inf if _physical_constraint_reason(vals) is not None else 0.0


def _validate_fixed_physical_constraints(fixed: Dict[str, float]) -> None:
    reason = _physical_constraint_reason(fixed)
    if reason is not None:
        raise ValueError(f"Fixed physical constraints are invalid: {reason}")


def _validate_sampling_bounds_physical_constraints(names: Sequence[str], bounds: np.ndarray) -> None:
    b = np.asarray(bounds, float)
    for name, (lo, hi) in zip([str(n) for n in names], b):
        if name in _POSITIVE_PARAMS and lo <= 0.0:
            raise ValueError(f"Prior bounds for '{name}' must be > 0.")
        if name in _NONNEGATIVE_PARAMS and lo < 0.0:
            raise ValueError(f"Prior bounds for '{name}' must be >= 0.")
        if name in _UNIT_INTERVAL_PARAMS and (lo < 0.0 or hi > 1.0):
            raise ValueError(f"Prior bounds for '{name}' must stay within [0, 1].")


def _assemble_model_params_from_values(
    vals: Dict[str, float],
    names_all: List[str],
):
    t_shift = float(vals.get("t_shift", 0.0))

    model_params: Dict[str, float] = {}
    for n in names_all:
        if n == "t_shift":
            continue
        if n not in vals:
            raise KeyError(
                f"Missing parameter '{n}'. "
                f"Either provide it in priors (to sample) or in fixed."
            )
        model_params[str(n)] = float(vals[n])

    return model_params, t_shift


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


def _t_shift_upper_for_fit(names_all: Sequence[str], bounds_all: np.ndarray, fixed: Dict[str, float]) -> float:
    if "t_shift" in fixed:
        return float(fixed["t_shift"])
    names = [str(n) for n in names_all]
    if "t_shift" not in names:
        return 0.0
    idx = names.index("t_shift")
    return float(np.asarray(bounds_all, float)[idx, 1])


def _resolve_fit_t_max_days(
    model_kwargs: Dict[str, Any],
    *,
    t_obs: np.ndarray,
    names_all: Sequence[str],
    bounds_all: np.ndarray,
    fixed: Dict[str, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ensure fit-time model grids cover all possible t_obs + t_shift values.

    Explicit user values are accepted when they cover the allowed time range.
    Automatic values add a small padding for interpolation/plot reuse.
    """
    mk = dict(model_kwargs or {})
    t_max_obs = float(np.nanmax(np.asarray(t_obs, float)))
    t_shift_upper = max(0.0, _t_shift_upper_for_fit(names_all, bounds_all, fixed))
    required = max(0.0, t_max_obs + t_shift_upper)

    if "t_max_days" in mk:
        try:
            user_t_max = float(mk["t_max_days"])
        except Exception as exc:
            raise ValueError("model_kwargs['t_max_days'] must be a positive finite number.") from exc
        if not np.isfinite(user_t_max) or user_t_max <= 0.0:
            raise ValueError("model_kwargs['t_max_days'] must be a positive finite number.")
        if user_t_max + 1e-12 < required:
            raise ValueError(
                "model_kwargs['t_max_days'] is too small for data.t_days and the allowed t_shift range. "
                f"Need at least {required:.6g} observer-frame days, got {user_t_max:.6g}."
            )
        mk["t_max_days"] = user_t_max
        return mk, dict(
            t_max_days=user_t_max,
            t_max_days_auto=False,
            t_max_days_required=required,
            t_shift_upper=t_shift_upper,
        )

    auto_t_max = max(_FIT_T_MAX_DAYS_DEFAULT, required + _FIT_T_MAX_DAYS_PADDING)
    mk["t_max_days"] = float(auto_t_max)
    return mk, dict(
        t_max_days=float(auto_t_max),
        t_max_days_auto=True,
        t_max_days_required=required,
        t_shift_upper=t_shift_upper,
        t_max_days_padding=_FIT_T_MAX_DAYS_PADDING,
    )


def fit_multiband(
    *,
    data: MultiBandData,
    model: str,
    z: Optional[float] = None,
    distance_modulus: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None,
    y_kind: Literal["mag", "flux"] = "mag",
    mag_system: Literal["ab", "vega"] = "ab",
    extinction: Optional[Dict[str, Any] | ExtinctionSpec] = None,
    priors: Optional[Dict[str, Any]] = None,
    fixed: Optional[Dict[str, float]] = None,
    sampler: str = "emcee",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> FitResult:
    model = canonical_model_name(model, warn_legacy=True)
    ctx = _context_from_fit_inputs(
        z=z,
        distance_modulus=distance_modulus,
        filters=filters,
        y_kind=y_kind,
        mag_system=mag_system,
        extinction=extinction,
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
    _validate_sampling_bounds_physical_constraints(names_all, bounds_all)
    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed)
    _validate_fixed_physical_constraints(fixed)
    model_kwargs_pred, tmax_meta = _resolve_fit_t_max_days(
        model_kwargs_pred,
        t_obs=t_obs,
        names_all=names_all,
        bounds_all=bounds_all,
        fixed=fixed,
    )
    log_flags_samp = [n in log_set_all for n in names_samp]
    prior = MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp)

    # ---- band check (case-sensitive) ----
    if ctx.filters is None:
        raise ValueError("ctx.filters is required for multiband. For bolometric you can omit it.")
    uniq_b = sorted(set(band.tolist()))
    filters = validate_filter_map(
        ctx.filters,
        used_bands=uniq_b,
        mag_system=_effective_mag_system(ctx.y_kind, ctx.mag_system),
    )
    extinction_spec = validate_extinction_spec(
        ctx.extinction,
        used_bands=uniq_b,
        filter_map=filters,
        z=ctx.distance.get_z(),
    )

    # ---- lnprob ----
    def lnprob(sample_vec: np.ndarray) -> float:
        lp = prior.lnprior(sample_vec)
        if not np.isfinite(lp):
            return -np.inf

        vals = _param_values_from_sample(sample_vec, names_samp, fixed)
        lp_phys = _physical_constraints_lnprior(vals)
        if not np.isfinite(lp_phys):
            return -np.inf

        model_params, t_shift = _assemble_model_params_from_values(vals, names_all)
        # Shift model time axis by t_shift and compare on observed t_obs.
        # Legacy convention:
        # y_model_shifted(t_obs) = y_model_raw(t_obs + t_shift)
        t_eval = t_obs + t_shift

        try:
            y_mod = predict_multiband(
                model=model,
                params=model_params,
                z=ctx.distance.z,
                distance_modulus=None if ctx.distance.DL_cm is None else _distance_modulus_from_cm(float(ctx.distance.DL_cm)),
                filters=filters,
                t_days=t_eval,
                band=band,
                y_kind=ctx.y_kind,
                mag_system=ctx.mag_system,
                extinction=extinction_spec,
                interp_fill=interp_fill_fit,
                **model_kwargs_pred,
            )
        except _NonPhysicalModelOutput:
            return -np.inf

        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        return lp + lp_phys + gaussian_lnlike_for_observation(
            y_kind=ctx.y_kind,
            y_obs=y_obs,
            y_model=y_mod,
            y_err=y_err,
        )

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
            mag_system=ctx.mag_system,
            names_all=names_all,
            bounds_all=np.asarray(bounds_all, float),
            bounds_samp=np.asarray(bounds_samp, float),
            priors_input=dict(priors or {}),
            priors_linear=dict(priors_lin or {}),
            priors_log10=dict(priors_log10 or {}),
            log_prior_names=sorted([n for n in names_samp if n in log_set_all]),
            interp_fill_fit=interp_fill_fit,
            model_kwargs=model_kwargs_pred,
            t_max_days_policy=tmax_meta,
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
    model = canonical_model_name(model, warn_legacy=True)
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
        distance_modulus=None,
        filters=None,
        y_kind="mag",
        mag_system="ab",
        extinction=None,
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
    _validate_sampling_bounds_physical_constraints(names_all, bounds_all)

    fixed = dict(fixed or {})

    names_samp, bounds_samp, fixed = _split_sampling(names_all, bounds_all, fixed=fixed)
    _validate_fixed_physical_constraints(fixed)
    model_kwargs_pred, tmax_meta = _resolve_fit_t_max_days(
        model_kwargs_pred,
        t_obs=t_obs,
        names_all=names_all,
        bounds_all=bounds_all,
        fixed=fixed,
    )
    log_flags_samp = [n in log_set_all for n in names_samp]
    prior = MixedBoundsPrior(bounds=bounds_samp, param_names=names_samp, log_flags=log_flags_samp)

    def lnprob(sample_vec: np.ndarray) -> float:
        lp = prior.lnprior(sample_vec)
        if not np.isfinite(lp):
            return -np.inf

        vals = _param_values_from_sample(sample_vec, names_samp, fixed)
        lp_phys = _physical_constraints_lnprior(vals)
        if not np.isfinite(lp_phys):
            return -np.inf

        model_params, t_shift = _assemble_model_params_from_values(vals, names_all)
        # Shift model time axis by t_shift and compare on observed t_obs.
        # Legacy convention:
        # y_model_shifted(t_obs) = y_model_raw(t_obs + t_shift)
        t_eval = t_obs + t_shift

        try:
            y_mod = predict_bol(
                model=model,
                params=model_params,
                z=ctx.distance.get_z(),
                t_days=t_eval,
                interp_fill=interp_fill_fit,
                **model_kwargs_pred,
            )
        except _NonPhysicalModelOutput:
            return -np.inf

        if np.any(~np.isfinite(y_mod)):
            return -np.inf

        return lp + lp_phys + gaussian_lnlike_flux(y_obs, y_mod, y_err)

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
            t_max_days_policy=tmax_meta,
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
