# TransFit API and Parameter Reference

This document describes the stable public Python interface of TransFit. The
README and tutorial intentionally show only minimal examples; this file is the
place for argument meanings, model parameters, result fields, and advanced
options.

Chinese version: [中文 API 和参数参考](api_reference_chinese.md).

All public time inputs and outputs use **observer-frame days**. The physical
models are solved internally in rest-frame time and converted back to the
observer frame at the API boundary.

## Public Entry Points

| Category | Entry points |
|---|---|
| Data containers | `BolometricData`, `MultiBandData` |
| Model inspection | `model_param_names(model)`, `param_template(model)` |
| Forward light curves | `lightcurve_bol(...)`, `lightcurve_multiband(...)` |
| Interpolated predictions | `predict_bol(...)`, `predict_multiband(...)` |
| Fitting | `fit_bol(...)`, `fit_multiband(...)` |
| Result I/O | `save(res, path=None)`, `load(path, trusted=False)` |
| Plotting | `transfit.plot.fit_bol`, `transfit.plot.fit_multiband`, `transfit.plot.corner` |

## Model Names and Parameters

Accepted canonical model names are `nickel`, `magnetar`, `magnetar_ni`, and
`csm`. Some aliases are accepted for backward compatibility, but new scripts
should use the canonical names.

### `nickel`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, \(M_\odot\) |
| `v_ej` | ejecta velocity, \(10^9\,{\rm cm\,s^{-1}}\) |
| `E_Th_in` | initial thermal energy, \(10^{49}\,{\rm erg}\) |
| `M_Ni` | nickel mass, \(M_\odot\) |
| `R_0` | initial radius, \(R_\odot\) |
| `x_Ni` | nickel mixing coordinate, dimensionless |
| `kappa` | optical opacity, \({\rm cm^2\,g^{-1}}\) |
| `kappa_gamma` | gamma-ray opacity, \({\rm cm^2\,g^{-1}}\) |
| `T_floor` | temperature floor, K |

### `magnetar`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, \(M_\odot\) |
| `v_ej` | ejecta velocity, \(10^9\,{\rm cm\,s^{-1}}\) |
| `E_Th_in` | initial thermal energy, \(10^{49}\,{\rm erg}\) |
| `P_ms` | magnetar spin period, ms |
| `B14` | magnetar dipole field, \(10^{14}\,{\rm G}\) |
| `R_0` | initial radius, \(R_\odot\) |
| `kappa` | optical opacity, \({\rm cm^2\,g^{-1}}\) |
| `kappa_gamma` | gamma-ray opacity, \({\rm cm^2\,g^{-1}}\) |
| `T_floor` | temperature floor, K |

### `magnetar_ni`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, \(M_\odot\) |
| `v_ej` | ejecta velocity, \(10^9\,{\rm cm\,s^{-1}}\) |
| `P_ms` | magnetar spin period, ms |
| `B14` | magnetar dipole field, \(10^{14}\,{\rm G}\) |
| `M_Ni` | nickel mass, \(M_\odot\) |
| `kappa` | optical opacity, \({\rm cm^2\,g^{-1}}\) |
| `kappa_gamma` | gamma-ray opacity, \({\rm cm^2\,g^{-1}}\) |
| `T_floor` | temperature floor, K |

### `csm`

| Parameter | Meaning and unit |
|---|---|
| `M_ej` | ejecta mass, \(M_\odot\) |
| `E_sn` | explosion energy, \(10^{51}\,{\rm erg}\) |
| `M_csm` | circumstellar-material mass, \(M_\odot\) |
| `R_csm_out` | outer CSM radius, \(R_\odot\) |
| `kappa` | optical opacity, \({\rm cm^2\,g^{-1}}\) |
| `s` | CSM density power-law index |
| `eps_sh` | shock radiation efficiency |
| `T_floor` | temperature floor, K |

The optional fit parameter `t_shift` is appended when
`include_t_shift=True` or during fitting. It is constrained to be non-negative.
In fitting, the model is evaluated at:

```text
t_eval = t_obs + t_shift
```

A positive `t_shift` means that the model start is earlier than the user's
observational zero point.

## Data Containers

### `BolometricData`

```python
tf.BolometricData(t_days, y, yerr, mask=None)
```

`BolometricData` stores bolometric observations.

| Field | Meaning |
|---|---|
| `t_days` | observer-frame time in days |
| `y` | bolometric luminosity, \({\rm erg\,s^{-1}}\) |
| `yerr` | one-sigma uncertainty in the same units as `y` |
| `mask` | optional boolean mask; only masked-in points are used |

Unmasked luminosities and uncertainties must be positive and finite for
fitting.

### `MultiBandData`

```python
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

`MultiBandData` stores multi-band photometry.

| Field | Meaning |
|---|---|
| `t_days` | observer-frame time in days |
| `band` | band label for each point |
| `y` | magnitude if `y_kind="mag"`, flux density if `y_kind="flux"` |
| `yerr` | one-sigma uncertainty in the same units as `y` |
| `mask` | optional boolean mask; only masked-in points are used |

## Forward and Prediction Calls

Bolometric forward call:

```python
tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    t_max_days=150.0,
    solver_kwargs=None,
)
```

Returns `BolometricLC` with:

- `t_days`
- `Lbol`
- `Teff`
- `Rph`

Multi-band forward call:

```python
tf.lightcurve_multiband(
    model="nickel",
    params=params,
    z=0.001728,
    distance_modulus=None,
    filters=filters,
    bands=["B", "V"],
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    sed=None,
    t_max_days=150.0,
    solver_kwargs=None,
)
```

Returns `MultiBandLC` with:

- `t_days`
- `bands`
- `y[band]`

`predict_bol` and `predict_multiband` evaluate the same models at
user-supplied observer-frame times. `interp_fill` may be `"nan"`, `"raise"`,
or `"edge"` for prediction calls. During fitting, `"edge"` is rejected to avoid
silently extrapolating outside the model grid.

## Fitting Calls

Bolometric fit:

```python
res = tf.fit_bol(
    data=data,
    model="nickel",
    z=0.001728,
    priors=priors,
    fixed=fixed,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)
```

Multi-band fit:

```python
res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.001728,
    distance_modulus=None,
    filters=filters,
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    priors=priors,
    fixed=fixed,
    sampler="emcee",
    sed=None,
    sampler_kwargs=None,
    model_kwargs=None,
)
```

`priors` maps parameter names to bounds. A linear uniform prior uses
`(lo, hi)`. A base-10 log-uniform prior uses `("log10", lo, hi)`, where `lo`
and `hi` are bounds in log10 space.

`fixed` maps parameter names to fixed values. Any model parameter not supplied
in `fixed` is sampled using its default bounds or the bounds supplied in
`priors`.

### `sigma_int`

`sigma_int` is a likelihood nuisance parameter, not a physical model
parameter. It may be fixed or sampled through `fixed` and `priors`.

| Observation space | Meaning |
|---|---|
| `y_kind="mag"` | additional magnitude scatter |
| `y_kind="flux"` | converted to fractional flux scatter using \(0.4\ln(10)\sigma_{\rm int}\) |

## Keyword Dictionaries

### `sampler_kwargs`

Common `emcee` and `zeus` keys:

| Key | Meaning |
|---|---|
| `nwalkers` | number of walkers |
| `nsteps` | production chain length |
| `burnin` | burn-in steps before production |
| `thin` | thinning factor |
| `seed` | random seed |
| `init` | initial-position mode or array |
| `pool` | user-supplied parallel pool |
| `progress` | show sampler progress |

Common `dynesty` keys:

| Key | Meaning |
|---|---|
| `nlive` | number of live points |
| `sample` | dynesty sampling method |
| `bound` | dynesty bounding method |
| `dlogz` | stopping tolerance |
| `maxiter` | maximum iterations |
| `maxcall` | maximum likelihood calls |
| `seed` | random seed |
| `progress` | show sampler progress |
| `nsamples` | number of posterior samples returned |
| `add_live` | include live points in posterior |
| `pool` | user-supplied parallel pool |
| `queue_size` | dynesty queue size |

### `model_kwargs`

Fit-time model options are passed through `model_kwargs`.

| Key | Meaning |
|---|---|
| `t_max_days` | observer-frame model duration in days |
| `interp_fill` | interpolation fill policy; `"edge"` is not allowed during fitting |
| `solver_kwargs` | advanced numerical-grid options |

If `t_max_days` is omitted, TransFit chooses a value large enough to cover the
data and the allowed `t_shift` range.

### `solver_kwargs`

`solver_kwargs` is the advanced numerical-grid interface.

| Key | Default | Meaning |
|---|---:|---|
| `Nx` | `100` | spatial/grid resolution parameter |
| `Ny` | `1000` | time/grid resolution parameter |

Both values must be positive integers. Beginner examples intentionally do not
expose these controls.

## SED Choices

The default multi-band SED is `BlackbodySED`.

```python
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED

sed = BlackbodySED()
sed = CutoffBlackbodySED(
    cutoff_wavelength_A=3000.0,
    uv_slope=2.0,
    min_factor=0.0,
)
```

`CutoffBlackbodySED` applies a short-wavelength cutoff:

```text
L_nu_cutoff = C(lambda_rest) * L_nu_blackbody
```

with:

```text
C(lambda) = 1                                      for lambda >= lambda_cut
C(lambda) = max(f_min, (lambda/lambda_cut)^a)      for lambda < lambda_cut
```

where:

| Symbol | API parameter |
|---|---|
| `lambda_cut` | `cutoff_wavelength_A` |
| `a` | `uv_slope` |
| `f_min` | `min_factor` |

Set `min_factor=0` for a pure power-law cutoff.

## FitResult Fields

`fit_bol` and `fit_multiband` return a `FitResult`.

| Field/property | Meaning |
|---|---|
| `res.best_params` | rounded best-fit parameter dictionary |
| `res.best_params_raw` | full-precision best-fit parameter dictionary |
| `res.median_params` | posterior median parameter dictionary |
| `res.best_fit` | compact record with parameters, errors, best log probability, and best sample |
| `res.best_index` | index of the best posterior sample |
| `res.best_log_prob` | best log posterior value |
| `res.best_sample` | raw best sample vector in `res.param_names` order |
| `res.samples` | flattened posterior samples |
| `res.log_prob` | log posterior values |
| `res.meta` | sampler, prior, model, SED, and context metadata |

## Citation Rules

All model use should cite the TransFit software paper. The `csm` model should
additionally cite the TransFit-CSM paper. See
[model_citations.md](model_citations.md) for BibTeX entries and
model-specific guidance.
