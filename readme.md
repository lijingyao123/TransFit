# Transfit

Transfit is a lightweight framework for transient light-curve modeling and Bayesian fitting.
It supports:
- Bolometric light curves
- Multi-band light curves (AB magnitude or flux density)
- Multiple samplers (`emcee`, `zeus`, `dynesty`)
- Linear and log-uniform priors

The full walkthrough is in `examples/tutorial.ipynb`.

## Project Layout
- `transfit/`: core package
- `transfit/models/`: physical light-curve engines
- `transfit/priors/`: default parameter names and bounds
- `transfit/samplers/`: sampler backends and result container
- `transfit/modules/`: plotting and I/O helpers
- `examples/`: tutorial notebooks
- `examples/data/`: sample datasets

## Environment
Run from repository root so `import transfit` works directly.

Main dependencies:
- `numpy`
- `matplotlib`
- `pandas`
- `numba`
- `astropy`

Optional dependencies:
- `corner` (for corner plots)
- `emcee`, `zeus`, `dynesty` (for fitting backends)

Optional sampler backends are imported lazily, so `import transfit` does not require all sampler packages to be installed.

## Standard Workflow

For most users, you only need to understand three things:
- choose a data container
- choose a model key
- call `tf.fit_bol(...)` or `tf.fit_multiband(...)`

### 1. Data containers
- `tf.BolometricData(t_days, y, yerr)`
- `tf.MultiBandData(t_days, band, y, yerr)`
- Both containers also accept optional `mask` and provide `.filtered()` helpers.

All arrays must have matching lengths; `yerr` must be finite and positive.

### 2. Distance and filters
- For bolometric fitting, pass `z=...` or `DL_cm=...` directly to `tf.fit_bol(...)`.
- For multi-band fitting, use `tf.fit_multiband(...)` and also pass `filters={"B": 6.8e14, "V": 5.5e14, ...}`.
- `y_kind` defaults to `"mag"` and only needs to be changed for flux fitting.

### 3. Advanced context API
`Context` / `Distance` are internal forward-model helpers. Standard fitting does not require them.

## Models and Parameter Order

Standard model keys:

| Model key | Physical meaning | Parameter order (`theta`) |
|---|---|---|
| `nickel` | Ni-powered model | `(M_ej, v_ej, M_Ni, x_Ni, kappa, kappa_gamma, T_floor)` |
| `sc_ni` | Shock-cooling + Ni | `(M_ej, v_ej, E_Th_in, M_Ni, R_0, x_Ni, kappa, kappa_gamma, T_floor)` |
| `magnetar` | Pure magnetar | `(M_ej, v_ej, P_ms, B14, kappa, kappa_gamma, T_floor)` |
| `magnetar_ni` | Magnetar + Ni | `(M_ej, v_ej, P_ms, B14, M_Ni, kappa, kappa_gamma, T_floor)` |
| `sc_magnetar` | Shock-cooling + magnetar | `(M_ej, v_ej, E_Th_in, P_ms, B14, R_0, kappa, kappa_gamma, T_floor)` |

Compatibility aliases are still accepted in code, but the names above are the recommended public API.

Built-in fixed assumptions:
- `nickel`: `E_Th_in=0`, `R_0=10 R_sun`
- `magnetar`: `E_Th_in=0`, `R_0=1 R_sun`
- `magnetar_ni`: `E_Th_in=0`, `R_0=1 R_sun`

## Parameter Glossary

| Parameter | Meaning | Unit |
|---|---|---|
| `M_ej` | Ejecta mass | `M_sun` |
| `v_ej` | Characteristic ejecta velocity | `1e9 cm s^-1` |
| `M_Ni` | Nickel-56 mass | `M_sun` |
| `E_Th_in` | Initial thermal energy scale | `1e49 erg` |
| `R_0` | Initial outer-radius scale | `R_sun` |
| `x_Ni` | Heating radius fraction | dimensionless `[0,1]` |
| `kappa` | Optical opacity | `cm^2 g^-1` |
| `kappa_gamma` | Gamma-ray opacity | `cm^2 g^-1` |
| `P_ms` | Magnetar initial spin period | `ms` |
| `B14` | Magnetar magnetic field scale | `1e14 G` |
| `T_floor` | Temperature floor for photosphere | `K` |
| `t_shift` | Time offset between model and observed timeline | `day` |

`t_shift` convention in fitting:
- Likelihood is evaluated as `model(t_obs + t_shift)`.
- Positive `t_shift` shifts the model curve to earlier observed times.

## Fitting API

### `tf.fit_bol(...)`
Key arguments:
- `data`: `BolometricData`
- `model`: model key
- `z` or `DL_cm`: distance input
- `priors`: bounds specification
- `fixed`: fixed parameter values
- `sampler`: optional advanced choice, default is `"emcee"`
- `sampler_kwargs`: optional advanced sampler config
- `include_t_shift`: optional advanced flag

Recommended standard usage:
- use `tf.fit_bol(...)` for bolometric data
- use `tf.fit_multiband(...)` for multi-band data
- use standard model keys
- use simple prior bounds `{"param": (lo, hi)}`
- keep the default sampler unless you have a reason to change it

### `tf.fit_multiband(...)`
Key differences from `fit_bol(...)`:
- `data` must be `MultiBandData`
- `filters` is required
- `y_kind` can be `"mag"` or `"flux"`

### Prior formats
Recommended:
- `"M_ej": (1.0, 8.0)`

Advanced formats still supported:
- Log-uniform: `"M_Ni": ("log10", -3.0, -0.3)`
- Dict style: `"kappa": {"bounds": (0.03, 0.3), "scale": "linear"}`

## Sampler Notes

Supported samplers:
- `emcee`
- `zeus`
- `dynesty`

Typical `sampler_kwargs`:
- `emcee` / `zeus`: `nwalkers`, `nsteps`, `burnin`, `thin`, `seed`, `progress`
- `dynesty`: `nlive`, `sample`, `bound`, `dlogz`, `maxiter`, `maxcall`, `seed`, `progress`, `nsamples`

## Result Object (`FitResult`)

`fit_bol` / `fit_multiband` returns `FitResult` with:
- `model`, `sampler`, `ctx`
- `param_names`: sampled/free parameters
- `all_param_names`: full parameter order
- `fixed`: fixed parameters
- `samples`: posterior samples, shape `(Ns, ndim)`
- `log_prob`: log posterior for each sample
- `meta`: run metadata (bounds, priors, solver config, etc.)

Recommended accessors:
- `res.best_fit`
- `res.best_params`
- `res.median_params`

Advanced accessors are also available if needed, such as `best_log_prob`, `best_sample`, and `best_theta`.

## Plotting

- `tf.plot.corner(res)`
- `tf.plot.fit_bol(...)`
- `tf.plot.fit_multiband(...)`

Plot notes:
- Fit plots default to no grid.
- `show_1sigma=True` draws posterior 16%-84% band.
- Corner labels include units and LaTeX formatting.

## Save and Load

- Save: `path = tf.save(res, path="mcmc_out/fit_demo.npz")`
- Load: `loaded = tf.load(path)`

Loaded objects are plain dictionaries and can be passed directly to plotting functions.

## Minimal Example

```python
import numpy as np
import transfit as tf

# Bolometric data
a = np.loadtxt("examples/data/sn1993j_lbol.txt")
t = a[:, 0] - a[:, 0].min()
data = tf.BolometricData(t_days=t, y=a[:, 1], yerr=a[:, 2])

# Quick fit
res = tf.fit_bol(
    data=data,
    model="sc_ni",
    z=0.001728,
    priors={
        "M_ej": (0.5, 8.0),
        "v_ej": (0.2, 3.0),
        "M_Ni": ("log10", -3.0, -0.2),
        "E_Th_in": (0.05, 8.0),
        "R_0": (10.0, 400.0),
    },
    fixed={"x_Ni": 0.2, "kappa": 0.12, "kappa_gamma": 0.03},
    sampler="emcee",
    sampler_kwargs=dict(nwalkers=32, nsteps=600, burnin=200, thin=5, seed=123),
)

print(res.best_fit)
fig = tf.plot.fit_bol(res, data=data, show_1sigma=True)
```

## Tutorial Notebook
- `examples/tutorial.ipynb`

## Contact
For questions about this project, please contact the following by email:

- Liangduan Liu ([liuld@ccnu.edu.cn])
- Yuhao Zhang ([zhangyh2001@foxmail.com])

## Citation
If you use this software in research, please cite:

```bibtex
@ARTICLE{2025ApJ...992...20L,
       author = {{Liu}, Liang-Duan and {Zhang}, Yu-Hao and {Yu}, Yun-Wei and {Du}, Ze-Xin and {Li}, Jing-Yao and {Wu}, Guang-Lei and {Dai}, Zi-Gao},
        title = "{TransFit: An Efficient Framework for Transient Light-curve Fitting with Time-dependent Radiative Diffusion}",
      journal = {\apj},
         year = 2025,
       volume = {992},
       number = {1},
          eid = {20},
        pages = {20},
          doi = {10.3847/1538-4357/adfed6},
archivePrefix = {arXiv},
       eprint = {2505.13825}
}
```

Some code and tutorial content were generated by Codex.
