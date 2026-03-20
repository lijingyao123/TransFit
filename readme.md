# TransFit

---

<table>
<tr>
<td width="42%" align="center" valign="top">
  <img src="docs/TransFit_logo.png" width="420" alt="TransFit logo">
</td>
<td width="58%" valign="top">

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-MCMC-C0392B">
  <img alt="Models" src="https://img.shields.io/badge/Models-Diffusion%20Powered-1F4E79">
  <img alt="Data" src="https://img.shields.io/badge/Data-Bolometric%20%7C%20Multi--band-2E8B57">
  <img alt="Acceleration" src="https://img.shields.io/badge/Acceleration-Numba-00A3E0">
</p>

<p>
  <a href="examples/tutorial.ipynb">Tutorial Notebook</a> |
  <a href="examples/data">Example Data</a> |
  <a href="#usage">Usage</a> |
  <a href="https://doi.org/10.3847/1538-4357/adfed6">Paper</a>
</p>

<p><strong>TransFit</strong> is a fast and physically motivated <strong>light-curve fitting framework</strong> for astronomical transients such as supernovae.
It numerically solves the <strong>time-dependent radiative diffusion equation</strong> in expanding ejecta and performs <strong>Bayesian parameter inference using MCMC sampling</strong>, enabling efficient fitting of observed transient light curves.</p>

<p>Compared with traditional <strong>semi-analytical models</strong> (e.g., Arnett-like models), which rely on simplified and time-independent temperature structures, TransFit directly solves the diffusion equation. This allows the model to capture <strong>time-dependent temperature evolution, non-uniform heating distributions, and the transition from shock-cooling to radioactive-powered emission</strong>, while maintaining <strong>computational speeds comparable to semi-analytical approaches</strong>.</p>

<p>By combining <strong>physical realism with efficient MCMC fitting</strong>, TransFit provides a practical tool for rapid light-curve modeling and parameter inference in the era of large time-domain surveys.</p>

</td>
</tr>
</table>

## Features

- Physically motivated light-curve fitting based on the time-dependent radiative diffusion equation.
- Supports both `bolometric` and `multi-band` transient light-curve fitting.
- Fast Bayesian inference with MCMC samplers such as `emcee`, `zeus`, and `dynesty`.
- Simple public workflow centered on `tf.BolometricData(...)`, `tf.MultiBandData(...)`, `tf.fit_bol(...)`, and `tf.fit_multiband(...)`.
- Direct forward-model helpers accept named `params={...}` dictionaries instead of opaque parameter tuples.
- Built-in result inspection, fit plotting, corner plotting, and save/load helpers.
- Internal solver runs in CGS units, while the public time interface uses observer-frame days for easier scientific use.

## Installation

TransFit currently runs from source. After cloning the repository, use it from the repository root so that `import transfit` works directly.

Requirements:
- Python `3.10+`
- `numpy`
- `matplotlib`
- `pandas`
- `numba`
- `astropy`

Optional packages:
- `corner` for posterior corner plots
- `emcee`, `zeus`, `dynesty` for fitting backends

Notes:
- Optional sampler backends are imported lazily, so `import transfit` does not require all sampler packages to be installed.
- Public fitting APIs use `z` directly.
- Public forward-model helpers also use direct inputs such as `z` and `filters`.

## Usage


### Light Curve Calculation

You can either generate a theoretical light curve directly from model parameters or plot a fitted result.

If you are unsure about the required parameter names for a model:

```python
tf.model_param_names("sc_ni")
tf.param_template("sc_ni")
```

#### Draw a bolometric light curve directly

```python
import matplotlib.pyplot as plt
import transfit as tf

params_scni = {
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

bol = tf.lightcurve_bol(
    model="sc_ni",
    params=params_scni,
    z=0.001728,
    t_max_days=180.0,
)

fig, ax = plt.subplots()
ax.plot(bol.t_days, bol.Lbol)
ax.set_yscale("log")
ax.set_xlabel("Observer time (days)")
ax.set_ylabel("Bolometric luminosity (erg s$^{-1}$)")
```
<p align="center">
  <img src="docs/lightcurve_bol.png" width="47%" alt="Example bolometric light curve">
</p>

#### Draw a multi-band light curve directly

```python
import matplotlib.pyplot as plt
import transfit as tf

filters = {
    "B": 6.8e14,
    "V": 5.5e14,
    "R": 4.7e14,
    "I": 3.9e14,
}

params_ni = {
    "M_ej": 3.0,
    "v_ej": 1.0,
    "M_Ni": 0.08,
    "x_Ni": 0.2,
    "kappa": 0.12,
    "kappa_gamma": 0.03,
    "T_floor": 3000.0,
}

mb = tf.lightcurve_multiband(
    model="nickel",
    params=params_ni,
    z=0.001728,
    filters=filters,
    bands=["B", "V", "R", "I"],
    y_kind="mag",
    t_max_days=180.0,
)

fig, ax = plt.subplots()
for b in mb.bands:
    ax.plot(mb.t_days, mb.y[b], label=b)
ax.invert_yaxis()
ax.set_xlabel("Observer time (days)")
ax.set_ylabel("AB magnitude")
ax.legend()
```

<p align="center">
  <img src="docs/lightcurve_bol.png" width="47%" alt="Example bolometric light curve">
</p>

#### Plot a fitted result

```python
fig_bol = tf.plot.fit_bol(res_bol, data=data_bol, show_1sigma=True)
fig_mb = tf.plot.fit_multiband(res_mb, data=data_mb, show_1sigma=True)
```

#### Plot the posterior corner

```python
corner_fig = tf.plot.corner(res)
```

Common result accessors:
- `res.best_fit`
- `res.best_params`
- `res.median_params`
- `res.best_log_prob`

### 2. How to fit data

#### Prepare the data

Choose the data container that matches your observations:

- `tf.BolometricData(t_days, y, yerr, mask=None)`
- `tf.MultiBandData(t_days, band, y, yerr, mask=None)`

Rules:
- all arrays must have matching lengths
- `yerr` must be finite and positive
- `mask` is optional and is applied automatically during fitting
- `fit_bol(...)` does not fit `T_floor`; an internal `1000 K` floor is kept only for numerical stability

#### Fit a bolometric light curve

```python
import numpy as np
import transfit as tf

a = np.loadtxt("examples/data/sn1993j_lbol.txt")
t = a[:, 0] - a[:, 0].min()

data = tf.BolometricData(
    t_days=t,
    y=a[:, 1],
    yerr=a[:, 2],
)

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
    fixed={
        "x_Ni": 0.2,
        "kappa": 0.12,
        "kappa_gamma": 0.03,
    },
    sampler="emcee",
    sampler_kwargs=dict(
        nwalkers=32,
        nsteps=600,
        burnin=200,
        thin=5,
        seed=123,
        progress=True,
    ),
)
```

#### Fit a multi-band light curve

```python
import transfit as tf

data = tf.MultiBandData(
    t_days=t_days,
    band=band,
    y=y,
    yerr=yerr,
)

filters = {
    "B": 6.8e14,
    "V": 5.5e14,
    "R": 4.7e14,
    "I": 3.9e14,
}

res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.001728,
    filters=filters,
    priors={
        "M_ej": (1.0, 5.0),
        "v_ej": (0.3, 3.0),
        "M_Ni": (0.01, 0.5),
        "T_floor": (3000.0, 8000.0),
    },
    fixed={
        "kappa": 0.06,
    },
)
```

#### Save and reload the result

```python
path = tf.save(res, path="mcmc_out/fit_demo.npz")
loaded = tf.load(path)
```

#### Models, parameters, priors, and samplers

Recommended public model keys:

| Model key | Physical meaning | Parameter order (`theta`) |
|---|---|---|
| `nickel` | Ni-powered model | `(M_ej, v_ej, M_Ni, x_Ni, kappa, kappa_gamma, T_floor)` |
| `sc_ni` | Shock-cooling + Ni | `(M_ej, v_ej, E_Th_in, M_Ni, R_0, x_Ni, kappa, kappa_gamma, T_floor)` |
| `magnetar` | Pure magnetar | `(M_ej, v_ej, P_ms, B14, kappa, kappa_gamma, T_floor)` |
| `magnetar_ni` | Magnetar + Ni | `(M_ej, v_ej, P_ms, B14, M_Ni, kappa, kappa_gamma, T_floor)` |
| `sc_magnetar` | Shock-cooling + magnetar | `(M_ej, v_ej, E_Th_in, P_ms, B14, R_0, kappa, kappa_gamma, T_floor)` |

Compatibility aliases are still accepted internally, but the names above are the recommended public API.

Parameter glossary:

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
- likelihood is evaluated as `model(t_obs + t_shift)`
- positive `t_shift` shifts the model curve to earlier observed times
- if you do not want to fit `t_shift`, fix it with `fixed={"t_shift": 0.0}`

Recommended prior style:

```python
priors = {
    "M_ej": (1.0, 8.0),
    "v_ej": (0.2, 3.0),
}
```

Advanced prior styles are also supported:
- log-uniform: `"M_Ni": ("log10", -3.0, -0.3)`
- dict style: `"kappa": {"bounds": (0.03, 0.3), "scale": "linear"}`

Supported samplers:
- `emcee`
- `zeus`
- `dynesty`

Typical `sampler_kwargs`:
- `emcee` or `zeus`: `nwalkers`, `nsteps`, `burnin`, `thin`, `seed`, `progress`
- `dynesty`: `nlive`, `sample`, `bound`, `dlogz`, `maxiter`, `maxcall`, `seed`, `progress`, `nsamples`

For most users, `emcee` is the simplest default choice.

More examples:
- `examples/tutorial.ipynb`
- `examples/data`

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
