<p align="center">
  <img src="docs/TransFit_logo.png" width="300">
</p>

# TransFit

**TransFit** is a fast and physically motivated **light-curve fitting framework** for astronomical transients such as supernovae.  
It numerically solves the **time-dependent radiative diffusion equation** in expanding ejecta and performs **Bayesian parameter inference using MCMC sampling**, enabling efficient fitting of observed transient light curves.

Compared with traditional **semi-analytical models** (e.g., Arnett-like models), which rely on simplified and time-independent temperature structures, TransFit directly solves the diffusion equation. This allows the model to capture **time-dependent temperature evolution, non-uniform heating distributions, and the transition from shock-cooling to radioactive-powered emission**, while maintaining **computational speeds comparable to semi-analytical approaches**.

By combining **physical realism with efficient MCMC fitting**, TransFit provides a practical tool for rapid light-curve modeling and parameter inference in the era of large time-domain surveys.

---

## Advantages

- Physically motivated: directly solves the time-dependent diffusion equation instead of relying on static semi-analytical approximations.
- Fast for fitting: designed for efficient Bayesian inference with MCMC backends such as `emcee`, `zeus`, and `dynesty`.
- Built for common workflows: supports both bolometric light-curve fitting and multi-band light-curve fitting.
- Simple public API: most users only need `tf.BolometricData(...)` or `tf.MultiBandData(...)`, plus `tf.fit_bol(...)` or `tf.fit_multiband(...)`.
- Practical output tools: includes fit plotting, corner plotting, and save/load helpers for standard analysis workflows.

## How To Use

Run from the repository root so `import transfit` works directly.

Main dependencies:
- `numpy`
- `matplotlib`
- `pandas`
- `numba`
- `astropy`

Optional dependencies:
- `corner` for posterior corner plots
- `emcee`, `zeus`, `dynesty` for fitting backends

Optional sampler backends are imported lazily, so `import transfit` does not require all sampler packages to be installed.

### 1. Prepare your data

Choose the data container that matches your observations:

- `tf.BolometricData(t_days, y, yerr, mask=None)`
- `tf.MultiBandData(t_days, band, y, yerr, mask=None)`

Rules:
- all arrays must have matching lengths
- `yerr` must be finite and positive
- `mask` is optional and will be respected automatically during fitting

### 2. Fit a bolometric light curve

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
    ),
)
```

### 3. Fit a multi-band light curve

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

### 4. Read, plot, and save the result

`fit_bol(...)` and `fit_multiband(...)` return `FitResult`.

Most useful accessors:
- `res.best_fit`
- `res.best_params`
- `res.median_params`
- `res.best_log_prob`

Common follow-up operations:

```python
print(res.best_fit)

fig = tf.plot.fit_bol(res, data=data, show_1sigma=True)
# or
fig = tf.plot.fit_multiband(res, data=data, show_1sigma=True)

corner_fig = tf.plot.corner(res)

path = tf.save(res, path="mcmc_out/fit_demo.npz")
loaded = tf.load(path)
```

## Models

Recommended public model keys:

| Model key | Physical meaning | Parameter order (`theta`) |
|---|---|---|
| `nickel` | Ni-powered model | `(M_ej, v_ej, M_Ni, x_Ni, kappa, kappa_gamma, T_floor)` |
| `sc_ni` | Shock-cooling + Ni | `(M_ej, v_ej, E_Th_in, M_Ni, R_0, x_Ni, kappa, kappa_gamma, T_floor)` |
| `magnetar` | Pure magnetar | `(M_ej, v_ej, P_ms, B14, kappa, kappa_gamma, T_floor)` |
| `magnetar_ni` | Magnetar + Ni | `(M_ej, v_ej, P_ms, B14, M_Ni, kappa, kappa_gamma, T_floor)` |
| `sc_magnetar` | Shock-cooling + magnetar | `(M_ej, v_ej, E_Th_in, P_ms, B14, R_0, kappa, kappa_gamma, T_floor)` |

Compatibility aliases are still accepted internally, but the names above are the recommended public API.

Built-in fixed assumptions:
- `nickel`: `E_Th_in=0`, `R_0=10 R_sun`
- `magnetar`: `E_Th_in=0`, `R_0=1 R_sun`
- `magnetar_ni`: `E_Th_in=0`, `R_0=1 R_sun`

### Parameter Glossary

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

## Priors and Samplers

### Recommended prior style

Use simple bounds when possible:

```python
priors = {
    "M_ej": (1.0, 8.0),
    "v_ej": (0.2, 3.0),
}
```

Advanced prior styles are also supported:
- log-uniform: `"M_Ni": ("log10", -3.0, -0.3)`
- dict style: `"kappa": {"bounds": (0.03, 0.3), "scale": "linear"}`

### Supported samplers

- `emcee`
- `zeus`
- `dynesty`

Typical `sampler_kwargs`:
- `emcee` / `zeus`: `nwalkers`, `nsteps`, `burnin`, `thin`, `seed`, `progress`
- `dynesty`: `nlive`, `sample`, `bound`, `dlogz`, `maxiter`, `maxcall`, `seed`, `progress`, `nsamples`

For most users, `emcee` is the simplest default choice.

## Plotting

Standard plotting helpers:
- `tf.plot.fit_bol(...)`
- `tf.plot.fit_multiband(...)`
- `tf.plot.corner(...)`

Notes:
- fit plots default to no grid
- `show_1sigma=True` draws the posterior 16%-84% band
- corner labels include units and LaTeX formatting

## Advanced Usage

The package also provides forward-model helpers for users who want custom prediction or custom plotting workflows:
- `lightcurve_bol(...)`
- `lightcurve_multiband(...)`
- `predict_bol(...)`
- `predict_multiband(...)`

These helpers use internal forward-model context objects and are not required for standard fitting workflows.

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
