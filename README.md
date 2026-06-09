# TransFit

<p align="right">
  <strong>Language:</strong> English | <a href="docs/README_chinese.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/TransFit_logo.png" width="430" alt="TransFit logo">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-MCMC-C0392B">
  <img alt="Models" src="https://img.shields.io/badge/Models-Diffusion%20Powered-1F4E79">
  <img alt="Data" src="https://img.shields.io/badge/Data-Bolometric%20%7C%20Multi--band-2E8B57">
  <img alt="Acceleration" src="https://img.shields.io/badge/Acceleration-Numba-00A3E0">
</p>

<p align="center">
  <a href="#installation">Install</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#public-api">Public API</a> |
  <a href="examples/tutorial.ipynb">Tutorial Notebook</a> |
  <a href="examples/data">Example Data</a> |
  <a href="https://doi.org/10.3847/1538-4357/adfed6">Paper</a>
</p>

TransFit is a light-curve modeling and fitting framework for astronomical
transients such as supernovae. It provides a compact Python interface for
forward modeling, bolometric fitting, and multi-band photometric fitting with
Bayesian samplers.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Inspect Model Parameters](#inspect-model-parameters)
  - [Forward Light Curves](#forward-light-curves)
  - [Fit Data](#fit-data)
- [Public API](#public-api)
- [Validation](#validation)
- [Documentation](#documentation)
- [Contact](#contact)
- [Citation](#citation)

---

## Features

### Physical Light-curve Models

- Diffusion-powered transient light-curve models with bolometric outputs.
- Nickel, magnetar, and magnetar-plus-nickel model families through a unified
  parameter interface.

### Multi-band Photometry

- Multi-band light curves in flux or magnitude space.
- Filter mapping through stable band names such as `B`, `V`, `R`, and `I`.
- Extinction and photometric-system support.

### Bayesian Fitting

- Bolometric and multi-band fitting with the same result object.
- Default fitting with `emcee`, with optional `zeus` and `dynesty` backends.
- Compact result access through `res.best_params`, `res.best_params_raw`,
  `res.median_params`, and `res.best_fit`.

---

## Installation

Install TransFit with the default fitting backend (`emcee`):

```bash
python -m pip install transfit
```

For local development from a cloned repository:

```bash
git clone <your-repo-url>
cd TransFit
python -m pip install -e ".[plot,examples]"
```

Install all sampler backends (`emcee`, `zeus`, and `dynesty`):

```bash
python -m pip install "transfit[all-samplers]"
```

For a lightweight forward-model-only environment without installing `emcee`,
install the core numerical dependencies yourself and then install TransFit
without dependencies:

```bash
python -m pip install numpy numba astropy scipy
python -m pip install transfit --no-deps
```

From a cloned repository, the equivalent lightweight editable install is:

```bash
python -m pip install -e . --no-deps
```

This lightweight path is intended for forward light-curve generation only. The
default `fit_bol()` and `fit_multiband()` samplers require `emcee`.

---

## Quick Start

<details>
<summary><strong>Inspect Model Parameters</strong></summary>

```python
import transfit as tf

tf.model_param_names("nickel")
tf.param_template("nickel")
```

</details>

<details>
<summary><strong>Forward Light Curves</strong></summary>

Bolometric light curve:

```python
import matplotlib.pyplot as plt
import transfit as tf

params_nickel = {
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
    model="nickel",
    params=params_nickel,
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
  <img src="docs/lightcurve_bol.png" alt="Bolometric light curve example">
</p>

Multi-band light curve:

```python
import matplotlib.pyplot as plt
import transfit as tf

filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
    "R": "johnson_cousins.R",
    "I": "johnson_cousins.I",
}

params_nickel = {
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

mb = tf.lightcurve_multiband(
    model="nickel",
    params=params_nickel,
    z=0.001728,
    distance_modulus=29.84,
    filters=filters,
    bands=["B", "V", "R", "I"],
    y_kind="mag",
    mag_system="vega",
    extinction={"mw": {"ebv": 0.04, "rv": 3.1, "law": "odonnell94"}},
    t_max_days=180.0,
)

fig, ax = plt.subplots()
for band in mb.bands:
    ax.plot(mb.t_days, mb.y[band], label=band)
ax.invert_yaxis()
ax.set_xlabel("Observer time (days)")
ax.set_ylabel("Vega magnitude")
ax.legend()
```

<p align="center">
  <img src="docs/lightcurve_multiband.png" alt="Multi-band light curve example">
</p>

</details>

<details>
<summary><strong>Fit Data</strong></summary>

Prepare a data container:

```python
import numpy as np
import transfit as tf

arr = np.loadtxt("examples/data/sn1993j_lbol.txt")
t_days = arr[:, 0] - arr[:, 0].min()

data_bol = tf.BolometricData(
    t_days=t_days,
    y=arr[:, 1],
    yerr=arr[:, 2],
)
```

Run a bolometric fit:

```python
res_bol = tf.fit_bol(
    data=data_bol,
    model="nickel",
    z=0.001728,
    priors={
        "M_ej": (0.5, 8.0),
        "v_ej": (0.2, 3.0),
        "E_Th_in": (0.05, 8.0),
        "M_Ni": ("log10", -3.0, -0.2),
        "R_0": (10.0, 400.0),
    },
    fixed={
        "x_Ni": 0.2,
        "kappa": 0.12,
        "kappa_gamma": 0.03,
    },
    sampler="emcee",
    sampler_kwargs={
        "nwalkers": 32,
        "nsteps": 600,
        "burnin": 200,
        "thin": 5,
        "seed": 123,
        "progress": True,
    },
)

print(res_bol.best_params_raw)
print(res_bol.best_fit)
```

Plot and save results:

```python
fig = tf.plot.fit_bol(res_bol, data=data_bol, show_1sigma=True)

path = tf.save(res_bol, path="mcmc_out/fit_nickel_bol_demo.npz")
loaded = tf.load(path)
print(path)
print(loaded["samples"].shape)
```

</details>

---

## Public API

The main public entry points are:

```python
tf.BolometricData(t_days, y, yerr, mask=None)
tf.MultiBandData(t_days, band, y, yerr, mask=None)

tf.model_param_names(model)
tf.param_template(model)

tf.lightcurve_bol(model=..., params=..., z=..., t_max_days=...)
tf.lightcurve_multiband(model=..., params=..., z=..., distance_modulus=..., filters=..., bands=...)

tf.fit_bol(data=..., model=..., z=..., priors=..., fixed=...)
tf.fit_multiband(data=..., model=..., z=..., distance_modulus=..., filters=..., priors=..., fixed=...)

tf.save(res, path=None)
tf.load(path, trusted=False)
```

Advanced interpolation helpers are also available when you need model values at
specific observation times:

```python
tf.predict_bol(...)
tf.predict_multiband(...)
```

`fit_bol()` and `fit_multiband()` return a `FitResult`. The main result
properties are:

```python
res.best_params       # rounded best-fit parameter dict
res.best_params_raw   # full-precision best-fit parameter dict
res.median_params     # posterior median parameter dict
res.best_fit          # compact report with params, errors, log_prob, sample
res.best_index        # index of the best posterior sample
res.best_log_prob     # best log posterior value
res.best_sample       # raw best posterior sample vector
res.samples           # flattened posterior samples
res.log_prob          # log posterior values for samples
res.meta              # priors, bounds, sampler metadata, model settings
```

## Validation

Run the test suite with:

```bash
python -m pytest -q
```

The current tests cover the public API, fitting workflow, multi-band
photometry, extinction handling, and result-object accessors.

---

## Documentation

- [Tutorial notebook](examples/tutorial.ipynb)
- [Example data](examples/data)
- [Multi-band photometry design](docs/multiband_photometry_design.md)
- [Model parameter reference](docs/model_parameter_reference.tex)
- [Model citation guide](docs/model_citations.md)
- [Physical regression and convergence tests](examples/physical_regression_and_convergence_tests.ipynb)

---

## Contact

For questions about this project, please contact:

- Liangduan Liu ([liuld@ccnu.edu.cn](mailto:liuld@ccnu.edu.cn))
- Yuhao Zhang ([zhangyh2001@foxmail.com](mailto:zhangyh2001@foxmail.com))
- GuangLei Wu ([wuguanglei@mails.ccnu.edu.cn](mailto:wuguanglei@mails.ccnu.edu.cn))

---

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

For model-specific citation details, see the
[model citation guide](docs/model_citations.md). Cite TransFit for all models,
and additionally cite TransFit-CSM when using `csm`.

Some code and tutorial content were generated with assistance from Codex.
