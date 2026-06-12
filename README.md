# TransFit

<p align="right">
  <strong>Language:</strong> English | <a href="docs/README_chinese.md">Simplified Chinese</a>
</p>

<p align="center">
  <img src="docs/TransFit_logo.png" width="430" alt="TransFit logo">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-MCMC-C0392B">
  <img alt="Models" src="https://img.shields.io/badge/Models-Transient%20Light%20Curves-1F4E79">
  <img alt="Data" src="https://img.shields.io/badge/Data-Bolometric%20%7C%20Multi--band-2E8B57">
</p>

TransFit is a Python package for forward modeling and fitting astronomical
transient light curves. It provides a compact interface for bolometric and
multi-band data, with built-in nickel, magnetar, magnetar-plus-nickel, and CSM
interaction models.

## Features

- Physical light-curve models with bolometric luminosity, effective
  temperature, and photospheric radius outputs.
- Multi-band photometry in flux or magnitude space, including filter,
  extinction, and SED handling.
- Bayesian fitting through a consistent result object, with `emcee` installed
  by default and optional `zeus` and `dynesty` backends.

## Installation

```bash
python -m pip install transfit
```

For local development:

```bash
git clone <your-repo-url>
cd TransFit
python -m pip install -e ".[plot,examples]"
```

Install optional sampler backends with:

```bash
python -m pip install "transfit[all-samplers]"
```

## Quick Start

Forward bolometric light curve:

```python
import matplotlib.pyplot as plt
import transfit as tf

params = {
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

lc = tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    t_max_days=120.0,
)

plt.plot(lc.t_days, lc.Lbol)
plt.yscale("log")
plt.xlabel("Observer-frame time (days)")
plt.ylabel("Bolometric luminosity (erg s$^{-1}$)")
plt.show()
```

Fit a bolometric light curve:

```python
import numpy as np
import transfit as tf

arr = np.loadtxt("examples/data/sn1993j_lbol.txt")
data = tf.BolometricData(
    t_days=arr[:, 0] - arr[:, 0].min(),
    y=arr[:, 1],
    yerr=arr[:, 2],
)

res = tf.fit_bol(
    data=data,
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
    sampler_kwargs={"nwalkers": 32, "nsteps": 600, "burnin": 200, "seed": 123},
)

print(res.best_params_raw)
tf.save(res, "mcmc_out/nickel_bol_demo.npz")
```

## Documentation

- [Tutorial notebook](examples/tutorial.ipynb)
- [API and parameter reference](docs/api_reference.md)
- [Model citation guide](docs/model_citations.md)
- [Chinese README](docs/README_chinese.md)

## Tests

```bash
python -m pytest -q
```

## Citation

If you use TransFit in research, cite the TransFit software paper:

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

For model-specific citation rules, especially when using `csm`, see the
[model citation guide](docs/model_citations.md).
