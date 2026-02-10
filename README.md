# Transfit

Transfit is a Python package for supernova light-curve generation and fitting. It supports both bolometric and multi-band light-curve simulation and MCMC parameter inference. This README is adapted from `examples/Tutorial.ipynb`.

**Directory Layout**
- `transfit/`: core source code
- `examples/`: examples and tutorials
- `examples/data/`: example data (`SN1993j_lbol.txt`, `SN2007gr.csv`)
- `MCMC_out/`: example fitting outputs

**Environment and Dependencies**
- Python 3.x
- Main dependencies: `numpy`, `matplotlib`, `pandas`, `emcee`
- Optional dependencies: `corner` (corner plots), `joblib` (parallel sampling)

It is recommended to run examples from the repository root so the local `transfit` package can be imported directly.

## Quick Start

**Import dependencies**
```python
import numpy as np
import matplotlib.pyplot as plt
import transfit as tf
```

## 1. Generate a Bolometric Light Curve

```python
ctx = tf.Context(
    distance=tf.Distance(z=0.05)
)

# Parameter order:
# (M_ej, v_ej, E_Th_in, M_Ni, R_max_in, x_s, kappa0, kappa_gamma, T_floor)
theta = (5, 1.0, 1.0, 0.2, 100.0, 0.5, 0.2, 0.03, 4000.0)

bol = tf.lightcurve_bol(
    model="scni",
    theta=theta,
    ctx=ctx,
    Nx=100,
    Ny=1000,
    t_max_days=180.0,
)

plt.plot(bol.t_days, bol.Lbol)
plt.yscale("log")
plt.xlabel("Observer Time (days)")
plt.ylabel("Bolometric Luminosity (erg/s)")
plt.show()
```

## 2. Generate Multi-band Light Curves

```python
# band -> nu_eff (Hz)
filters = {"g": 6.2e14, "r": 4.8e14}

ctx = tf.Context(
    distance=tf.Distance(z=0.05, DL_cm=1.0e27),
    filters=filters,
    y_kind="mag",  # "mag" gives AB magnitude; switch to "flux" for Fnu
)

mb = tf.lightcurve_multiband(
    model="scNi",
    theta=theta,
    ctx=ctx,
    bands=["g", "r"],
    Nx=100,
    Ny=1000,
    t_max_days=180.0,
)

plt.figure()
plt.plot(mb.t_days, mb.y["g"], label="g")
plt.plot(mb.t_days, mb.y["r"], label="r")
plt.gca().invert_yaxis()  # lower magnitude means brighter
plt.xlabel("t (days)")
plt.ylabel("AB mag")
plt.legend()
plt.show()
```

## 3. Bolometric Fitting

```python
# Example data
data_bol = np.loadtxt("examples/data/SN1993j_lbol.txt")

t = data_bol[:, 0]
t = t - t.min()
Lbol = data_bol[:, 1]
Lbol_err = data_bol[:, 2]

data_bol_tf = tf.BolometricData(
    t_days=t,
    y=Lbol,
    yerr=Lbol_err,
)

ctx = tf.Context(
    distance=tf.Distance(z=0.001728),
    filters={"B": 6.8e14, "V": 5.5e14, "R": 4.7e14, "I": 3.9e14},
    y_kind="mag",
)

# Run fitting (example; tune parameters as needed)
# res_bol = tf.fit_bol(
#     data=data_bol_tf,
#     model="scni",
#     ctx=ctx,
#     priors={
#         "M_ej": (1, 5),
#         "v_ej": (0.3, 3.0),
#         "M_Ni": (0.01, 0.5),
#     },
#     fixed={"kappa0": 0.1, "T_floor": 2000},
#     sampler_kwargs=dict(
#         nwalkers=40, nsteps=5000, burnin=300, thin=10,
#         seed=123, progress=True,
#     ),
#     model_kwargs=dict(
#         Nx=100, Ny=1000, t_max_days=float(np.max(data_bol_tf.t_days) + 50),
#     ),
# )
```

## 4. Multi-band Fitting

**Load observations and reshape to long format**
```python
import pandas as pd

csv_path = "examples/data/SN2007gr.csv"
df = pd.read_csv(csv_path)

# Use JD as the time axis
t0 = float(np.nanmin(df["JD"].to_numpy(float)))
df["t_days"] = df["JD"].to_numpy(float) - t0

band_map = [
    ("B", "Bmag", "e_Bmag"),
    ("V", "Vmag", "e_Vmag"),
    ("R", "Rmag", "e_Rmag"),
    ("I", "Imag", "e_Imag"),
]

rows = []
for b, mcol, ecol in band_map:
    if mcol not in df.columns or ecol not in df.columns:
        continue
    m = pd.to_numeric(df[mcol], errors="coerce").to_numpy(float)
    e = pd.to_numeric(df[ecol], errors="coerce").to_numpy(float)
    ok = np.isfinite(df["t_days"].to_numpy(float)) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    if not np.any(ok):
        continue
    rows.append(
        pd.DataFrame(
            {
                "t_days": df.loc[ok, "t_days"].to_numpy(float),
                "band": np.full(np.sum(ok), b, dtype=object),
                "mag": m[ok],
                "emag": e[ok],
            }
        )
    )

lc = pd.concat(rows, ignore_index=True).sort_values("t_days").reset_index(drop=True)
lc = lc[lc["t_days"] < 100].reset_index(drop=True)

data = tf.MultiBandData(
    t_days=lc["t_days"].to_numpy(float),
    band=lc["band"].to_numpy(),
    y=lc["mag"].to_numpy(float),
    yerr=lc["emag"].to_numpy(float),
)
```

**Run fitting (example)**
```python
filters = {"B": 6.8e14, "V": 5.5e14, "R": 4.7e14, "I": 3.9e14}
ctx = tf.Context(distance=tf.Distance(z=0.001728), filters=filters, y_kind="mag")

# res = tf.fit_multiband(
#     data=data,
#     model="ni",
#     ctx=ctx,
#     priors={
#         "M_ej": (1, 5),
#         "v_ej": (0.3, 3.0),
#         "M_Ni": (0.01, 0.5),
#         "T_floor": (3000, 8000),
#     },
#     fixed={"kappa0": 0.1},
#     sampler_kwargs=dict(
#         nwalkers=40, nsteps=5000, burnin=300, thin=10,
#         seed=123, progress=True,
#     ),
#     model_kwargs=dict(
#         Nx=100, Ny=1000, t_max_days=float(np.max(data.t_days) + 50),
#     ),
# )
```

## 5. Save, Load, and Plot

```python
# Save
# path = tf.save(res_bol, path="MCMC_out/fit_scni_test.npz")

# Load
res_bol_loaded = tf.load("MCMC_out/fit_scni_test.npz")

# Corner plot
tf.plot.corner(res_bol_loaded)

# Bolometric fit plot
tf.plot.fit_bol(res_bol_loaded, data=data_bol_tf)

# Multi-band fit plots
# tf.plot.corner(res)
# tf.plot.fit_multiband(res, data=data)
```

## Notes
- `emcee` is the default MCMC sampler.
- If `corner` is missing, you will see an import warning; install it if needed.
- All paths in the examples are relative to the repository root.

## Examples and Tutorial
- Notebook: `examples/Tutorial.ipynb`
- Example data: `examples/data/`

## Citation
If you use this software in your research, please cite:
```bibtex
@ARTICLE{2025ApJ...992...20L,
       author = {{Liu}, Liang-Duan and {Zhang}, Yu-Hao and {Yu}, Yun-Wei and {Du}, Ze-Xin and {Li}, Jing-Yao and {Wu}, Guang-Lei and {Dai}, Zi-Gao},
        title = "{TransFit: An Efficient Framework for Transient Light-curve Fitting with Time-dependent Radiative Diffusion}",
      journal = {\\apj},
     keywords = {Supernovae, Radiative transfer, Core-collapse supernovae, Time domain astronomy, 1668, 1335, 304, 2109, High Energy Astrophysical Phenomena, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = oct,
       volume = {992},
       number = {1},
          eid = {20},
        pages = {20},
          doi = {10.3847/1538-4357/adfed6},
archivePrefix = {arXiv},
       eprint = {2505.13825},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...992...20L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Some code and tutorial content were generated by Codex.
