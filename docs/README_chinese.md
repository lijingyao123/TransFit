# TransFit

<p align="right">
  <strong>语言：</strong><a href="../README.md">English</a> | 简体中文
</p>

<p align="center">
  <img src="TransFit_logo.png" width="430" alt="TransFit logo">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue">
  <img alt="Inference" src="https://img.shields.io/badge/Inference-MCMC-C0392B">
  <img alt="Models" src="https://img.shields.io/badge/Models-Transient%20Light%20Curves-1F4E79">
  <img alt="Data" src="https://img.shields.io/badge/Data-Bolometric%20%7C%20Multi--band-2E8B57">
</p>

TransFit 是一个用于暂现源光变曲线正向建模和拟合的 Python 软件包。它提供简洁的
bolometric 和多波段数据接口，并内置 nickel、magnetar、magnetar-plus-nickel
以及 CSM 相互作用模型。

## 主要功能

- 输出 bolometric 光度、有效温度和光球半径的物理光变模型。
- 支持流量或星等空间的多波段测光，包括滤光片、消光和 SED 设置。
- 使用统一结果对象进行贝叶斯拟合，默认安装 `emcee`，也可选装 `zeus` 和
  `dynesty`。

## 安装

```bash
python -m pip install transfit
```

本地开发安装：

```bash
git clone <your-repo-url>
cd TransFit
python -m pip install -e ".[plot,examples]"
```

安装可选采样器后端：

```bash
python -m pip install "transfit[all-samplers]"
```

## 快速开始

<details>
<summary><strong>正向计算 bolometric 光变曲线</strong></summary>

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

<p align="center">
  <img src="lightcurve_bol.png" alt="Bolometric forward model example">
</p>

</details>

<details>
<summary><strong>正向计算多波段光变曲线</strong></summary>

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

filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
    "R": "johnson_cousins.R",
    "I": "johnson_cousins.I",
}

lc = tf.lightcurve_multiband(
    model="nickel",
    params=params,
    z=0.001728,
    filters=filters,
    bands=["B", "V", "R", "I"],
    y_kind="mag",
    mag_system="vega",
    t_max_days=120.0,
)

for band in lc.bands:
    plt.plot(lc.t_days, lc.y[band], label=band)
plt.gca().invert_yaxis()
plt.xlabel("Observer-frame time (days)")
plt.ylabel("Vega magnitude")
plt.legend()
plt.show()
```

<p align="center">
  <img src="lightcurve_multiband.png" alt="Multi-band forward model example">
</p>

</details>

<details>
<summary><strong>拟合 bolometric 光变曲线</strong></summary>

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
        "t_shift": (0.0, 20.0),
    },
    fixed={
        "x_Ni": 0.2,
        "kappa": 0.12,
        "kappa_gamma": 0.03,
    },
    sampler_kwargs={"nwalkers": 32, "nsteps": 5000, "burnin": 1000, "thin": 10},
)

print(res.best_params_raw)
tf.save(res, "mcmc_out/sn1993j_bol_nickel.npz")
```

</details>

<details>
<summary><strong>拟合多波段光变曲线</strong></summary>

```python
import numpy as np
import transfit as tf

raw = np.genfromtxt(
    "examples/data/sn2007gr.csv",
    delimiter=",",
    names=True,
    dtype=float,
    encoding="utf-8",
)

bands, t_days, y, yerr = [], [], [], []
t0 = np.nanmin(raw["Phase"])
columns = {
    "B": ("Bmag", "e_Bmag"),
    "V": ("Vmag", "e_Vmag"),
    "R": ("Rmag", "e_Rmag"),
    "I": ("Imag", "e_Imag"),
}

for band, (mag_col, err_col) in columns.items():
    good = (
        np.isfinite(raw["Phase"])
        & np.isfinite(raw[mag_col])
        & np.isfinite(raw[err_col])
        & (raw[err_col] > 0)
    )
    t_days.extend((raw["Phase"][good] - t0).tolist())
    y.extend(raw[mag_col][good].tolist())
    yerr.extend(raw[err_col][good].tolist())
    bands.extend([band] * int(np.sum(good)))

data = tf.MultiBandData(
    t_days=np.asarray(t_days, float),
    band=np.asarray(bands, dtype=object),
    y=np.asarray(y, float),
    yerr=np.asarray(yerr, float),
)

filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
    "R": "johnson_cousins.R",
    "I": "johnson_cousins.I",
}

res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.001728,
    filters=filters,
    y_kind="mag",
    mag_system="vega",
    priors={
        "M_ej": (0.5, 8.0),
        "v_ej": (0.2, 3.0),
        "E_Th_in": (0.05, 8.0),
        "M_Ni": ("log10", -3.0, -0.2),
        "R_0": (10.0, 400.0),
        "t_shift": (0.0, 20.0),
    },
    fixed={
        "x_Ni": 0.2,
        "kappa": 0.12,
        "kappa_gamma": 0.03,
        "T_floor": 4500.0,
    },
    sampler_kwargs={"nwalkers": 32, "nsteps": 5000, "burnin": 1000, "thin": 10},
)

print(res.best_params_raw)
tf.save(res, "mcmc_out/sn2007gr_multiband_nickel.npz")
```

</details>

## 公开 API

<details>
<summary><strong>数据容器</strong></summary>

```python
tf.BolometricData(t_days, y, yerr, mask=None)
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

</details>

<details>
<summary><strong>模型参数查看</strong></summary>

```python
tf.model_param_names("nickel")
tf.param_template("csm")
```

规范模型名包括 `nickel`、`magnetar`、`magnetar_ni` 和 `csm`。

</details>

<details>
<summary><strong>正向计算和插值预测</strong></summary>

```python
tf.lightcurve_bol(model=..., params=..., z=..., t_max_days=...)
tf.lightcurve_multiband(
    model=...,
    params=...,
    z=...,
    filters=...,
    bands=...,
    y_kind="mag",
)

tf.predict_bol(model=..., params=..., z=..., t_days=...)
tf.predict_multiband(
    model=...,
    params=...,
    z=...,
    filters=...,
    t_days=...,
    band=...,
)
```

</details>

<details>
<summary><strong>拟合</strong></summary>

```python
tf.fit_bol(
    data=...,
    model=...,
    z=...,
    priors=...,
    fixed=...,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)

tf.fit_multiband(
    data=...,
    model=...,
    z=...,
    filters=...,
    y_kind="mag",
    priors=...,
    fixed=...,
    sed=None,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)
```

</details>

<details>
<summary><strong>结果、画图和读写</strong></summary>

```python
res.best_params
res.best_params_raw
res.median_params
res.best_fit
res.best_index
res.best_log_prob
res.best_sample
res.samples
res.log_prob
res.meta

tf.plot.fit_bol(res, data=data)
tf.plot.fit_multiband(res, data=data)
tf.plot.corner(res)

path = tf.save(res, path="mcmc_out/result.npz")
loaded = tf.load(path)
```

</details>

完整说明见 [API 和参数参考](api_reference.md)。

## 文档

- [教程 notebook](../examples/tutorial.ipynb)
- [API 和参数参考](api_reference.md)
- [English API reference](api_reference.md)
- [模型引用指南](model_citations.md)

## 引用

如果你在科研工作中使用 TransFit，请引用 TransFit 软件论文：

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

不同模型的具体引用规则请见[模型引用指南](model_citations.md)；使用 `csm`
模型时还需要引用 TransFit-CSM 论文。
