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

正向计算 bolometric 光变曲线：

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

拟合 bolometric 光变曲线：

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

## 文档

- [教程 notebook](../examples/tutorial.ipynb)
- [API 和参数参考](api_reference.tex)
- [模型引用指南](model_citations.md)
- [英文 README](../README.md)

## 测试

```bash
python -m pytest -q
```

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
