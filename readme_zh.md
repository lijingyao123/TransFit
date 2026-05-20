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
  <a href="examples/tutorial_zh.ipynb">中文教程 Notebook</a> |
  <a href="examples/tutorial.ipynb">English Tutorial</a> |
  <a href="examples/data">示例数据</a> |
  <a href="#使用方式">使用方式</a> |
  <a href="https://doi.org/10.3847/1538-4357/adfed6">论文</a>
</p>

<p><strong>TransFit</strong> 是一个面向超新星等瞬变天体的光变曲线拟合框架。它既支持理论光变计算，也支持 MCMC 贝叶斯拟合；在多波段路径里，内部统一先计算 observer-frame <code>f_nu</code>，再施加消光并转换成流量或星等。</p>

</td>
</tr>
</table>

## 简介

TransFit 主要面向两类任务：
- 画理论测光光变或多波段光变
- 用 MCMC 拟合观测数据并查看后验结果

最常用的公开接口是：
- `tf.lightcurve_bol(...)`
- `tf.lightcurve_multiband(...)`
- `tf.fit_bol(...)`
- `tf.fit_multiband(...)`
- `tf.save(...)`
- `tf.load(...)`

## 安装方式

先克隆仓库，然后用 editable 模式安装：

```bash
git clone <your-repo-url>
cd TransFit
python -m pip install -e .
```

如果你想运行 notebook 或绘图示例，安装示例依赖：

```bash
python -m pip install -e .[examples]
```

如果你还想安装拟合后端和角图依赖，可以直接安装完整可选依赖：

```bash
python -m pip install -e .[all]
```

## 使用方式

如果你想先查看某个模型需要哪些参数，可以直接运行：

```python
import transfit as tf

tf.model_param_names("nickel")
tf.param_template("nickel")
```

### 1. 如何画一个光变曲线

测光光变示例：

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

多波段光变示例：

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
for b in mb.bands:
    ax.plot(mb.t_days, mb.y[b], label=b)
ax.invert_yaxis()
ax.set_xlabel("Observer time (days)")
ax.set_ylabel("Vega magnitude")
ax.legend()
```

### 2. 如何拟合

先准备对应的数据容器：

- `tf.BolometricData(t_days, y, yerr, mask=None)`
- `tf.MultiBandData(t_days, band, y, yerr, mask=None)`

测光拟合示例：

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
```

多波段拟合示例：

```python
import numpy as np
import pandas as pd
import transfit as tf

df = pd.read_csv("examples/data/sn2007gr.csv")
t0 = float(np.nanmin(df["JD"].to_numpy(float)))
df["t_days"] = df["JD"].to_numpy(float) - t0

rows = []
for band_name, ycol, ecol in [
    ("B", "Bmag", "e_Bmag"),
    ("V", "Vmag", "e_Vmag"),
    ("R", "Rmag", "e_Rmag"),
    ("I", "Imag", "e_Imag"),
]:
    m = df[ycol].notna() & df[ecol].notna()
    rows.append(
        pd.DataFrame(
            {
                "t_days": df.loc[m, "t_days"].to_numpy(float),
                "band": band_name,
                "y": df.loc[m, ycol].to_numpy(float),
                "yerr": df.loc[m, ecol].to_numpy(float),
            }
        )
    )

long_df = pd.concat(rows, ignore_index=True)

data_mb = tf.MultiBandData(
    t_days=long_df["t_days"].to_numpy(float),
    band=long_df["band"].to_numpy(str),
    y=long_df["y"].to_numpy(float),
    yerr=long_df["yerr"].to_numpy(float),
)

filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
    "R": "johnson_cousins.R",
    "I": "johnson_cousins.I",
}

res_mb = tf.fit_multiband(
    data=data_mb,
    model="nickel",
    z=0.001728,
    distance_modulus=29.84,
    filters=filters,
    y_kind="mag",
    mag_system="vega",
    extinction={"mw": {"ebv": 0.04, "rv": 3.1, "law": "odonnell94"}},
    priors={
        "M_ej": (1.0, 5.0),
        "v_ej": (0.3, 3.0),
        "E_Th_in": (0.05, 8.0),
        "M_Ni": (0.01, 0.5),
        "R_0": (10.0, 400.0),
        "T_floor": (3000.0, 8000.0),
    },
    fixed={"kappa": 0.06},
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
```

拟合完成后，可以直接画图并保存链：

```python
fig_bol = tf.plot.fit_bol(res_bol, data=data_bol, show_1sigma=True)
fig_mb = tf.plot.fit_multiband(res_mb, data=data_mb, show_1sigma=True)
corner_fig = tf.plot.corner(res_mb)

path = tf.save(res_mb, path="mcmc_out/fit_nickel_multiband_demo.npz")
loaded = tf.load(path)
print(path)
print(loaded["samples"].shape)
```

## 联系方式

如有问题，可通过邮件联系：

- Liangduan Liu ([liuld@ccnu.edu.cn])
- Yuhao Zhang ([zhangyh2001@foxmail.com])

## 引用

如果你在科研工作中使用了本软件，请引用：

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

部分代码与教程内容由 Codex 协助生成。
