# Transfit V0.4

Transfit 是一个用于超新星光变曲线生成与拟合的 Python 工具包，支持热光变曲线与多波段光变曲线的模拟与参数拟合（MCMC）。本 README 依据 `examples/Tutorial.ipynb` 整理。

**目录结构**
- `transfit/`: 核心源码
- `examples/`: 示例与教程
- `examples/data/`: 示例数据（`SN1993j_lbol.txt`, `SN2007gr.csv`）
- `MCMC_out/`: 拟合输出示例

**环境与依赖**
- Python 3.x
- 主要依赖: `numpy`, `matplotlib`, `pandas`, `emcee`
- 可选依赖: `corner`（用于角图）、`joblib`（并行采样）

建议在仓库根目录运行示例代码，以确保本地 `transfit` 包可被直接导入。

## 快速开始

**导入依赖**
```python
import numpy as np
import matplotlib.pyplot as plt
import transfit as tf
```

## 1. 生成热光变曲线（Bolometric）

```python
ctx = tf.Context(
    distance=tf.Distance(z=0.05)
)

# 模型参数顺序：
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

## 2. 生成多波段光变曲线

```python
# band -> nu_eff (Hz)
filters = {"g": 6.2e14, "r": 4.8e14}

ctx = tf.Context(
    distance=tf.Distance(z=0.05, DL_cm=1.0e27),
    filters=filters,
    y_kind="mag",  # "mag" -> AB magnitude；改成 "flux" 可输出 Fnu
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
plt.gca().invert_yaxis()  # 星等越小越亮
plt.xlabel("t (days)")
plt.ylabel("AB mag")
plt.legend()
plt.show()
```

## 3. 热光变拟合（Bolometric Fit）

```python
# 示例数据
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

# 运行拟合（示例，参数可按需调整）
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

## 4. 多波段拟合（Multi-band Fit）

**读取观测数据并整理为长表**
```python
import pandas as pd

csv_path = "examples/data/SN2007gr.csv"
df = pd.read_csv(csv_path)

# 用 JD 做时间
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

**执行拟合（示例）**
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

## 5. 保存、加载与绘图

```python
# 保存
# path = tf.save(res_bol, path="MCMC_out/fit_scni_test.npz")

# 加载
res_bol_loaded = tf.load("MCMC_out/fit_scni_test.npz")

# 角图
tf.plot.corner(res_bol_loaded)

# 拟合结果
tf.plot.fit_bol(res_bol_loaded, data=data_bol_tf)

# 多波段拟合结果
# tf.plot.corner(res)
# tf.plot.fit_multiband(res, data=data)
```

## 备注
- `emcee` 是当前默认的 MCMC 采样器。
- `corner` 未安装时会提示缺失，可按需安装。
- 示例路径均相对仓库根目录。

## 示例与教程
- Notebook: `examples/Tutorial.ipynb`
- 示例数据: `examples/data/`
