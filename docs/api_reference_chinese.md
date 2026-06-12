# TransFit API 和参数参考

本文档说明 TransFit 稳定公开 Python 接口。README 和 tutorial 只展示最小用法；
API 参数含义、模型参数、结果字段和高级选项统一放在这里。

所有公开时间输入和输出都使用 **observer-frame days**。内部物理模型仍在
rest-frame time 中求解，并在 API 边界转换回 observer frame。

## 公开入口

| 类别 | 入口 |
|---|---|
| 数据容器 | `BolometricData`, `MultiBandData` |
| 模型查看 | `model_param_names(model)`, `param_template(model)` |
| 正向光变曲线 | `lightcurve_bol(...)`, `lightcurve_multiband(...)` |
| 插值预测 | `predict_bol(...)`, `predict_multiband(...)` |
| 拟合 | `fit_bol(...)`, `fit_multiband(...)` |
| 结果读写 | `save(res, path=None)`, `load(path, trusted=False)` |
| 画图 | `transfit.plot.fit_bol`, `transfit.plot.fit_multiband`, `transfit.plot.corner` |

## 模型名和参数

规范模型名包括 `nickel`、`magnetar`、`magnetar_ni` 和 `csm`。为了兼容旧脚本，
部分别名仍可使用，但新代码建议使用规范模型名。

### `nickel`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `v_ej` | 抛射物速度，10^9 cm s^-1 |
| `E_Th_in` | 初始热能，10^49 erg |
| `M_Ni` | 镍质量，M☉ |
| `R_0` | 初始半径，R☉ |
| `x_Ni` | 镍混合位置，无量纲 |
| `kappa` | optical opacity，cm^2 g^-1 |
| `kappa_gamma` | gamma-ray opacity，cm^2 g^-1 |
| `T_floor` | 温度下限，K |

### `magnetar`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `v_ej` | 抛射物速度，10^9 cm s^-1 |
| `E_Th_in` | 初始热能，10^49 erg |
| `P_ms` | 磁星自转周期，ms |
| `B14` | 磁星偶极磁场，10^14 G |
| `R_0` | 初始半径，R☉ |
| `kappa` | optical opacity，cm^2 g^-1 |
| `kappa_gamma` | gamma-ray opacity，cm^2 g^-1 |
| `T_floor` | 温度下限，K |

### `magnetar_ni`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `v_ej` | 抛射物速度，10^9 cm s^-1 |
| `P_ms` | 磁星自转周期，ms |
| `B14` | 磁星偶极磁场，10^14 G |
| `M_Ni` | 镍质量，M☉ |
| `kappa` | optical opacity，cm^2 g^-1 |
| `kappa_gamma` | gamma-ray opacity，cm^2 g^-1 |
| `T_floor` | 温度下限，K |

### `csm`

| 参数 | 含义和单位 |
|---|---|
| `M_ej` | 抛射物质量，M☉ |
| `E_sn` | 爆炸能量，10^51 erg |
| `M_csm` | CSM 质量，M☉ |
| `R_csm_out` | CSM 外半径，R☉ |
| `kappa` | optical opacity，cm^2 g^-1 |
| `s` | CSM 密度幂律指数 |
| `eps_sh` | shock 辐射效率 |
| `T_floor` | 温度下限，K |

拟合时可加入可选参数 `t_shift`。当 `include_t_shift=True` 或运行拟合接口时，
它会被加入参数列表，并被限制为非负。拟合中模型在以下时间点计算：

```text
t_eval = t_obs + t_shift
```

因此 `t_shift > 0` 表示模型起点早于用户数据的观测零点。

## 数据容器

### `BolometricData`

```python
tf.BolometricData(t_days, y, yerr, mask=None)
```

| 字段 | 含义 |
|---|---|
| `t_days` | observer-frame days |
| `y` | bolometric luminosity，erg s^-1 |
| `yerr` | 与 `y` 同单位的一倍标准差误差 |
| `mask` | 可选布尔 mask；只有 mask 选中的点会进入拟合 |

未被 mask 排除的 luminosity 和误差必须为正且有限。

### `MultiBandData`

```python
tf.MultiBandData(t_days, band, y, yerr, mask=None)
```

| 字段 | 含义 |
|---|---|
| `t_days` | observer-frame days |
| `band` | 每个数据点对应的 band 标签 |
| `y` | 若 `y_kind="mag"` 则为星等；若 `y_kind="flux"` 则为 flux density |
| `yerr` | 与 `y` 同单位的一倍标准差误差 |
| `mask` | 可选布尔 mask；只有 mask 选中的点会进入拟合 |

## 正向计算和预测

Bolometric 正向计算：

```python
tf.lightcurve_bol(
    model="nickel",
    params=params,
    z=0.001728,
    t_max_days=150.0,
    solver_kwargs=None,
)
```

返回 `BolometricLC`，包含：

- `t_days`
- `Lbol`
- `Teff`
- `Rph`

多波段正向计算：

```python
tf.lightcurve_multiband(
    model="nickel",
    params=params,
    z=0.001728,
    distance_modulus=None,
    filters=filters,
    bands=["B", "V"],
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    sed=None,
    t_max_days=150.0,
    solver_kwargs=None,
)
```

返回 `MultiBandLC`，包含：

- `t_days`
- `bands`
- `y[band]`

`predict_bol` 和 `predict_multiband` 用于在用户给定的 observer-frame
时间点上计算模型值。`interp_fill` 可取 `"nan"`、`"raise"` 或 `"edge"`。
拟合接口中禁止 `"edge"`，避免在模型时间范围外静默使用边界值。

## 拟合接口

Bolometric 拟合：

```python
res = tf.fit_bol(
    data=data,
    model="nickel",
    z=0.001728,
    priors=priors,
    fixed=fixed,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)
```

多波段拟合：

```python
res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.001728,
    distance_modulus=None,
    filters=filters,
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    priors=priors,
    fixed=fixed,
    sampler="emcee",
    sed=None,
    sampler_kwargs=None,
    model_kwargs=None,
)
```

`priors` 是参数名到先验范围的映射。线性均匀先验写作 `(lo, hi)`。
以 10 为底的 log-uniform 先验写作 `("log10", lo, hi)`，其中 `lo` 和
`hi` 是 log10 空间中的边界。

`fixed` 是参数名到固定值的映射。没有放在 `fixed` 里的模型参数会被采样，
其范围来自默认边界或 `priors` 中用户给出的边界。

### `sigma_int`

`sigma_int` 是 likelihood nuisance parameter，不是物理模型参数。它可以通过
`fixed` 固定，也可以通过 `priors` 采样。

| 观测空间 | 含义 |
|---|---|
| `y_kind="mag"` | 额外星等 scatter |
| `y_kind="flux"` | 用 0.4 ln(10) sigma_int 转换成 fractional flux scatter |

## 关键字参数字典

### `sampler_kwargs`

`emcee` 和 `zeus` 常用键：

| 键 | 含义 |
|---|---|
| `nwalkers` | walker 数量 |
| `nsteps` | production chain 步数 |
| `burnin` | production 前的 burn-in 步数 |
| `thin` | thinning 因子 |
| `seed` | 随机种子 |
| `init` | 初始位置模式或数组 |
| `pool` | 用户传入的并行 pool |
| `progress` | 是否显示采样进度 |

`dynesty` 常用键：

| 键 | 含义 |
|---|---|
| `nlive` | live point 数量 |
| `sample` | dynesty sampling method |
| `bound` | dynesty bounding method |
| `dlogz` | 停止阈值 |
| `maxiter` | 最大迭代数 |
| `maxcall` | 最大 likelihood 调用数 |
| `seed` | 随机种子 |
| `progress` | 是否显示采样进度 |
| `nsamples` | 返回的 posterior sample 数 |
| `add_live` | 是否把 live points 加入 posterior |
| `pool` | 用户传入的并行 pool |
| `queue_size` | dynesty queue size |

### `model_kwargs`

拟合时传给模型计算的选项放在 `model_kwargs` 中。

| 键 | 含义 |
|---|---|
| `t_max_days` | observer-frame 模型计算时长，单位 days |
| `interp_fill` | 插值边界策略；拟合中不允许 `"edge"` |
| `solver_kwargs` | 高级数值网格选项 |

如果省略 `t_max_days`，TransFit 会自动选一个足够覆盖数据和 `t_shift`
允许范围的值。

### `solver_kwargs`

`solver_kwargs` 是高级数值网格接口。

| 键 | 默认值 | 含义 |
|---|---:|---|
| `Nx` | `100` | 空间/网格分辨率参数 |
| `Ny` | `1000` | 时间/网格分辨率参数 |

二者都必须是正整数。初级示例刻意不暴露这些数值控制参数。

## SED 选项

默认多波段 SED 是 `BlackbodySED`。

```python
from transfit.modules.sed import BlackbodySED, CutoffBlackbodySED

sed = BlackbodySED()
sed = CutoffBlackbodySED(
    cutoff_wavelength_A=3000.0,
    uv_slope=2.0,
    min_factor=0.0,
)
```

`CutoffBlackbodySED` 会对短波端施加 cutoff：

```text
L_nu_cutoff = C(lambda_rest) * L_nu_blackbody
```

其中：

```text
C(lambda) = 1                                      for lambda >= lambda_cut
C(lambda) = max(f_min, (lambda/lambda_cut)^a)      for lambda < lambda_cut
```

| 符号 | API 参数 |
|---|---|
| `lambda_cut` | `cutoff_wavelength_A` |
| `a` | `uv_slope` |
| `f_min` | `min_factor` |

设置 `min_factor=0` 时就是纯 power-law cutoff。

## FitResult 字段

`fit_bol` 和 `fit_multiband` 返回 `FitResult`。

| 字段/属性 | 含义 |
|---|---|
| `res.best_params` | 四舍五入后的 best-fit 参数字典 |
| `res.best_params_raw` | 全精度 best-fit 参数字典 |
| `res.median_params` | posterior median 参数字典 |
| `res.best_fit` | 包含参数、误差、best log probability 和 best sample 的紧凑记录 |
| `res.best_index` | best posterior sample 的索引 |
| `res.best_log_prob` | best log posterior 值 |
| `res.best_sample` | `res.param_names` 顺序下的原始 best sample 向量 |
| `res.samples` | 展平后的 posterior samples |
| `res.log_prob` | 每个 sample 的 log posterior |
| `res.meta` | sampler、prior、model、SED 和 context 元数据 |

## 引用规则

所有模型都应引用 TransFit 软件论文。使用 `csm` 模型时，还应额外引用
TransFit-CSM 论文。BibTeX 和模型引用规则见
[model_citations.md](model_citations.md)。
