# transfit/modules/plot.py
# -*- coding: utf-8 -*-
"""
One-click plotting helpers for transfit.

Public:
  - corner(res, ...)
  - fit_multiband(res, data, ...)
  - fit_bol(res, data, ...)

Accepted `res`:
  1) FitResult (from tf.fit_*)
  2) loaded dict (from your io.load(...))

Notes:
  - This module DOES NOT read file paths.
  - It NEVER calls plt.show() / display().
    In notebooks: returning a Figure will be displayed once automatically.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# style (good journal-like defaults, only inside rc_context)
# -----------------------------------------------------------------------------

def _journal_rc() -> Dict[str, Any]:
    return {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    }


def _maybe_invert_mag_axis(ax, y_kind: str):
    if str(y_kind).lower() == "mag":
        ax.invert_yaxis()


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _unique_in_order(seq: Sequence[Any]) -> List[Any]:
    """Unique values preserving first-seen order (case-sensitive)."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _to_loaded(res: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Accept FitResult or loaded dict. NO PATH support.
    loaded dict recommended keys:
      samples, log_prob, param_names, all_param_names, fixed, meta, ctx, model
    """
    if isinstance(res, dict):
        return res

    if hasattr(res, "samples") and hasattr(res, "param_names"):
        # pack ctx into a serializable dict (optional)
        ctx = getattr(res, "ctx", None)
        ctx_dict: Dict[str, Any] = {}
        if ctx is not None:
            dist = getattr(ctx, "distance", None)
            dist_dict: Dict[str, Any] = {}
            if dist is not None:
                dist_dict = dict(z=getattr(dist, "z", None), DL_cm=getattr(dist, "DL_cm", None))
            filters = getattr(ctx, "filters", {}) or {}
            filters = {str(k): float(v) for k, v in dict(filters).items()}
            y_kind = str(getattr(ctx, "y_kind", "mag"))
            ctx_dict = dict(distance=dist_dict, filters=filters, y_kind=y_kind)

        return dict(
            samples=np.asarray(res.samples, float),
            log_prob=np.asarray(getattr(res, "log_prob", None), float)
            if getattr(res, "log_prob", None) is not None
            else None,
            param_names=np.asarray(res.param_names, dtype=object),
            all_param_names=np.asarray(getattr(res, "all_param_names", []), dtype=object),
            fixed=dict(getattr(res, "fixed", {}) or {}),
            meta=dict(getattr(res, "meta", {}) or {}),
            ctx=ctx_dict,
            model=str(getattr(res, "model", "")),
            sampler=str(getattr(res, "sampler", "")),
        )

    raise TypeError(
        "plot only accepts FitResult or loaded dict.\n"
        "If you have a .npz path, load it first via your io.load(), then pass the loaded dict."
    )


def _make_ctx_from_dict(ctx_dict: Dict[str, Any]):
    """Lazy import Context/Distance to avoid circular import."""
    from ..api import Context, Distance

    ctx_dict = dict(ctx_dict or {})
    dist = dict(ctx_dict.get("distance", {}) or {})
    distance = Distance(z=dist.get("z", None), DL_cm=dist.get("DL_cm", None))

    filters = dict(ctx_dict.get("filters", {}) or {})
    filters = {str(k): float(v) for k, v in filters.items()}

    y_kind = str(ctx_dict.get("y_kind", "mag"))
    return Context(distance=distance, filters=filters, y_kind=y_kind)


def _get_model_name(loaded: Dict[str, Any], fallback: Optional[str] = None) -> str:
    if loaded.get("model"):
        return str(loaded["model"])
    meta = dict(loaded.get("meta", {}) or {})
    if meta.get("model"):
        return str(meta["model"])
    if fallback:
        return str(fallback)
    raise ValueError("Cannot determine model name; pass model=... to fit_*().")


def _best_subset(samples: np.ndarray, log_prob: Optional[np.ndarray], max_n: int) -> np.ndarray:
    samples = np.asarray(samples, float)
    if samples.shape[0] <= max_n:
        return samples
    if log_prob is not None:
        lp = np.asarray(log_prob, float)
        order = np.argsort(lp)[::-1]
        return samples[order[:max_n]]
    rng = np.random.default_rng(123)
    idx = rng.choice(samples.shape[0], size=max_n, replace=False)
    return samples[idx]


def _paramdict_from_samples(
    loaded: Dict[str, Any],
    samples: np.ndarray,
    *,
    use: str = "median",
) -> Dict[str, float]:
    """Full parameter dict including fixed + center of sampled parameters."""
    fixed = dict(loaded.get("fixed", {}) or {})
    pnames = [str(x) for x in loaded["param_names"]]
    samples = np.asarray(samples, float)

    vals = dict(fixed)
    center = np.nanmean(samples, axis=0) if use == "mean" else np.nanmedian(samples, axis=0)
    for i, n in enumerate(pnames):
        vals[n] = float(center[i])
    return vals


def _theta_from_paramdict(loaded: Dict[str, Any], p: Dict[str, float]) -> Tuple[Tuple[float, ...], float]:
    """
    theta tuple in the order of all_param_names excluding t_shift_days; and t_shift_days.
    If all_param_names is missing, fallback to param_names order.
    """
    all_names = [str(x) for x in loaded.get("all_param_names", []) if str(x)]
    if not all_names:
        all_names = [str(x) for x in loaded["param_names"]]

    t_shift = float(p.get("t_shift_days", 0.0))

    theta_names = [n for n in all_names if n != "t_shift_days"]
    theta: List[float] = []
    for n in theta_names:
        if n not in p:
            raise KeyError(f"Parameter '{n}' missing when building theta. Check all_param_names/fixed/param_names.")
        theta.append(float(p[n]))

    return tuple(theta), t_shift


# -----------------------------------------------------------------------------
# public: corner
# -----------------------------------------------------------------------------

def corner(
    res: Union[Dict[str, Any], Any],
    *,
    truths: Optional[Sequence[float]] = None,
    q: Tuple[float, float] = (0.005, 0.995),
    pad_frac: float = 0.06,
    smooth: float = 1.0,
    bins: int = 35,
    max_points: int = 12000,
    debug: bool = False,
    title_fmt: str = ".3g",
    levels: Tuple[float, float, float] = (0.68, 0.95, 0.997),
):
    """
    Corner plot of posterior samples (requires `corner` installed).
    Returns fig (does NOT show).
    """
    loaded = _to_loaded(res)
    samples = np.asarray(loaded["samples"], float)
    labels = [str(x) for x in loaded["param_names"]]
    logp = loaded.get("log_prob", None)
    logp = np.asarray(logp, float) if logp is not None else None

    samp = _best_subset(samples, logp, max_n=max_points)

    # truths default: median of posterior (including fixed if present)
    if truths is None:
        pmed = _paramdict_from_samples(loaded, samp, use="median")
        truths = [pmed.get(k, np.nan) for k in labels]

    # robust ranges
    lo = np.quantile(samp, q[0], axis=0)
    hi = np.quantile(samp, q[1], axis=0)
    ranges = []
    for i in range(samp.shape[1]):
        w = float(hi[i] - lo[i])
        pad = pad_frac * w if w > 0 else 1e-6
        ranges.append((float(lo[i] - pad), float(hi[i] + pad)))

    if debug:
        for name, (a, b) in zip(labels, ranges):
            print(f"{name:12s}  [{a:.4g}, {b:.4g}]  width={b-a:.4g}")

    with plt.rc_context(_journal_rc()):
        try:
            import corner as _corner
        except Exception as e:
            raise ImportError("`corner` package not found. Install with: pip install corner") from e

        fig = _corner.corner(
            samp,
            labels=labels,
            truths=truths,
            range=ranges,
            bins=bins,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=title_fmt,
            title_quantiles=[0.16, 0.5, 0.84],
            plot_datapoints=False,
            fill_contours=True,
            smooth=smooth,
            levels=levels,
        )

        # important: avoid notebook double-render
        plt.close(fig)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        return fig


# -----------------------------------------------------------------------------
# public: fit_bol
# -----------------------------------------------------------------------------

def fit_bol(
    res: Union[Dict[str, Any], Any],
    data,
    *,
    ctx=None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    # plotting controls
    t_pad: float = 50.0,
    n_t: int = 300,
    n_draws: int = 0,  # posterior band draws (0 = off)
    band_quantiles: Tuple[float, float] = (0.16, 0.84),
    alpha_band: float = 0.18,
    lw_model: float = 2.2,
    ms_data: float = 6.0,
    capsize: float = 2.0,
    figsize: Tuple[float, float] = (7.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Bolometric/thermal single-curve fit plot.
    data must provide: t_days, y, yerr
    """
    from ..api import predict_bol  # lazy import

    loaded = _to_loaded(res)

    # default model_kwargs: use saved meta if exists
    mk_saved = dict((loaded.get("meta", {}) or {}).get("model_kwargs", {}) or {})
    model_kwargs = dict(mk_saved if model_kwargs is None else model_kwargs)

    if ctx is None:
        # FitResult has ctx packed into loaded["ctx"] too, so ok
        ctx_dict = loaded.get("ctx", {}) or {}
        if ctx_dict:
            ctx = _make_ctx_from_dict(ctx_dict)
        else:
            raise ValueError("ctx is required (not found in loaded result).")

    y_kind = str(getattr(ctx, "y_kind", "mag")).lower()  # usually irrelevant for bol, but keep for label

    model_name = _get_model_name(loaded, fallback=model)

    samples = np.asarray(loaded["samples"], float)
    logp = loaded.get("log_prob", None)
    logp = np.asarray(logp, float) if logp is not None else None
    subset = _best_subset(samples, logp, max_n=max(3000, (n_draws * 20) if n_draws else 3000))

    pmed = _paramdict_from_samples(loaded, subset, use="median")
    theta_med, t_shift_med = _theta_from_paramdict(loaded, pmed)

    # time grid for model curve: x-axis uses t_plot, model uses t_plot + shift
    t_obs = np.asarray(data.t_days, float).reshape(-1)
    tmin = float(np.nanmin(t_obs))
    tmax = float(np.nanmax(t_obs) + float(t_pad))
    t_plot = np.linspace(tmin, tmax, int(n_t))

    y_obs = np.asarray(data.y, float).reshape(-1)
    y_err = np.asarray(data.yerr, float).reshape(-1)

    with plt.rc_context(_journal_rc()):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        y_line = predict_bol(
            model=model_name,
            theta=theta_med,
            ctx=ctx,
            t_days=t_plot ,
            **model_kwargs,
        )

        ax.errorbar(
                t_obs, y_obs - t_shift_med, yerr=y_err,
                fmt="o",
                ms=ms_data,    
                mec="k",        
                mew=1.2,              
                elinewidth=1.0,
                capsize=capsize,
                alpha=0.9,
                label="data",
            )

        ax.plot(t_plot, y_line, lw=lw_model, label="median model")

        if n_draws and n_draws > 0:
            rng = np.random.default_rng(123)
            draw = min(int(n_draws), subset.shape[0])
            jj = rng.choice(subset.shape[0], size=draw, replace=False)

            ys = []
            pnames = [str(x) for x in loaded["param_names"]]
            fixed = dict(loaded.get("fixed", {}) or {})

            for j in jj:
                p = dict(fixed)
                for i, name in enumerate(pnames):
                    p[name] = float(subset[j, i])
                theta_j, tshift_j = _theta_from_paramdict(loaded, p)
                yj = predict_bol(
                    model=model_name,
                    theta=theta_j,
                    ctx=ctx,
                    t_days=t_plot + tshift_j,
                    **model_kwargs,
                )
                ys.append(yj)

            ys = np.asarray(ys, float)
            lo = np.nanquantile(ys, band_quantiles[0], axis=0)
            hi = np.nanquantile(ys, band_quantiles[1], axis=0)
            ax.fill_between(
                t_plot, lo, hi,
                alpha=alpha_band,
                label=f"{band_quantiles[0]:.2f}-{band_quantiles[1]:.2f} posterior",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Since First Detection (days)")
        ax.set_ylabel("Bolometric Luminosity")
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.minorticks_on()
        ax.grid(alpha=0.15)
        ax.legend(ncol=1)
        fig.tight_layout()

        plt.close(fig)
        return fig


# -----------------------------------------------------------------------------
# public: fit_multiband
# -----------------------------------------------------------------------------

def fit_multiband(
    res: Union[Dict[str, Any], Any],
    data,
    *,
    ctx=None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    # plotting controls
    t_pad: float = 50.0,
    n_t: int = 300,
    n_draws: int = 0,  # posterior band draws (0 = off)
    band_quantiles: Tuple[float, float] = (0.16, 0.84),
    alpha_band: float = 0.18,
    lw_model: float = 2.2,
    ms_data: float = 6.0,
    capsize: float = 2.0,
    figsize: Tuple[float, float] = (7.0, 5.0),
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Multi-band fit plot in a SINGLE panel.
    data must provide: t_days, band, y, yerr
    Bands are grouped exactly as they appear (case-sensitive, no normalization).
    """
    from ..api import predict_multiband  # lazy import

    loaded = _to_loaded(res)

    # default model_kwargs: use saved meta if exists
    mk_saved = dict((loaded.get("meta", {}) or {}).get("model_kwargs", {}) or {})
    model_kwargs = dict(mk_saved if model_kwargs is None else model_kwargs)

    if ctx is None:
        ctx_dict = loaded.get("ctx", {}) or {}
        if ctx_dict:
            ctx = _make_ctx_from_dict(ctx_dict)
        else:
            raise ValueError("ctx is required (not found in loaded result).")

    y_kind = str(getattr(ctx, "y_kind", "mag")).lower()
    model_name = _get_model_name(loaded, fallback=model)

    samples = np.asarray(loaded["samples"], float)
    logp = loaded.get("log_prob", None)
    logp = np.asarray(logp, float) if logp is not None else None
    subset = _best_subset(samples, logp, max_n=max(3000, (n_draws * 20) if n_draws else 3000))

    pmed = _paramdict_from_samples(loaded, subset, use="median")
    theta_med, t_shift_med = _theta_from_paramdict(loaded, pmed)

    t_obs = np.asarray(data.t_days, float).reshape(-1)
    tmin = float(np.nanmin(t_obs))
    tmax = float(np.nanmax(t_obs) + float(t_pad))
    t_plot = np.linspace(tmin, tmax, int(n_t))

    y_obs = np.asarray(data.y, float).reshape(-1)
    y_err = np.asarray(data.yerr, float).reshape(-1)
    band = np.asarray(data.band).reshape(-1)

    bands = _unique_in_order(band.tolist())

    with plt.rc_context(_journal_rc()):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {b: cycle[i % len(cycle)] for i, b in enumerate(bands)}

        fixed = dict(loaded.get("fixed", {}) or {})
        pnames = [str(x) for x in loaded["param_names"]]

        for b in bands:
            c = color_map[b]
            m = (band == b)

            ax.errorbar(
                t_obs[m], y_obs[m], yerr=y_err[m],
                fmt=".", ms=ms_data, capsize=capsize,
                color=c, alpha=0.9, label=f"{b} data",
            )

            # x-axis uses t_plot, model uses (t_plot + shift)
            y_line = predict_multiband(
                model=model_name,
                theta=theta_med,
                ctx=ctx,
                t_days=t_plot ,
                band=np.array([b] * len(t_plot), dtype=object),
                **model_kwargs,
            )
            ax.plot(t_plot- t_shift_med, y_line, lw=lw_model, color=c, label=f"{b} model")

            if n_draws and n_draws > 0:
                rng = np.random.default_rng(1000 + (hash(str(b)) % 1000))
                draw = min(int(n_draws), subset.shape[0])
                jj = rng.choice(subset.shape[0], size=draw, replace=False)

                ys = []
                for j in jj:
                    p = dict(fixed)
                    for i, name in enumerate(pnames):
                        p[name] = float(subset[j, i])
                    theta_j, tshift_j = _theta_from_paramdict(loaded, p)

                    yj = predict_multiband(
                        model=model_name,
                        theta=theta_j,
                        ctx=ctx,
                        t_days=t_plot + tshift_j,
                        band=np.array([b] * len(t_plot), dtype=object),
                        **model_kwargs,
                    )
                    ys.append(yj)

                ys = np.asarray(ys, float)
                lo = np.nanquantile(ys, band_quantiles[0], axis=0)
                hi = np.nanquantile(ys, band_quantiles[1], axis=0)
                ax.fill_between(t_plot, lo, hi, color=c, alpha=alpha_band)
        ax.set_ylim(min(y_obs)-2, max(y_obs)+2)  # invert y-axis for mag/Fnu
        ax.set_xlim(tmin- t_shift_med-10, tmax+10 )
        ax.set_xlabel("Since First Detection (days)")
        ax.set_ylabel("AB mag" if y_kind == "mag" else "F$_\\nu$")
        _maybe_invert_mag_axis(ax, y_kind)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.minorticks_on()
        ax.grid(alpha=0.15)
        ax.legend(ncol=2, fontsize=10)
        fig.tight_layout()

        plt.close(fig)
        return fig


__all__ = ["corner", "fit_multiband", "fit_bol"]
