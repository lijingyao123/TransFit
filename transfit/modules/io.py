# transfit/modules/io.py
# -*- coding: utf-8 -*-
"""
transfit I/O:
- save(res, path=None): save MCMC fit results to an NPZ file
- load(path): load an NPZ file into a dict (without rebuilding dataclasses)
The payload includes complete context info (distance + filters + y_kind).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import datetime as _dt
import numpy as np

from ..samplers import FitResult


def default_outpath(model: str, ext: str = ".npz") -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path.cwd() / "mcmc_out"
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / f"fit_{str(model).strip()}_{ts}{ext}")


def _ctx_to_dict(ctx) -> Dict[str, Any]:
    """
    Convert ctx to a JSON-serializable dict.
    This keeps full context metadata in saved results.
    """
    if ctx is None:
        return {}

    dist = getattr(ctx, "distance", None)
    dist_dict = {}
    if dist is not None:
        z = getattr(dist, "z", None)
        DL_cm = getattr(dist, "DL_cm", None)
        dist_dict = {
            "z": None if z is None else float(z),
            "DL_cm": None if DL_cm is None else float(DL_cm),
        }

    filters = getattr(ctx, "filters", {}) or {}
    filters = {str(k): float(v) for k, v in dict(filters).items()}

    y_kind = str(getattr(ctx, "y_kind", "mag"))

    return {"distance": dist_dict, "filters": filters, "y_kind": y_kind}


def save(res: FitResult, path: Union[str, Path, None] = None) -> str:
    """
    Save fit results without mutating `res` or `res.meta`.

    path=None => default `./mcmc_out/fit_<model>_<time>.npz`
    Returns the saved path string.
    """
    if path is None:
        path = default_outpath(str(res.model), ext=".npz")

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(res.meta or {})
    meta.setdefault("saved_at", _dt.datetime.now().isoformat(timespec="seconds"))

    payload = dict(
        samples=np.asarray(res.samples),
        log_prob=np.asarray(res.log_prob),
        param_names=np.asarray(res.param_names, dtype=object),
        all_param_names=np.asarray(res.all_param_names, dtype=object),
        fixed=json.dumps(res.fixed or {}, ensure_ascii=False, default=str),
        meta=json.dumps(meta, ensure_ascii=False, default=str),
        ctx=json.dumps(_ctx_to_dict(getattr(res, "ctx", None)), ensure_ascii=False, default=str),
        model=str(res.model),
        sampler=str(res.sampler),
    )

    np.savez_compressed(p, **payload)
    return str(p)


def load(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load fit data as a plain dict (without rebuilding FitResult/dataclasses).
    The returned dict includes:
      samples, log_prob, param_names, all_param_names, fixed(dict), meta(dict), ctx(dict), model(str), sampler(str), path(str)
    """
    p = Path(path).expanduser().resolve()
    with np.load(p, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}

    # Decode JSON fields.
    out["fixed"] = json.loads(str(out.get("fixed", "{}")))
    out["meta"] = json.loads(str(out.get("meta", "{}")))
    out["ctx"] = json.loads(str(out.get("ctx", "{}")))

    # Normalize string fields.
    if "model" in out:
        out["model"] = str(out["model"])
    if "sampler" in out:
        out["sampler"] = str(out["sampler"])

    out["path"] = str(p)
    return out


__all__ = ["save", "load", "default_outpath"]
