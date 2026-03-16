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
        param_names=np.asarray(res.param_names, dtype=str),
        all_param_names=np.asarray(res.all_param_names, dtype=str),
        fixed=json.dumps(res.fixed or {}, ensure_ascii=False, default=str),
        meta=json.dumps(meta, ensure_ascii=False, default=str),
        ctx=json.dumps(_ctx_to_dict(getattr(res, "ctx", None)), ensure_ascii=False, default=str),
        model=str(res.model),
        sampler=str(res.sampler),
    )

    np.savez_compressed(p, **payload)
    return str(p)


def _required_fields() -> set[str]:
    return {
        "samples",
        "log_prob",
        "param_names",
        "all_param_names",
        "fixed",
        "meta",
        "ctx",
        "model",
        "sampler",
    }


def _as_text_scalar(value: Any, *, field: str) -> str:
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"Field '{field}' must be stored as a scalar string.")
    item = arr.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _as_text_array(value: Any, *, field: str, allow_object: bool = False) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"Field '{field}' must be a 1D string array.")
    if arr.dtype == object:
        if not allow_object:
            raise ValueError(
                f"Field '{field}' uses object dtype and requires pickle. "
                "Use a trusted legacy file or re-save with a newer TransFit version."
            )
        return np.asarray([str(x) for x in arr.tolist()], dtype=str)
    return arr.astype(str)


def _load_json_dict(value: Any, *, field: str) -> Dict[str, Any]:
    try:
        obj = json.loads(_as_text_scalar(value, field=field))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Field '{field}' does not contain valid JSON.") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Field '{field}' must decode to a JSON object.")
    return obj


def _validate_ctx_dict(ctx: Dict[str, Any]) -> Dict[str, Any]:
    distance = dict(ctx.get("distance", {}) or {})
    for key in ("z", "DL_cm"):
        val = distance.get(key, None)
        if val is not None:
            distance[key] = float(val)

    filters = dict(ctx.get("filters", {}) or {})
    filters = {str(k): float(v) for k, v in filters.items()}
    y_kind = str(ctx.get("y_kind", "mag"))
    return {"distance": distance, "filters": filters, "y_kind": y_kind}


def _decode_loaded_payload(out: Dict[str, Any], *, allow_object_strings: bool = False) -> Dict[str, Any]:
    missing = sorted(_required_fields() - set(out))
    if missing:
        raise ValueError(f"Missing required field(s) in saved result: {missing}")

    samples = np.asarray(out["samples"], float)
    log_prob = np.asarray(out["log_prob"], float).reshape(-1)
    if samples.ndim != 2:
        raise ValueError("Field 'samples' must be a 2D array.")
    if log_prob.ndim != 1 or log_prob.shape[0] != samples.shape[0]:
        raise ValueError(
            "Field 'log_prob' must be a 1D array with the same number of rows as 'samples'."
        )

    param_names = _as_text_array(
        out["param_names"],
        field="param_names",
        allow_object=allow_object_strings,
    )
    all_param_names = _as_text_array(
        out["all_param_names"],
        field="all_param_names",
        allow_object=allow_object_strings,
    )
    if param_names.size != samples.shape[1]:
        raise ValueError(
            "Field 'param_names' must have the same length as the sampled parameter dimension."
        )

    fixed = _load_json_dict(out["fixed"], field="fixed")
    meta = _load_json_dict(out["meta"], field="meta")
    ctx = _validate_ctx_dict(_load_json_dict(out["ctx"], field="ctx"))
    model = _as_text_scalar(out["model"], field="model")
    sampler = _as_text_scalar(out["sampler"], field="sampler")

    return dict(
        samples=samples,
        log_prob=log_prob,
        param_names=param_names,
        all_param_names=all_param_names,
        fixed=fixed,
        meta=meta,
        ctx=ctx,
        model=model,
        sampler=sampler,
    )


def load(path: Union[str, Path], *, trusted: bool = False) -> Dict[str, Any]:
    """
    Load fit data as a plain dict (without rebuilding FitResult/dataclasses).
    The returned dict includes:
      samples, log_prob, param_names, all_param_names, fixed(dict), meta(dict), ctx(dict), model(str), sampler(str), path(str)

    By default, loading is done with pickle disabled. Set `trusted=True`
    only when reading legacy files from a source you trust.
    """
    p = Path(path).expanduser().resolve()
    try:
        with np.load(p, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
    except ValueError as exc:
        if not trusted:
            raise ValueError(
                "This file requires pickle-enabled loading, which is disabled by default. "
                "If you trust the file source, call load(path, trusted=True)."
            ) from exc
        with np.load(p, allow_pickle=True) as z:
            out = {k: z[k] for k in z.files}

    decoded = _decode_loaded_payload(out, allow_object_strings=trusted)
    decoded["path"] = str(p)
    return decoded


__all__ = ["save", "load", "default_outpath"]
