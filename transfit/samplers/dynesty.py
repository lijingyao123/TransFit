from __future__ import annotations

from typing import Callable, Optional, Tuple, Dict, Any
import multiprocessing as mp
import numpy as np


def _build_prior_transform(bounds: np.ndarray, log_flags=None):
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    lf = np.asarray(log_flags, bool) if log_flags is not None else np.zeros(lo.size, dtype=bool)
    if lf.size != lo.size:
        raise ValueError("log_flags size must match bounds ndim for dynesty.")

    span_lin = hi - lo
    span_log = np.log(hi) - np.log(lo)

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        u = np.asarray(unit_cube, float)
        x = lo + span_lin * u
        if np.any(lf):
            x = np.asarray(x, float)
            x[lf] = np.exp(np.log(lo[lf]) + span_log[lf] * u[lf])
        return x

    return prior_transform


def run_dynesty(
    *,
    lnprob: Callable[[np.ndarray], float],
    prior,
    nlive: int = 200,
    sample: str = "rwalk",
    bound: str = "multi",
    dlogz: float = 0.1,
    maxiter: Optional[int] = None,
    maxcall: Optional[int] = None,
    seed: Optional[int] = None,
    progress: bool = True,
    nsamples: Optional[int] = None,
    add_live: bool = True,
    pool=None,
    queue_size: Optional[int] = None,
    nproc: Optional[int] = None,
    mp_start_method: str = "spawn",
    bootstrap: int = 0,
    walks: int = 25,
    slices: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run dynesty nested sampling and return posterior samples/log-prob arrays.

    Notes:
    - The posterior samples are drawn by weighted resampling from dynesty outputs.
    - `log_prob` is computed as log-likelihood + log-prior for the returned samples.
    """
    try:
        import dynesty  # type: ignore
    except Exception as exc:
        raise ImportError(
            "dynesty backend requires package `dynesty`. "
            "Install it with `pip install dynesty`."
        ) from exc

    if not hasattr(prior, "bounds"):
        raise ValueError("dynesty backend requires a prior object with `.bounds`.")

    bounds = np.asarray(prior.bounds, float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("prior.bounds must have shape (ndim, 2) for dynesty.")

    ndim = int(bounds.shape[0])
    nlive = int(nlive)
    log_flags = getattr(prior, "log_flags", None)
    if log_flags is None:
        log_flags = np.zeros(ndim, dtype=bool)
    else:
        log_flags = np.asarray(log_flags, bool).reshape(-1)
        if log_flags.size != ndim:
            raise ValueError("prior.log_flags size does not match prior bounds ndim.")
    prior_transform = _build_prior_transform(bounds, log_flags=log_flags)

    def loglike(theta: np.ndarray) -> float:
        x = np.asarray(theta, float)
        lp = float(prior.lnprior(x))
        if not np.isfinite(lp):
            return -np.inf
        try:
            post = float(lnprob(x))
        except Exception:
            return -np.inf
        if not np.isfinite(post):
            return -np.inf
        return float(post - lp)

    created_pool = None
    used_backend = "none"
    if pool is None and nproc is not None and int(nproc) > 1:
        ctx = mp.get_context(mp_start_method)
        created_pool = ctx.Pool(processes=int(nproc))
        pool = created_pool
        used_backend = f"mp:{mp_start_method}"
    elif pool is not None:
        used_backend = "external_pool"

    if queue_size is None and pool is not None:
        queue_size = int(nproc) if (nproc is not None) else 1

    rng = np.random.default_rng(seed)

    try:
        sampler_kwargs = dict(
            nlive=nlive,
            sample=sample,
            bound=bound,
            pool=pool,
            queue_size=queue_size,
            rstate=rng,
            bootstrap=int(bootstrap),
            walks=int(walks),
            slices=int(slices),
        )
        try:
            sampler = dynesty.NestedSampler(
                loglike,
                prior_transform,
                ndim,
                **sampler_kwargs,
            )
        except TypeError:
            sampler_kwargs.pop("bootstrap", None)
            sampler_kwargs.pop("walks", None)
            sampler_kwargs.pop("slices", None)
            sampler = dynesty.NestedSampler(
                loglike,
                prior_transform,
                ndim,
                **sampler_kwargs,
            )

        run_kwargs = dict(
            dlogz=float(dlogz),
            maxiter=maxiter,
            maxcall=maxcall,
            add_live=bool(add_live),
            print_progress=bool(progress),
        )
        try:
            sampler.run_nested(**run_kwargs)
        except TypeError:
            run_kwargs.pop("add_live", None)
            sampler.run_nested(**run_kwargs)

        res = sampler.results

        samples_raw = np.asarray(res.samples, float)
        logl_raw = np.asarray(res.logl, float).reshape(-1)
        logwt = np.asarray(res.logwt, float).reshape(-1)
        if samples_raw.ndim != 2 or samples_raw.shape[0] == 0:
            raise RuntimeError("dynesty returned no samples.")

        lw = logwt - np.max(logwt)
        weights = np.exp(lw)
        wsum = float(np.sum(weights))
        if not np.isfinite(wsum) or wsum <= 0:
            weights = np.ones_like(weights, dtype=float) / float(weights.size)
        else:
            weights = weights / wsum

        if nsamples is None:
            nsamples = samples_raw.shape[0]
        nsamples = int(nsamples)
        if nsamples <= 0:
            raise ValueError("nsamples must be > 0 for dynesty.")

        idx = rng.choice(samples_raw.shape[0], size=nsamples, replace=True, p=weights)
        samples = np.asarray(samples_raw[idx], float)
        logp = np.asarray(logl_raw[idx], float)

        try:
            lp_arr = np.asarray([prior.lnprior(s) for s in samples], float)
            finite_lp = np.isfinite(lp_arr)
            logp[finite_lp] = logp[finite_lp] + lp_arr[finite_lp]
            logp[~finite_lp] = -np.inf
        except Exception:
            pass

        logz = np.asarray(getattr(res, "logz", []), float)
        logzerr = np.asarray(getattr(res, "logzerr", []), float)

        ncall = getattr(res, "ncall", None)
        if ncall is None:
            ncall_total = None
        else:
            ncall_total = int(np.sum(np.asarray(ncall)))

        niter = getattr(res, "niter", None)
        if niter is None:
            niter = int(samples_raw.shape[0])
        else:
            niter = int(niter)

        meta = dict(
            nlive=nlive,
            sample=sample,
            bound=bound,
            dlogz=float(dlogz),
            maxiter=maxiter,
            maxcall=maxcall,
            log_evidence=(float(logz[-1]) if logz.size else float("nan")),
            log_evidence_err=(float(logzerr[-1]) if logzerr.size else float("nan")),
            niter=niter,
            ncall=ncall_total,
            nsamples_raw=int(samples_raw.shape[0]),
            nsamples_posterior=int(nsamples),
            nproc=int(nproc) if (nproc is not None) else 1,
            parallel_backend=used_backend,
        )
        return samples, logp, meta

    finally:
        if created_pool is not None:
            try:
                created_pool.close()
            finally:
                try:
                    created_pool.join()
                except Exception:
                    pass
