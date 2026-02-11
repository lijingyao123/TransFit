from __future__ import annotations

from typing import Callable, Optional, Tuple, Dict, Any, Sequence, Union
import multiprocessing as mp
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


def _get_prior_scale(prior, ndim: int) -> np.ndarray:
    """Scale used for jitter: prefer prior.bounds width, fallback to 1."""
    if hasattr(prior, "bounds"):
        b = np.asarray(prior.bounds, float)
        if b.shape == (ndim, 2):
            s = b[:, 1] - b[:, 0]
            s[s == 0] = 1.0
            return s
    return np.ones(ndim, float)


def _robust_prior_init(
    *,
    lnprob: Callable[[np.ndarray], float],
    prior,
    nwalkers: int,
    rng: np.random.Generator,
    jitter: float,
    max_draws: int = 20000,
    oversample: int = 20,
    top_frac: float = 0.2,
) -> np.ndarray:
    """
    Robust prior initialization:
    - Draw many candidate points from the prior.
    - Keep only candidates with finite lnprob.
    - Select a top fraction by log-probability.
    - Sample walker centers from that subset and add jitter.
    """
    ndim = len(prior.param_names)
    scale = _get_prior_scale(prior, ndim)

    n_cand = int(min(max(oversample * nwalkers, 2000), max_draws))
    cand = prior.sample(n_cand, rng=rng)

    logp = np.empty(n_cand, dtype=float)
    for i in range(n_cand):
        try:
            lp = lnprob(cand[i])
        except Exception:
            lp = -np.inf
        logp[i] = lp

    ok = np.isfinite(logp)
    if np.sum(ok) < nwalkers:
        p0 = np.empty((nwalkers, ndim), float)
        filled = 0
        tries = 0
        while filled < nwalkers:
            x = prior.sample(1, rng=rng)[0]
            try:
                lp = lnprob(x)
            except Exception:
                lp = -np.inf
            if np.isfinite(lp):
                p0[filled] = x
                filled += 1
            tries += 1
            if tries > max_draws:
                raise RuntimeError(
                    "robust prior init failed: too few finite lnprob points in the prior volume. "
                    "This usually means the model/lnprob is only valid in a tiny region or often returns NaN."
                )
        p0 = p0 + jitter * rng.normal(size=p0.shape) * scale
        return p0

    idx_ok = np.where(ok)[0]
    logp_ok = logp[idx_ok]
    order = np.argsort(logp_ok)[::-1]
    idx_ok_sorted = idx_ok[order]

    k = max(int(len(idx_ok_sorted) * top_frac), nwalkers)
    pool_idx = idx_ok_sorted[:k]

    chosen = rng.choice(pool_idx, size=nwalkers, replace=(len(pool_idx) < nwalkers))
    p0 = cand[chosen].copy()
    p0 = p0 + jitter * rng.normal(size=p0.shape) * scale
    return p0


def _ensure_finite_p0(
    p0: np.ndarray,
    lnprob: Callable[[np.ndarray], float],
    prior,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> np.ndarray:
    """Guardrail: ensure finite lnprob for each initial walker position."""
    p0 = np.asarray(p0, float)
    nwalkers, _ = p0.shape

    ok = np.zeros(nwalkers, dtype=bool)
    tries = 0
    while not np.all(ok):
        for i in range(nwalkers):
            if ok[i]:
                continue
            try:
                lp = lnprob(p0[i])
            except Exception:
                lp = -np.inf
            if np.isfinite(lp):
                ok[i] = True
            else:
                p0[i] = prior.sample(1, rng=rng)[0]
        tries += 1
        if tries >= max_tries:
            bad = np.where(~ok)[0].tolist()
            raise RuntimeError(
                f"Failed to find finite initial positions for walkers: {bad}. "
                "lnprob seems to be -inf/NaN for many prior samples."
            )
    return p0


def run_zeus(
    *,
    lnprob: Callable[[np.ndarray], float],
    prior,
    nwalkers: int,
    nsteps: int,
    burnin: int = 0,
    thin: int = 1,
    seed: Optional[int] = None,
    init: Union[str, np.ndarray] = "prior",
    pool=None,
    progress: bool = True,
    robust_init: bool = True,
    jitter: float = 1e-2,
    moves: Optional[Sequence] = None,
    nproc: Optional[int] = None,
    mp_start_method: str = "spawn",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run zeus MCMC and return flattened samples/log-prob arrays.
    """
    try:
        import zeus  # type: ignore
    except Exception as exc:
        raise ImportError(
            "zeus backend requires package `zeus-mcmc` (import name: zeus). "
            "Install it with `pip install zeus-mcmc`."
        ) from exc

    rng = np.random.default_rng(seed)
    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burnin = int(burnin)
    thin = int(thin)

    ndim = len(prior.param_names)

    if isinstance(init, str) and init == "prior":
        if robust_init:
            p0 = _robust_prior_init(
                lnprob=lnprob,
                prior=prior,
                nwalkers=nwalkers,
                rng=rng,
                jitter=jitter,
            )
        else:
            p0 = prior.sample(nwalkers, rng=rng)
    else:
        p0 = np.asarray(init, float)
        if p0.shape == (ndim,):
            scale = _get_prior_scale(prior, ndim)
            p0 = p0[None, :] + jitter * rng.normal(size=(nwalkers, ndim)) * scale
        if p0.shape != (nwalkers, ndim):
            raise ValueError(
                f"init must be 'prior' or array shape (nwalkers, ndim)={nwalkers, ndim} or (ndim,)"
            )

    p0 = _ensure_finite_p0(p0, lnprob, prior, rng)

    created_pool = None
    used_backend = "none"
    if pool is None and nproc is not None and int(nproc) > 1:
        ctx = mp.get_context(mp_start_method)
        created_pool = ctx.Pool(processes=int(nproc))
        pool = created_pool
        used_backend = f"mp:{mp_start_method}"
    elif pool is not None:
        used_backend = "external_pool"

    try:
        sampler = zeus.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            moves=moves,
            pool=pool,
        )

        sampler.run_mcmc(p0, nsteps, progress=progress)

        discard = max(0, burnin)
        chain = sampler.get_chain(discard=discard, thin=thin, flat=True)
        logp = sampler.get_log_prob(discard=discard, thin=thin, flat=True)

        try:
            tau = sampler.get_autocorr_time()
            tau = np.asarray(tau, float)
            tau_max = float(np.max(tau))
        except Exception:
            tau = None
            tau_max = float("nan")

        acc = getattr(sampler, "acceptance_fraction", None)
        if acc is None:
            acc_mean = float("nan")
        else:
            acc_mean = float(np.mean(np.asarray(acc, float)))

        meta = dict(
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            acceptance_fraction=acc_mean,
            autocorr_time=tau,
            autocorr_time_max=tau_max,
            robust_init=robust_init,
            jitter=jitter,
            nproc=int(nproc) if (nproc is not None) else 1,
            parallel_backend=used_backend,
        )
        return np.asarray(chain, float), np.asarray(logp, float), meta

    finally:
        if created_pool is not None:
            try:
                created_pool.close()
            finally:
                try:
                    created_pool.join()
                except Exception:
                    pass
