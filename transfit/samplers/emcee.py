# transfit/samplers/emcee.py
from __future__ import annotations

from typing import Callable, Optional, Tuple, Dict, Any, Sequence, Union
import numpy as np
import emcee

# ---- Parallel helpers ----
import multiprocessing as mp

# loky/joblib (best for Windows notebooks)
try:
    from joblib.externals.loky import get_reusable_executor  # type: ignore
    _HAS_LOKY = True
except Exception:
    _HAS_LOKY = False


ArrayLike = Union[np.ndarray, Sequence[float]]


def _default_moves():
    # More stable than pure Stretch, especially for correlated parameters.
    return [
        (emcee.moves.StretchMove(a=2.0), 0.6),
        (emcee.moves.DEMove(), 0.4),
    ]


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

    # Candidate count: at least oversample*nwalkers and at least 2000.
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
        # Too few feasible points: repeatedly resample until nwalkers are valid.
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
        # Add jitter to spread walkers.
        p0 = p0 + jitter * rng.normal(size=p0.shape) * scale
        return p0

    # Select top_frac of feasible points by log-probability.
    idx_ok = np.where(ok)[0]
    logp_ok = logp[idx_ok]
    # Sort from high to low.
    order = np.argsort(logp_ok)[::-1]
    idx_ok_sorted = idx_ok[order]

    k = max(int(len(idx_ok_sorted) * top_frac), nwalkers)
    pool_idx = idx_ok_sorted[:k]

    # Draw nwalkers centers.
    chosen = rng.choice(pool_idx, size=nwalkers, replace=(len(pool_idx) < nwalkers))
    p0 = cand[chosen].copy()

    # Jitter to avoid fully overlapping walkers and narrow-line degeneracy.
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


# -------------------------
# Loky pool wrapper (Windows notebook friendly)
# -------------------------

class LokyPool:
    """
    Minimal pool-like wrapper exposing .map, compatible with emcee.
    Uses loky reusable executor (via joblib).
    """
    def __init__(self, nproc: int):
        if not _HAS_LOKY:
            raise ImportError("loky/joblib not available. Please `pip install joblib`.")
        self._nproc = int(nproc)
        self._executor = get_reusable_executor(max_workers=self._nproc)

    def map(self, func, iterable):
        # emcee expects something iterable; list() is fine
        return list(self._executor.map(func, iterable))

    def close(self):
        # Reusable executor: don't force shutdown (keeps it stable in notebooks)
        pass

    def join(self):
        pass


def run_emcee(
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
    # Use robust prior initialization by default.
    robust_init: bool = True,
    jitter: float = 1e-2,
    moves: Optional[Sequence] = None,
    backend=None,
    # ---------- Cross-platform parallel ----------
    nproc: Optional[int] = None,
    parallel_backend: str = "auto",     # "auto" | "loky" | "mp" | "none"
    mp_start_method: str = "spawn",     # safest for Win/mac/Linux
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      samples: (Ns, ndim) flat
      logp:    (Ns,)
      meta:    dict

    Parallel behavior:
      1) If `pool` is passed explicitly, use it as-is.
      2) Otherwise, if `nproc > 1`:
         - parallel_backend="auto": prefer loky if available, else multiprocessing
         - parallel_backend="loky": force loky (recommended for Windows notebooks)
         - parallel_backend="mp": use multiprocessing.Pool
         - parallel_backend="none": disable parallel execution
    """
    rng = np.random.default_rng(seed)
    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burnin = int(burnin)
    thin = int(thin)

    ndim = len(prior.param_names)

    # -------- Initialize p0 --------
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
        # Allow user-provided p0 with shape (nwalkers, ndim) or (ndim,).
        p0 = np.asarray(init, float)
        if p0.shape == (ndim,):
            scale = _get_prior_scale(prior, ndim)
            p0 = p0[None, :] + jitter * rng.normal(size=(nwalkers, ndim)) * scale
        if p0.shape != (nwalkers, ndim):
            raise ValueError(
                f"init must be 'prior' or array shape (nwalkers, ndim)={nwalkers, ndim} or (ndim,)"
            )

    # Guardrail: ensure all walker initial points are valid.
    p0 = _ensure_finite_p0(p0, lnprob, prior, rng)

    # -------- moves --------
    if moves is None:
        moves = _default_moves()

    # -------- Create pool if needed --------
    created_pool = None
    used_backend = None

    if pool is None and nproc is not None and int(nproc) > 1:
        b = str(parallel_backend).lower().strip()

        if b == "auto":
            # Best default for Windows notebooks: loky if available, else mp.
            b = "loky" if _HAS_LOKY else "mp"

        if b == "loky":
            created_pool = LokyPool(int(nproc))
            pool = created_pool
            used_backend = "loky"
        elif b == "mp":
            ctx = mp.get_context(mp_start_method)
            created_pool = ctx.Pool(processes=int(nproc))
            pool = created_pool
            used_backend = f"mp:{mp_start_method}"
        elif b == "none":
            pool = None
            used_backend = "none"
        else:
            raise ValueError(f"parallel_backend must be one of 'auto','loky','mp','none', got {parallel_backend!r}")

    try:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob, pool=pool, moves=moves, backend=backend
        )

        # -------- run --------
        if burnin > 0:
            # Two-stage run: burn-in, reset, then production chain.
            state = sampler.run_mcmc(p0, burnin, progress=progress)
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=progress)
            discard = 0
        else:
            sampler.run_mcmc(p0, nsteps, progress=progress)
            discard = 0

        chain = sampler.get_chain(discard=discard, thin=thin, flat=True)
        logp = sampler.get_log_prob(discard=discard, thin=thin, flat=True)

        # -------- diagnostics --------
        try:
            tau = sampler.get_autocorr_time(tol=0)
            tau_max = float(np.max(tau))
        except Exception:
            tau = None
            tau_max = float("nan")

        meta = dict(
            nwalkers=nwalkers,
            nsteps=nsteps,
            burnin=burnin,
            thin=thin,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            autocorr_time=tau,
            autocorr_time_max=tau_max,
            robust_init=robust_init,
            jitter=jitter,
            # parallel info
            nproc=int(nproc) if (nproc is not None) else 1,
            parallel_backend=used_backend,
        )
        return np.asarray(chain, float), np.asarray(logp, float), meta

    finally:
        # Close only pools created inside this function.
        if created_pool is not None:
            try:
                created_pool.close()
            finally:
                try:
                    created_pool.join()
                except Exception:
                    pass
