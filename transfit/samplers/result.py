# transfit/samplers/result.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass(frozen=True)
class FitResult:
    """
    Minimal, stable fit result container for v0.x.

    Notes:
    - `ctx` uses `Any` to avoid circular imports (api.Context -> samplers -> api).
    """
    model: str
    ctx: Any
    sampler: str

    # sampling vector (free params only)
    param_names: List[str]
    fixed: Dict[str, float]

    # full parameter order (includes fixed + optional t_shift)
    all_param_names: List[str]

    # flattened samples
    samples: np.ndarray        # (Ns, ndim)
    log_prob: np.ndarray       # (Ns,)

    meta: Dict[str, Any]

    @staticmethod
    def _round3(x: float) -> float:
        return float(np.round(float(x), 3))

    def _param_dict_from_vec(self, vec: np.ndarray) -> Dict[str, float]:
        """
        Build a full parameter dict from one free-parameter vector.

        Order policy:
        1) follow `all_param_names` when available;
        2) append any remaining keys (for robustness).
        """
        vals = dict(self.fixed or {})
        vals.update({k: float(v) for k, v in zip(self.param_names, np.asarray(vec, float))})

        out: Dict[str, float] = {}
        for n in list(self.all_param_names or []):
            if n in vals:
                out[str(n)] = float(vals[n])
        for k, v in vals.items():
            if k not in out:
                out[str(k)] = float(v)
        return out

    def _best_idx(self) -> int:
        lp = np.asarray(self.log_prob, float).reshape(-1)
        if lp.size == 0:
            raise ValueError("log_prob is empty; cannot determine best-fit sample.")

        finite = np.isfinite(lp)
        if np.any(finite):
            idx_pool = np.where(finite)[0]
            return int(idx_pool[int(np.argmax(lp[finite]))])
        return int(np.argmax(lp))

    def _posterior_interval_map(self) -> Dict[str, Any]:
        """
        16-50-84 posterior summaries in sampled-parameter order.
        """
        samp = np.asarray(self.samples, float)
        if samp.ndim != 2 or samp.shape[0] == 0:
            return {}

        out: Dict[str, Any] = {}
        for i, n in enumerate(self.param_names):
            q16, q50, q84 = np.quantile(samp[:, i], [0.16, 0.5, 0.84])
            out[str(n)] = (float(q16), float(q50), float(q84))
        return out

    def median(self) -> Dict[str, float]:
        """Median of posterior, returned as a full parameter dict."""
        samp = np.asarray(self.samples, float)
        if samp.ndim != 2 or samp.shape[0] == 0:
            raise ValueError("samples is empty; cannot compute posterior median.")
        med = np.median(samp, axis=0)
        return self._param_dict_from_vec(med)

    def best(self) -> Dict[str, float]:
        """MAP-like best-fit parameters (argmax over `log_prob`)."""
        return self.best_params

    @property
    def best_index(self) -> int:
        """Index of the best-fit sample (argmax over finite log_prob)."""
        return self._best_idx()

    @property
    def best_log_prob(self) -> float:
        """Best-fit log probability."""
        i = self._best_idx()
        return float(np.asarray(self.log_prob, float).reshape(-1)[i])

    @property
    def best_sample(self) -> np.ndarray:
        """Best-fit free-parameter vector in `param_names` order."""
        i = self._best_idx()
        return np.asarray(self.samples, float)[i].copy()

    @property
    def best_params(self) -> Dict[str, float]:
        """Best-fit full parameter dict (sampled + fixed)."""
        raw = self._param_dict_from_vec(self.best_sample)
        return {k: self._round3(v) for k, v in raw.items()}

    @property
    def best_params_raw(self) -> Dict[str, float]:
        """Best-fit full parameter dict at full precision."""
        return self._param_dict_from_vec(self.best_sample)

    @property
    def best_fit_params(self) -> Dict[str, float]:
        """Alias of `best_params` for explicit readability."""
        return self.best_params

    @property
    def median_params(self) -> Dict[str, float]:
        """Alias of `median()` for explicit readability."""
        return self.median()

    @property
    def best_fit(self) -> Dict[str, Any]:
        """
        Compact best-fit record:
          - index
          - log_prob
          - params / params_fmt
          - errors / errors_fmt (posterior 16-50-84 summary)
          - sample (free vector)
        """
        p_best_raw = self.best_params_raw
        post = self._posterior_interval_map()

        params = {k: self._round3(v) for k, v in p_best_raw.items()}
        params_fmt = {k: f"{float(v):.3f}" for k, v in params.items()}
        errors: Dict[str, Any] = {}
        errors_fmt: Dict[str, Any] = {}
        for k, vbest in p_best_raw.items():
            if k in post:
                q16, q50, q84 = post[k]
                ek = dict(
                    minus=self._round3(q50 - q16),
                    plus=self._round3(q84 - q50),
                    q16=self._round3(q16),
                    q50=self._round3(q50),
                    q84=self._round3(q84),
                    fixed=False,
                )
                errors[k] = ek
                errors_fmt[k] = f"-{ek['minus']:.3f}/+{ek['plus']:.3f} (q16={ek['q16']:.3f}, q84={ek['q84']:.3f})"
            else:
                ek = dict(
                    minus=0.0,
                    plus=0.0,
                    q16=self._round3(vbest),
                    q50=self._round3(vbest),
                    q84=self._round3(vbest),
                    fixed=True,
                )
                errors[k] = ek
                errors_fmt[k] = "fixed"

        return dict(
            index=self.best_index,
            log_prob=self.best_log_prob,
            params=params,
            params_fmt=params_fmt,
            errors=errors,
            errors_fmt=errors_fmt,
            sample=self.best_sample,
        )
