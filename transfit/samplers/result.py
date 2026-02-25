# transfit/samplers/result.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
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

    def _ordered_all_param_names(self) -> List[str]:
        if self.all_param_names:
            return [str(x) for x in self.all_param_names]
        return [str(x) for x in self.param_names]

    def _theta_and_shift_from_param_dict(self, p: Dict[str, float]) -> Tuple[Tuple[float, ...], float]:
        names = self._ordered_all_param_names()
        theta: List[float] = []
        for n in names:
            if n == "t_shift":
                continue
            if n not in p:
                raise KeyError(f"Missing parameter '{n}' when building theta.")
            theta.append(float(p[n]))
        t_shift = float(p.get("t_shift", 0.0))
        return tuple(theta), t_shift

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
    def best_theta_and_shift(self) -> Tuple[Tuple[float, ...], float]:
        """Best-fit model input in API-ready form: (theta, t_shift)."""
        return self._theta_and_shift_from_param_dict(self.best_params_raw)

    @property
    def best_theta(self) -> Tuple[float, ...]:
        """Best-fit theta tuple (without t_shift)."""
        return self.best_theta_and_shift[0]

    @property
    def best_t_shift(self) -> float:
        """Best-fit time shift."""
        return self.best_theta_and_shift[1]

    @property
    def best_fit(self) -> Dict[str, Any]:
        """
        Compact best-fit record:
          - index
          - log_prob
          - params
          - errors (posterior 16-50-84 summary)
          - sample (free vector)
        """
        p_best_raw = self.best_params_raw
        post = self._posterior_interval_map()

        params = {k: self._round3(v) for k, v in p_best_raw.items()}
        errors: Dict[str, Any] = {}
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

        return dict(
            index=self.best_index,
            log_prob=self.best_log_prob,
            params=params,
            errors=errors,
            sample=self.best_sample,
        )
