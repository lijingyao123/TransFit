from __future__ import annotations

import numpy as np

from .core import DustComponent


def _as_lambda_um_array(lambda_um: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(lambda_um, float)
    if arr.size == 0:
        raise ValueError("lambda_um must be non-empty.")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("lambda_um must contain positive finite values.")
    x = 1.0 / arr
    if np.any((x < 0.3) | (x > 10.0)):
        raise ValueError(
            "Extinction laws currently support wavelengths in the range "
            "0.1-3.33 micron (x = 0.3-10 micron^-1)."
        )
    return arr


def _ccm89_ab(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros_like(x, dtype=float)
    b = np.zeros_like(x, dtype=float)

    ir = (x >= 0.3) & (x < 1.1)
    if np.any(ir):
        xx = x[ir]
        a[ir] = 0.574 * xx ** 1.61
        b[ir] = -0.527 * xx ** 1.61

    opt = (x >= 1.1) & (x < 3.3)
    if np.any(opt):
        y = x[opt] - 1.82
        a[opt] = (
            1.0
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b[opt] = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )

    uv = (x >= 3.3) & (x <= 8.0)
    if np.any(uv):
        xx = x[uv]
        fa = np.zeros_like(xx, dtype=float)
        fb = np.zeros_like(xx, dtype=float)
        mask = xx >= 5.9
        if np.any(mask):
            y = xx[mask] - 5.9
            fa[mask] = -0.04473 * y**2 - 0.009779 * y**3
            fb[mask] = 0.2130 * y**2 + 0.1207 * y**3
        a[uv] = 1.752 - 0.316 * xx - 0.104 / ((xx - 4.67) ** 2 + 0.341) + fa
        b[uv] = -3.090 + 1.825 * xx + 1.206 / ((xx - 4.62) ** 2 + 0.263) + fb

    fuv = (x > 8.0) & (x <= 10.0)
    if np.any(fuv):
        y = x[fuv] - 8.0
        a[fuv] = -1.073 - 0.628 * y + 0.137 * y**2 - 0.070 * y**3
        b[fuv] = 13.670 + 4.257 * y - 0.420 * y**2 + 0.374 * y**3

    return a, b


def _odonnell94_ab(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = _ccm89_ab(x)

    opt = (x >= 1.1) & (x < 3.3)
    if np.any(opt):
        y = x[opt] - 1.82
        a[opt] = (
            1.0
            + 0.104 * y
            - 0.609 * y**2
            + 0.701 * y**3
            + 1.137 * y**4
            - 1.718 * y**5
            - 0.827 * y**6
            + 1.647 * y**7
            - 0.505 * y**8
        )
        b[opt] = (
            1.952 * y
            + 2.908 * y**2
            - 3.989 * y**3
            - 7.985 * y**4
            + 11.102 * y**5
            + 5.491 * y**6
            - 10.805 * y**7
            + 3.347 * y**8
        )
    return a, b


def extinction_axav(lambda_um: np.ndarray | float, *, law: str, rv: float) -> np.ndarray:
    arr = _as_lambda_um_array(lambda_um)
    x = 1.0 / arr
    law_n = str(law).strip().lower()
    rv_n = float(rv)
    if not np.isfinite(rv_n) or rv_n <= 0.0:
        raise ValueError("rv must be finite and > 0.")
    if law_n in ("ccm89", "cardelli89", "cardelli_clayton_mathis89"):
        a, b = _ccm89_ab(x)
    elif law_n in ("odonnell94", "o94", "od94"):
        a, b = _odonnell94_ab(x)
        law_n = "odonnell94"
    else:
        raise ValueError(
            f"Unsupported extinction law {law!r}. "
            "Supported laws: 'ccm89', 'odonnell94'."
        )
    out = a + b / rv_n
    return np.asarray(out, float)


def component_extinction_mag(component: DustComponent, *, lambda_um: np.ndarray | float) -> np.ndarray:
    axav = extinction_axav(lambda_um, law=component.law, rv=component.rv)
    return np.asarray(component.ebv * component.rv * axav, float)
