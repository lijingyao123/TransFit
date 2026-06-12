from __future__ import annotations

import statistics as stats
import time

import numpy as np

from transfit.models.csm import CSMModel


THETA = (5.0, 1.0, 1.0, 3000.0, 0.2, 2.0, 0.5, 4500.0)
SOLVER_KWARGS = dict(Nx=100, Ny=1000, t_max_days=150.0)


def _time_solver(model: CSMModel, solver: str, repeats: int = 20) -> tuple[float, float]:
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.calculate_light_curve(THETA, shock_ode_solver=solver, **SOLVER_KWARGS)
        values.append(time.perf_counter() - t0)
    return stats.median(values), stats.mean(values)


def _luminosity_error(reference, candidate) -> tuple[float, float, float]:
    t_ref, l_ref, _, _ = reference
    t_new, l_new, _, _ = candidate
    l_on_ref = 10.0 ** np.interp(
        t_ref,
        t_new,
        np.log10(np.maximum(l_new, 1.0e-300)),
    )
    valid = (l_ref > 1.0e-200) & np.isfinite(l_on_ref)
    rel = np.abs(l_on_ref[valid] - l_ref[valid]) / np.maximum(np.abs(l_ref[valid]), 1.0e-300)
    rms_log = np.sqrt(
        np.mean(
            (
                np.log10(np.maximum(l_on_ref[valid], 1.0e-300))
                - np.log10(np.maximum(l_ref[valid], 1.0e-300))
            )
            ** 2
        )
    )
    return float(np.max(rel)), float(np.percentile(rel, 95.0)), float(rms_log)


def main() -> None:
    model = CSMModel()

    # Warm up Numba and SciPy paths outside the timed region.
    reference = model.calculate_light_curve(THETA, shock_ode_solver="scipy", **SOLVER_KWARGS)
    candidate = model.calculate_light_curve(THETA, shock_ode_solver="numba", **SOLVER_KWARGS)

    scipy_median, scipy_mean = _time_solver(model, "scipy", repeats=10)
    numba_median, numba_mean = _time_solver(model, "numba", repeats=30)
    max_rel, p95_rel, rms_log = _luminosity_error(reference, candidate)

    print("CSM shock ODE benchmark")
    print(f"grid: Nx={SOLVER_KWARGS['Nx']} Ny={SOLVER_KWARGS['Ny']}")
    print(f"scipy/BDF median: {scipy_median:.6f} s  mean: {scipy_mean:.6f} s")
    print(f"numba/RK4 median: {numba_median:.6f} s  mean: {numba_mean:.6f} s")
    print(f"speedup: {scipy_median / numba_median:.2f}x")
    print(f"Lbol max relative error: {max_rel:.3e}")
    print(f"Lbol p95 relative error: {p95_rel:.3e}")
    print(f"Lbol RMS log10 error: {rms_log:.3e}")


if __name__ == "__main__":
    main()
