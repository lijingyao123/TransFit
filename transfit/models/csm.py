# CSM.py
# -*- coding: utf-8 -*-

import numpy as np
import numba
from scipy.integrate import solve_ivp
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

try:
    from transfit.modules.interp import interp_fit
except ImportError:
    def interp_fit(x_grid, y_grid, x_obs, yscale="linear", fill="edge"):
        x_grid = np.asarray(x_grid, dtype=float)
        y_grid = np.asarray(y_grid, dtype=float)
        x_obs = np.asarray(x_obs, dtype=float)
        if yscale == "log10":
            y_safe = np.maximum(y_grid, 1.0e-300)
            out = 10.0 ** np.interp(
                x_obs,
                x_grid,
                np.log10(y_safe),
                left=np.log10(y_safe[0]),
                right=np.log10(y_safe[-1]),
            )
        else:
            out = np.interp(x_obs, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])
        return float(out) if np.ndim(x_obs) == 0 else out


# CGS constants
PI = np.pi
C_LIGHT = 2.99792458e10
DAY = 86400.0
M_SUN = 1.98847e33
R_SUN = 6.957e10
SIGMA_SB = 5.670374419e-5
H_PLANCK = 6.62607015e-27
K_BOLTZ = 1.380649e-16

DEFAULT_R_CSM_IN_RSUN = 100.0
DEFAULT_EJECTA_N = 10.0
DEFAULT_EJECTA_DELTA = 0.0


# -----------------------------------------------------------------------------
# Thin-shell dynamics and CSM shock heating
# -----------------------------------------------------------------------------

def _integral_power_law(x_lo, x_hi, power):
    if not (x_hi > x_lo > 0.0):
        raise ValueError("Integral bounds must satisfy x_hi > x_lo > 0.")
    if np.isclose(power, -1.0):
        return float(np.log(x_hi / x_lo))
    return float((x_hi ** (power + 1.0) - x_lo ** (power + 1.0)) / (power + 1.0))


def _build_params(
    theta,
    Nx,
    Ny,
    t_max_days,
    s,
    n,
    delta_inner,
    T_floor,
    rtol_ode,
    atol_ode,
    shock_max_step,
    shock_kernel_cells,
    shock_kernel_width_Rsun,
):
    theta = tuple(theta)
    if len(theta) == 7:
        if float(theta[4]) <= 1.0:
            M_ej, E_sn, M_csm, R_csm, kappa, s, eps_sh = theta
            R_in = DEFAULT_R_CSM_IN_RSUN
        elif float(theta[3]) > float(theta[4]):
            M_ej, E_sn, M_csm, R_csm, R_in, kappa, eps_sh = theta
        else:
            M_csm, M_ej, E_sn, R_in, R_csm, kappa, eps_sh = theta
    elif len(theta) == 8:
        if float(theta[4]) <= 1.0:
            M_ej, E_sn, M_csm, R_csm, kappa, s, eps_sh, T_floor = theta
            R_in = DEFAULT_R_CSM_IN_RSUN
        elif float(theta[3]) > float(theta[4]):
            M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, eps_sh = theta
        else:
            M_csm, M_ej, E_sn, R_in, R_csm, kappa, eps_sh, extra = theta
            if float(extra) > 100.0:
                T_floor = extra
            else:
                s = extra
    elif len(theta) == 9:
        if float(theta[3]) > float(theta[4]):
            M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, eps_sh, T_floor = theta
        else:
            M_csm, M_ej, E_sn, R_in, R_csm, kappa, eps_sh, s, T_floor = theta
    elif len(theta) == 10:
        M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, n, delta_inner, eps_sh = theta
    elif len(theta) == 11:
        M_ej, E_sn, M_csm, R_csm, R_in, kappa, s, n, delta_inner, eps_sh, T_floor = theta
    else:
        raise ValueError(
            "CSMModel theta must have length 7, 8, 9, 10, or 11. "
            "Use canonical order "
            "(M_ej, E_sn, M_csm, R_csm_out, kappa, s, eps_sh, T_floor). "
            "Advanced direct calls may include R_csm_in, n, and delta. "
            "Legacy shorter forms may omit s and/or T_floor."
        )

    p = {
        "M_csm": float(M_csm) * M_SUN,
        "M_ej": float(M_ej) * M_SUN,
        "E_sn": float(E_sn) * 1.0e51,
        "R_in": float(R_in) * R_SUN,
        "R_csm": float(R_csm) * R_SUN,
        "kappa": float(kappa),
        "eps_sh": float(eps_sh),
        "T_floor": float(T_floor),
        "s": float(s),
        "n": float(n),
        "delta_inner": float(delta_inner),
        "N_x": int(Nx),
        "N_t": int(Ny),
        "t_max_days": float(t_max_days),
        "rtol_ode": float(rtol_ode),
        "atol_ode": float(atol_ode),
        "shock_max_step": float(shock_max_step),
        "shock_kernel_cells": int(shock_kernel_cells),
        "shock_kernel_width": float(shock_kernel_width_Rsun) * R_SUN,
    }
    p["x_max"] = p["R_csm"] / p["R_in"]
    _validate_params(p)
    return p


def _validate_params(p):
    for name, value in p.items():
        if name in {"N_x", "N_t", "shock_kernel_cells"}:
            continue
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if not (p["x_max"] > 1.0):
        raise ValueError("R_csm must be larger than R_in.")
    if not (p["M_csm"] > 0.0 and p["M_ej"] > 0.0 and p["E_sn"] > 0.0):
        raise ValueError("M_csm, M_ej, and E_sn must be positive.")
    if not (p["kappa"] > 0.0):
        raise ValueError("kappa must be positive.")
    if not (0.0 <= p["eps_sh"] <= 1.0):
        raise ValueError("eps_sh must satisfy 0 <= eps_sh <= 1.")
    if not (0.0 <= p["s"] < 3.0):
        raise ValueError("s must satisfy 0 <= s < 3.")
    if not (p["n"] > 5.0 and p["n"] > p["s"]):
        raise ValueError("The ejecta/CSM model requires n > 5 and n > s.")
    if not (0.0 <= p["delta_inner"] < 3.0):
        raise ValueError("delta must satisfy 0 <= delta < 3.")
    if not (p["T_floor"] > 0.0):
        raise ValueError("T_floor must be positive.")
    if not (p["N_x"] >= 10 and p["N_t"] >= 10):
        raise ValueError("Nx and Ny must be at least 10.")
    if not (p["t_max_days"] > 0.0):
        raise ValueError("t_max_days must be positive.")
    if not (p["shock_max_step"] > 0.0):
        raise ValueError("shock_max_step must be positive.")
    if not (p["shock_kernel_cells"] >= 0):
        raise ValueError("shock_kernel_cells must be non-negative.")
    if not (p["shock_kernel_width"] >= 0.0):
        raise ValueError("shock_kernel_width_Rsun must be non-negative.")


def _compute_csm_scales(p):
    x_max = p["x_max"]
    I_M_csm = _integral_power_law(1.0, x_max, 2.0 - p["s"])
    rho_0 = p["M_csm"] / (4.0 * PI * p["R_in"] ** 3 * I_M_csm)
    tau_in = p["kappa"] * rho_0 * p["R_in"]
    t_d = 3.0 * p["kappa"] * rho_0 * p["R_csm"] ** 2 / C_LIGHT

    zeta = (p["n"] - 3.0) * (3.0 - p["delta_inner"])
    zeta /= 4.0 * PI * (p["n"] - p["delta_inner"])
    f_v = 2.0 * (5.0 - p["delta_inner"]) * (p["n"] - 5.0)
    f_v /= (3.0 - p["delta_inner"]) * (p["n"] - 3.0)
    v_tr = np.sqrt(f_v * p["E_sn"] / p["M_ej"])
    v_ej_mean = np.sqrt(2.0 * p["E_sn"] / p["M_ej"])
    v_max = 3.0 * v_tr

    alpha = 1.0 / (p["n"] - 3.0)
    eta_mass = p["M_csm"] / p["M_ej"]
    lambda_ = (p["n"] - 3.0) / (p["n"] - p["s"])
    t_se = eta_mass ** alpha * p["R_csm"] / v_ej_mean
    t_in = p["R_in"] / v_max
    y_in = t_in / t_d
    w_tr = v_tr / v_max
    rho_ej_tr = zeta * p["M_ej"] / p["R_in"] ** 3
    A1 = rho_ej_tr / rho_0

    u0 = t_d * rho_0 * v_max ** 3 / (2.0 * p["R_in"])
    beta_int = 2.0 * x_max ** p["s"] / (3.0 * tau_in)
    L0 = 4.0 * PI * C_LIGHT * p["R_in"] * u0 / (3.0 * p["kappa"] * rho_0)

    return {
        "I_M_csm": I_M_csm,
        "rho_0": rho_0,
        "tau_in": tau_in,
        "t_d": t_d,
        "zeta": zeta,
        "f_v": f_v,
        "v_tr": v_tr,
        "v_ej_mean": v_ej_mean,
        "v_max": v_max,
        "alpha": alpha,
        "eta_mass": eta_mass,
        "lambda_": lambda_,
        "t_se": t_se,
        "t_in": t_in,
        "y_in": y_in,
        "w_tr": w_tr,
        "rho_ej_tr": rho_ej_tr,
        "A1": A1,
        "u0": u0,
        "beta_int": beta_int,
        "L0": L0,
    }


def _rho_csm_of_x(x, p, scales):
    x_arr = np.asarray(x, dtype=float)
    values = scales["rho_0"] * x_arr ** (-p["s"])
    return float(values) if np.ndim(x_arr) == 0 else values


def _f_rho_ej(x, y_sh, p, scales):
    x_arr = np.asarray(x, dtype=float)
    x_break = scales["w_tr"] * y_sh
    core = x_break ** (-3.0) * (x_arr / x_break) ** (-p["delta_inner"])
    env = x_break ** (-3.0) * (x_arr / x_break) ** (-p["n"])
    values = np.where(x_arr < x_break, core, env)
    return float(values) if np.ndim(x_arr) == 0 else values


def _shock_rhs(y_sh, state, p, scales):
    x_sh, z, w = state
    z_eff = max(float(z), 1.0e-12)
    rel_speed = x_sh / y_sh - w
    f_ej = _f_rho_ej(x_sh, y_sh, p, scales)

    dx_dysh = w
    dz_dysh = scales["A1"] * x_sh ** 2 * f_ej * rel_speed + x_sh ** (2.0 - p["s"]) * w
    dw_dysh = (
        scales["A1"] * x_sh ** 2 * f_ej * rel_speed ** 2
        - x_sh ** (2.0 - p["s"]) * w ** 2
    ) / z_eff
    return np.array([dx_dysh, dz_dysh, dw_dysh], dtype=float)


def _solve_shock_ode(p):
    scales = _compute_csm_scales(p)

    def rhs(y_sh, state):
        return _shock_rhs(y_sh, state, p, scales)

    def event_xmax(y_sh, state):
        del y_sh
        return float(state[0] - p["x_max"])

    event_xmax.terminal = True
    event_xmax.direction = 1.0

    initial_state = np.array([0.9999, 1.0e-3, 0.999], dtype=float)
    sol = solve_ivp(
        rhs,
        (1.0, 1.0e5),
        initial_state,
        method="BDF",
        events=event_xmax,
        dense_output=True,
        rtol=p["rtol_ode"],
        atol=p["atol_ode"],
        max_step=p["shock_max_step"],
    )
    if sol.status < 0:
        raise RuntimeError(f"Shock ODE failed: {sol.message}")
    if sol.t_events[0].size == 0:
        raise RuntimeError("Shock ODE did not reach x_sh = R_csm / R_in.")
    if sol.sol is None:
        raise RuntimeError("Shock ODE did not return dense output.")

    y_sh_end = float(sol.t_events[0][0])
    final_state = np.asarray(sol.sol(y_sh_end), dtype=float)
    y_end = float(scales["y_in"] * y_sh_end)
    t_end = float(scales["t_d"] * y_end)

    return {
        "solver": sol,
        "scales": scales,
        "y_sh_end": y_sh_end,
        "y_end": y_end,
        "t_end": t_end,
        "t_end_days": t_end / DAY,
        "x_end": float(final_state[0]),
        "z_end": float(final_state[1]),
        "w_end": float(final_state[2]),
    }


def _build_y_grid(y_start, y_end, y_max, n_steps):
    y_base = np.linspace(y_start, y_max, n_steps + 1)
    dy0 = float((y_max - y_start) / max(n_steps, 1))
    y_refine = y_end + dy0 * np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    y_grid = np.unique(np.sort(np.concatenate([y_base, np.clip(y_refine, y_start, y_max), [y_end]])))
    y_grid[0] = y_start
    y_grid[-1] = y_max
    return y_grid


def _build_expansion_profile(y_grid, shock, p, scales):
    y_end = float(shock["y_end"])
    y_sh_end = float(shock["y_sh_end"])
    w_end = max(float(shock["w_end"]), 1.0e-10)

    v_sh_end = scales["v_max"] * w_end
    t_sc = p["R_csm"] / v_sh_end
    cooling_rate = scales["t_d"] / t_sc

    expansion_factor = np.ones_like(y_grid, dtype=float)
    after_end = y_grid > y_end
    expansion_factor[after_end] = 1.0 + cooling_rate * (y_grid[after_end] - y_end)
    surface_beta = scales["beta_int"] * expansion_factor ** 2
    shock_active = y_grid <= (y_end + 1.0e-14)

    y_sh_grid = np.clip(y_grid / scales["y_in"], 1.0, y_sh_end)
    dense_state = np.asarray(shock["solver"].sol(y_sh_grid), dtype=float)
    x_sh = np.asarray(dense_state[0], dtype=float)
    z_sh = np.asarray(dense_state[1], dtype=float)
    w_sh = np.maximum(np.asarray(dense_state[2], dtype=float), 0.0)
    x_sh[~shock_active] = p["x_max"]
    z_sh[~shock_active] = float(shock["z_end"])
    w_sh[~shock_active] = w_end

    return expansion_factor, surface_beta, shock_active, x_sh, z_sh, w_sh


def _deposit_shock_source(x_grid, x_sh, w, p):
    dx = float(x_grid[1] - x_grid[0])
    source = np.zeros_like(x_grid, dtype=float)
    n_cells = x_grid.size - 1
    A_sh = p["eps_sh"] * x_sh ** (-p["s"]) * max(float(w), 0.0) ** 3
    if A_sh == 0.0:
        return source, A_sh

    if p["shock_kernel_width"] > 0.0:
        kernel_width_x = p["shock_kernel_width"] / p["R_in"]
        kernel_cells = max(3, int(np.ceil(6.0 * kernel_width_x / dx)))
        use_delta_kernel = False
    else:
        kernel_width_x = 0.0
        kernel_cells = int(p["shock_kernel_cells"])
        use_delta_kernel = kernel_cells <= 2

    if use_delta_kernel:
        if x_sh <= x_grid[0]:
            source[1] = A_sh / dx
        elif x_sh >= x_grid[-1]:
            source[n_cells - 1] = A_sh / dx
        else:
            j = int(np.searchsorted(x_grid, x_sh, side="right") - 1)
            j = int(np.clip(j, 0, n_cells - 1))
            theta = (x_sh - x_grid[j]) / dx
            source[j] += A_sh * (1.0 - theta) / dx
            source[j + 1] += A_sh * theta / dx
            if j == 0:
                source[1] += source[0]
                source[0] = 0.0
            if j + 1 == n_cells:
                source[n_cells - 1] += source[n_cells]
                source[n_cells] = 0.0
    elif n_cells <= 2:
        source[1] = A_sh / dx
    else:
        valid_start = 1
        valid_stop = n_cells - 1
        width = min(kernel_cells, valid_stop - valid_start + 1)
        center = int(np.argmin(np.abs(x_grid[valid_start : valid_stop + 1] - x_sh))) + valid_start
        left = center - width // 2
        right = left + width - 1
        if left < valid_start:
            right += valid_start - left
            left = valid_start
        if right > valid_stop:
            left -= right - valid_stop
            right = valid_stop
        indices = np.arange(left, right + 1)
        sigma_x = max(kernel_width_x, 0.5 * float(kernel_cells) * dx, 1.0e-300)
        weights = np.exp(-0.5 * ((x_grid[indices] - x_sh) / sigma_x) ** 2)
        weight_norm = float(np.sum(x_grid[indices] ** 2 * weights) * dx)
        if weight_norm <= 0.0 or not np.isfinite(weight_norm):
            source[center] = A_sh / dx
        else:
            source[indices] = A_sh * x_sh ** 2 * weights / weight_norm

    physical_integral = float(np.sum(x_grid ** 2 * source) * dx / max(x_sh ** 2, 1.0e-300))
    if physical_integral > 0.0 and np.isfinite(physical_integral):
        source *= A_sh / physical_integral
    return source, A_sh


def _precompute_sources(y_grid, x_grid, p, scales, shock_active, x_sh, w_sh):
    shock_source = np.zeros((y_grid.size, x_grid.size), dtype=float)
    A_sh = np.zeros(y_grid.size, dtype=float)
    L_sh_heat = np.zeros(y_grid.size, dtype=float)

    for i in range(y_grid.size):
        if not shock_active[i]:
            continue
        shock_source[i], A_sh[i] = _deposit_shock_source(x_grid, float(x_sh[i]), float(w_sh[i]), p)
        L_sh_heat[i] = (
            p["eps_sh"]
            * 2.0
            * PI
            * (p["R_in"] * x_sh[i]) ** 2
            * _rho_csm_of_x(x_sh[i], p, scales)
            * (scales["v_max"] * w_sh[i]) ** 3
        )

    return {"S_total": shock_source, "A_sh": A_sh, "L_sh_heat": L_sh_heat}


# -----------------------------------------------------------------------------
# CN diffusion loop (Numba JIT)
# -----------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def thomas_algorithm(a, b, c_up, d, c_prime, d_prime, x_out):
    n = len(d)
    c_prime[0] = c_up[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c_up[i] / denom
        else:
            c_prime[i] = 0.0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x_out[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = d_prime[i] - c_prime[i] * x_out[i + 1]


@numba.njit(fastmath=True, cache=True)
def _fast_pde_loop(
    dy_steps,
    expansion_vals,
    beta_vals,
    total_source,
    coeff_imh_base,
    coeff_iph_base,
    dx,
    Lfac,
    store_history,
):
    n_times, n_pts = total_source.shape
    n_inner = n_pts - 2

    e_now = np.zeros(n_pts, dtype=np.float64)
    e_next = np.zeros(n_pts, dtype=np.float64)
    L_bol = np.zeros(n_times, dtype=np.float64)

    if store_history:
        e_hist = np.zeros((n_times, n_pts), dtype=np.float64)
        e_hist[0, :] = e_now
    else:
        e_hist = np.zeros((1, 1), dtype=np.float64)

    a = np.zeros(n_pts, dtype=np.float64)
    b_diag = np.zeros(n_pts, dtype=np.float64)
    c_up = np.zeros(n_pts, dtype=np.float64)
    rhs = np.zeros(n_pts, dtype=np.float64)
    c_prime = np.zeros(n_pts, dtype=np.float64)
    d_prime = np.zeros(n_pts, dtype=np.float64)

    for n in range(1, n_times):
        dy = dy_steps[n - 1]
        expansion = expansion_vals[n]
        beta = beta_vals[n]

        a[:] = 0.0
        b_diag[:] = 0.0
        c_up[:] = 0.0
        rhs[:] = 0.0

        b_diag[0] = 1.0
        c_up[0] = -1.0

        for i in range(n_inner):
            idx = i + 1
            lower = -dy * expansion * coeff_imh_base[i]
            upper = -dy * expansion * coeff_iph_base[i]
            a[idx] = lower
            b_diag[idx] = 1.0 - lower - upper
            c_up[idx] = upper
            rhs[idx] = e_now[idx] + dy * total_source[n, idx]

        a[n_pts - 1] = -beta / dx
        b_diag[n_pts - 1] = 1.0 + beta / dx

        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)
        L_bol[n] = Lfac * (e_next[n_pts - 2] - e_next[n_pts - 1])
        if store_history:
            e_hist[n, :] = e_next

        e_now, e_next = e_next, e_now

    return L_bol, e_hist


def _photosphere(L_bol, expansion_factor, shock_active, p):
    del shock_active
    R_out = p["R_csm"] * expansion_factor
    L_pos = np.maximum(np.asarray(L_bol, dtype=float), 1.0e-300)

    T_surface = (L_pos / (4.0 * PI * SIGMA_SB * np.maximum(R_out, 1.0e-300) ** 2)) ** 0.25
    R_floor = np.sqrt(L_pos / (4.0 * PI * SIGMA_SB * p["T_floor"] ** 4))

    floor_active = T_surface < p["T_floor"]
    R_ph = R_out.copy()
    R_ph[floor_active] = R_floor[floor_active]
    T_eff = (L_pos / (4.0 * PI * SIGMA_SB * np.maximum(R_ph, 1.0e-300) ** 2)) ** 0.25
    return T_eff, R_ph, R_out


def _planck_specific_intensity_nu(nu_hz, temperature_k):
    temperature = np.maximum(np.asarray(temperature_k, dtype=float), 1.0e-30)
    x = H_PLANCK * nu_hz / (K_BOLTZ * temperature)
    x_clip = np.clip(x, 1.0e-12, 700.0)
    intensity = 2.0 * H_PLANCK * nu_hz ** 3 / C_LIGHT ** 2 / np.expm1(x_clip)
    return np.where(x > 700.0, 0.0, intensity)


class CSMModel:
    """
    Pure CSM-interaction light-curve model.

    Canonical theta order:
    (M_ej, E_sn, M_csm, R_csm_out, kappa, s, eps_sh, T_floor)

    Units:
    - M_csm, M_ej: solar mass
    - E_sn: 1e51 erg
    - R_csm_out: solar radius
    - kappa: cm^2 g^-1
    - s: CSM density power-law index
    - eps_sh: dimensionless

    Public API fixed defaults:
    - R_csm_in = 100 R_sun
    - n = 10
    - delta = 0

    Legacy shorter theta:
    (M_csm, M_ej, E_sn_51, R_in, R_csm, kappa, eps_sh)
    (M_csm, M_ej, E_sn_51, R_in, R_csm, kappa, eps_sh, s)
    """

    _warmup_theta = (5.0, 1.0, 1.0, 10000.0, 0.34, 2.0, 1.0, 5000.0)
    _warmup_kwargs = {"Nx": 20, "Ny": 40, "t_max_days": 20.0}

    def __init__(self, *, warmup=False):
        if warmup:
            self.warmup()

    def warmup(self, **kwargs):
        warmup_kwargs = dict(self._warmup_kwargs)
        warmup_kwargs.update(kwargs)
        self.calculate_light_curve(self._warmup_theta, **warmup_kwargs)
        return self

    def _params_from_theta(
        self,
        theta,
        Nx,
        Ny,
        t_max_days,
        s=2.0,
        n=DEFAULT_EJECTA_N,
        delta_inner=DEFAULT_EJECTA_DELTA,
        T_floor=5000.0,
        rtol_ode=1.0e-6,
        atol_ode=1.0e-8,
        shock_max_step=0.8,
        shock_kernel_cells=2,
        shock_kernel_width_Rsun=0.0,
    ):
        return _build_params(
            theta,
            Nx,
            Ny,
            t_max_days,
            s,
            n,
            delta_inner,
            T_floor,
            rtol_ode,
            atol_ode,
            shock_max_step,
            shock_kernel_cells,
            shock_kernel_width_Rsun,
        )

    def calculate_light_curve(self, theta, Nx=140, Ny=1000, t_max_days=150.0, return_full=False, **kwargs):
        p = self._params_from_theta(theta, Nx, Ny, t_max_days, **kwargs)
        shock = _solve_shock_ode(p)
        scales = shock["scales"]

        y_max = max(p["t_max_days"] * DAY / scales["t_d"], shock["y_end"])
        x_grid = np.linspace(1.0, p["x_max"], p["N_x"] + 1, dtype=float)
        y_grid = _build_y_grid(scales["y_in"], shock["y_end"], y_max, p["N_t"])
        t_s = y_grid * scales["t_d"]
        dx = float(x_grid[1] - x_grid[0])
        dy_steps = np.diff(y_grid).astype(float)

        (
            expansion_factor,
            surface_beta,
            shock_active,
            x_sh,
            z_sh,
            w_sh,
        ) = _build_expansion_profile(y_grid, shock, p, scales)

        sources = _precompute_sources(y_grid, x_grid, p, scales, shock_active, x_sh, w_sh)

        x_inner = x_grid[1:-1]
        xi_sq = (1.0 / p["x_max"]) ** 2
        dx_sq = dx ** 2
        coeff_imh_base = (x_inner - 0.5 * dx) ** (2.0 + p["s"]) / (xi_sq * x_inner ** 2 * dx_sq)
        coeff_iph_base = (x_inner + 0.5 * dx) ** (2.0 + p["s"]) / (xi_sq * x_inner ** 2 * dx_sq)
        Lfac = scales["L0"] * p["x_max"] ** (2.0 + p["s"]) / dx

        L_bol, e_hist = _fast_pde_loop(
            dy_steps,
            expansion_factor.astype(float),
            surface_beta.astype(float),
            sources["S_total"].astype(float),
            coeff_imh_base.astype(float),
            coeff_iph_base.astype(float),
            dx,
            Lfac,
            bool(return_full),
        )
        T_eff, R_ph, R_out = _photosphere(L_bol, expansion_factor, shock_active, p)

        out = slice(1, None)
        t_s_out = t_s[out]
        y_grid_out = y_grid[out]
        L_bol_out = L_bol[out]
        L_sh_heat_out = sources["L_sh_heat"][out]
        T_eff_out = T_eff[out]
        R_ph_out = R_ph[out]
        R_out_out = R_out[out]
        x_sh_out = x_sh[out]
        z_sh_out = z_sh[out]
        w_sh_out = w_sh[out]
        shock_active_out = shock_active[out]
        expansion_factor_out = expansion_factor[out]
        e_hist_out = e_hist[out] if bool(return_full) else e_hist

        if return_full:
            return {
                "theta": tuple(theta),
                "params": p,
                "scales": scales,
                "t_s": t_s_out,
                "t_day": t_s_out / DAY,
                "x_grid": x_grid,
                "y_grid": y_grid_out,
                "L_bol": L_bol_out,
                "L_sh_heat": L_sh_heat_out,
                "T_eff": T_eff_out,
                "R_ph": R_ph_out,
                "R_out": R_out_out,
                "x_sh": x_sh_out,
                "z_sh": z_sh_out,
                "w_sh": w_sh_out,
                "v_sh": scales["v_max"] * w_sh_out,
                "shock_active": shock_active_out,
                "shock_end_day": shock["t_end_days"],
                "expansion_factor": expansion_factor_out,
                "e_hist": e_hist_out,
            }

        return t_s_out, L_bol_out, T_eff_out, R_ph_out

    def calculate_dynamics(self, theta, Nx=140, Ny=1000, t_max_days=150.0, **kwargs):
        result = self.calculate_light_curve(theta, Nx=Nx, Ny=Ny, t_max_days=t_max_days, return_full=True, **kwargs)
        return result["t_s"], result["v_sh"], result["L_sh_heat"]

    def L_bol(self, t_obs, theta, z=0.0, **kwargs):
        t_s, L_series, _, _ = self.calculate_light_curve(theta, **kwargs)
        t_obs_days = np.asarray(t_obs, dtype=float)
        t_obs_grid_days = (t_s * (1.0 + z)) / DAY
        return interp_fit(
            t_obs_grid_days,
            np.maximum(np.asarray(L_series, dtype=float), 1.0e-300),
            t_obs_days,
            yscale="log10",
            fill="edge",
        )

    def M_ab(self, t_obs, theta, nu_obs, z, **kwargs):
        """Observed AB magnitude at observer-frame times in days."""
        nu_rest = nu_obs * (1.0 + z)
        DL_z = cosmo.luminosity_distance(z).to(u.cm).value

        t_s, L_bol_values, T_eff_values, R_ph = self.calculate_light_curve(theta, **kwargs)
        del L_bol_values
        t_obs_days = np.asarray(t_obs, dtype=float)
        t_obs_grid_days = (t_s * (1.0 + z)) / DAY

        B_nu = _planck_specific_intensity_nu(nu_rest, T_eff_values)
        L_nu = 4.0 * PI * PI * R_ph ** 2 * B_nu
        F_nu = ((1.0 + z) * L_nu) / (4.0 * PI * DL_z ** 2)
        M_ab_values = -2.5 * np.log10(np.maximum(F_nu, 1.0e-300)) - 48.6
        return np.interp(t_obs_days, t_obs_grid_days, M_ab_values, left=M_ab_values[0], right=M_ab_values[-1])
